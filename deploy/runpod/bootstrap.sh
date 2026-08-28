#!/usr/bin/env bash
#
# Prepare a RunPod pod to build TensorRT engines and run the inference sweep.
#
# Run this once on the pod, from the repository root, after the code has been
# synced up. It is idempotent, so running it twice is harmless.
#
# This script exists because the RunPod PyTorch image is close to what this
# project needs but not identical to it. The image already carries a CUDA 12.8
# build of torch, so torch is deliberately left alone. What the image does not
# carry is TensorRT, and that is the piece without which nothing in this
# repository will build an engine.
#
# Every fact this script relies on was measured on a real pod on 2026-08-28
# against the image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, which
# reported driver 570.172.08 and torch 2.8.0+cu128.
#
# Usage
#     bash deploy/runpod/bootstrap.sh
#     bash deploy/runpod/bootstrap.sh --trt-version 10.8.0.43

set -euo pipefail

TRT_VERSION="${TRT_VERSION:-10.8.0.43}"

while [ $# -gt 0 ]; do
    case "$1" in
        --trt-version) TRT_VERSION="$2"; shift 2 ;;
        -h|--help) sed -n "2,25p" "$0" | sed -e "s/^#\{0,1\} \{0,1\}//"; exit 0 ;;
        *) echo "unknown argument $1. Try --help." >&2; exit 2 ;;
    esac
done

say() { printf '\n=== %s ===\n' "$1"; }

# The CUDA toolkit is present on the image but nothing points at it. Without
# this line nvcc and ncu both look missing to any which based probe, which sends
# people chasing an installation problem that does not exist.
say "putting CUDA on PATH"
export PATH=/usr/local/cuda/bin:${PATH}
echo "PATH now includes /usr/local/cuda/bin"
nvcc --version 2>/dev/null | tail -2 || echo "nvcc not found, which is unexpected on this image"

say "GPU and driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is missing. This is not a GPU pod and nothing below will work." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
    --format=csv,noheader

# PEP 668 marks this image's python externally managed, so a bare pip install
# refuses to run. Everything below carries the override.
PIP="pip install --break-system-packages --no-cache-dir"

say "python packages already present"
python3 - <<'PY'
import importlib.metadata as md
for name in ("torch", "numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "tensorrt"):
    try:
        print(f"  {name:18s} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:18s} not installed")
PY

# torch is deliberately not touched. The image ships 2.8.0+cu128, which is
# already the CUDA 12.8 build this project wants, and reinstalling it would be a
# slow downgrade for no benefit.
say "installing the repository requirements without disturbing torch"
$PIP -r requirements.txt || {
    echo "requirements install failed. Retrying without the torch line." >&2
    grep -viE '^\s*torch([=<>]|$)' requirements.txt > /tmp/reqs_no_torch.txt
    $PIP -r /tmp/reqs_no_torch.txt
}

# onnxruntime and onnxruntime-gpu install the same import name and cannot
# coexist. requirements.txt pulls the cpu build, so it is removed before the gpu
# build goes in, otherwise the cpu wheel wins and the CUDA and TensorRT
# execution providers never appear.
say "swapping onnxruntime for the gpu build"
pip uninstall -y --break-system-packages onnxruntime 2>/dev/null || true
$PIP onnxruntime-gpu

say "installing NVML bindings"
# nvidia-ml-py is the maintained package and it provides the pynvml import name.
# The PyPI package literally called pynvml is deprecated and is not the one to
# install here.
$PIP nvidia-ml-py

say "installing TensorRT ${TRT_VERSION}"
# TensorRT is not on the RunPod PyTorch image. The plain PyPI name works on
# recent releases. Older or pinned builds sometimes only resolve through
# NVIDIA's own index, so that is the fallback rather than the default.
if ! $PIP "tensorrt==${TRT_VERSION}"; then
    echo "plain PyPI install failed. Falling back to the NVIDIA index." >&2
    $PIP --extra-index-url https://pypi.nvidia.com "tensorrt==${TRT_VERSION}"
fi

say "verification"
python3 - <<'PY'
import sys

ok = True

import torch
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device           {torch.cuda.get_device_name(0)}")
    print(f"  capability       {torch.cuda.get_device_capability(0)}")
else:
    print("  torch cannot see a cuda device, so every gpu lane will be skipped")
    ok = False

try:
    import tensorrt as trt
    print(f"  tensorrt         {trt.__version__}")
except Exception as exc:
    print(f"  tensorrt         FAILED to import ({exc})")
    ok = False

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  onnxruntime      {ort.__version__}")
    print(f"  providers        {providers}")
    for needed in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        if needed not in providers:
            print(f"  NOTE {needed} is absent, that backend will report not available")
except Exception as exc:
    print(f"  onnxruntime      FAILED to import ({exc})")
    ok = False

try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    watts = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    print(f"  nvml driver      {pynvml.nvmlSystemGetDriverVersion()}")
    print(f"  power draw       {watts:.1f} W")
except Exception as exc:
    print(f"  nvml             FAILED ({exc}). The perf per watt columns will be skipped.")

sys.exit(0 if ok else 1)
PY

say "done"
cat <<'EOF'
The pod is ready. Two notes before you record any number from it.

Nsight Compute hardware counters are blocked on RunPod and this cannot be fixed
from inside the pod. The ncu binary will attach and then fail at the counter
read with ERR_NVGPUCTRPERM. Nothing here depends on those counters. Nsight
Systems does not need that permission and is expected to work.

Every number produced on this pod is a rented cloud GPU number. Label it with
the exact card, the driver, and the TensorRT version printed above, and never
blend it with the Apple Silicon CPU figures from this same project.

Next.
    python3 scripts/build_trt_engines.py --models deepfm dcn --precisions fp32 fp16 int8
    python3 scripts/run_inference_benchmark.py --models deepfm dcn --batch-sizes 1 32 256 1024 4096
EOF
