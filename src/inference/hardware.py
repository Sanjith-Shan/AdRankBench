"""Hardware and software provenance for the inference benchmark.

An inference number without the machine it came from is not a result, it is a
rumour. This module collects one record that names the cpu, the operating
system, the python and library versions, and when a NVIDIA gpu is present its
name, driver version, cuda runtime version, TensorRT version, and total memory.
The record is embedded in every report and every json artifact so no measurement
in this project is ever printed unlabeled.

The gpu probe tries three sources in order. It first tries pynvml, the NVIDIA
management library binding, which gives the richest record. It then falls back
to parsing the output of the nvidia-smi command line tool. It finally falls back
to a record that says not available and states the reason. Nothing here raises
on a machine with no gpu, which is the whole point of the module. On the Apple
Silicon development machine this project is written on, every gpu field reads
not available and the reason names the missing driver.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.inference.common import NOT_AVAILABLE, human_bytes


def _run_command(args: List[str], timeout: float = 5.0) -> Optional[str]:
    """Run a short command and return its stripped stdout, or None on any failure.

    Every failure mode collapses to None. A missing binary, a non zero exit, a
    timeout, and a permissions error are all the same thing to the caller, which
    is that this probe cannot answer.
    """
    if not args or shutil.which(args[0]) is None:
        return None
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:  # noqa: BLE001 any failure means the probe cannot answer
        return None
    if completed.returncode != 0:
        return None
    out = (completed.stdout or "").strip()
    return out or None


def cpu_model() -> str:
    """Return a readable cpu model string for the host.

    macOS answers through sysctl, Linux through the model name line in
    /proc/cpuinfo, and anything else falls back to the platform processor
    string. A blank answer becomes the not available marker rather than an
    empty cell.
    """
    system = platform.system()
    if system == "Darwin":
        brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    fallback = platform.processor() or platform.machine()
    return fallback or NOT_AVAILABLE


def _version_of(module_name: str, attr: str = "__version__") -> str:
    """Import a module and return its version string, or the not available marker."""
    try:
        module = __import__(module_name)
    except Exception:  # noqa: BLE001 a missing or broken package is not an error here
        return NOT_AVAILABLE
    if module is None:
        return NOT_AVAILABLE
    value = getattr(module, attr, None)
    if value is None:
        return NOT_AVAILABLE
    return str(value)


def torch_record() -> Dict[str, Any]:
    """Describe the installed torch build and the accelerators it can see."""
    record: Dict[str, Any] = {
        "version": NOT_AVAILABLE,
        "cuda_build_version": NOT_AVAILABLE,
        "cudnn_version": NOT_AVAILABLE,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "device_names": [],
    }
    try:
        import torch
    except Exception:  # noqa: BLE001 torch missing is reported, not raised
        return record

    record["version"] = str(torch.__version__)
    record["cuda_build_version"] = str(getattr(torch.version, "cuda", None) or NOT_AVAILABLE)
    try:
        record["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 a broken driver must not crash the probe
        record["cuda_available"] = False
    try:
        mps = getattr(torch.backends, "mps", None)
        record["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception:  # noqa: BLE001
        record["mps_available"] = False
    if record["cuda_available"]:
        try:
            record["device_count"] = int(torch.cuda.device_count())
            record["device_names"] = [
                str(torch.cuda.get_device_name(i)) for i in range(record["device_count"])
            ]
            cudnn = getattr(torch.backends, "cudnn", None)
            if cudnn is not None and cudnn.is_available():
                record["cudnn_version"] = str(cudnn.version())
        except Exception:  # noqa: BLE001
            pass
    return record


def onnxruntime_record() -> Dict[str, Any]:
    """Describe the installed ONNX Runtime and the execution providers it offers."""
    record: Dict[str, Any] = {
        "version": NOT_AVAILABLE,
        "available_providers": [],
        "device": NOT_AVAILABLE,
    }
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        return record
    record["version"] = str(ort.__version__)
    try:
        record["available_providers"] = [str(p) for p in ort.get_available_providers()]
    except Exception:  # noqa: BLE001
        record["available_providers"] = []
    try:
        record["device"] = str(ort.get_device())
    except Exception:  # noqa: BLE001
        record["device"] = NOT_AVAILABLE
    return record


def tensorrt_record() -> Dict[str, Any]:
    """Describe the installed TensorRT python package.

    Importing tensorrt on a machine with no NVIDIA driver raises, so the import
    is wrapped and the failure text is carried into the record as the reason.
    """
    record: Dict[str, Any] = {
        "version": NOT_AVAILABLE,
        "available": False,
        "reason": "the tensorrt python package is not installed",
    }
    try:
        import tensorrt as trt
    except ImportError as exc:
        record["reason"] = f"the tensorrt python package is not installed ({exc})"
        return record
    except Exception as exc:  # noqa: BLE001 a driver mismatch raises here too
        record["reason"] = f"tensorrt failed to import ({exc})"
        return record
    record["version"] = str(getattr(trt, "__version__", NOT_AVAILABLE))
    record["available"] = True
    record["reason"] = ""
    return record


def _nvml_gpu_record(device_index: int = 0) -> Optional[Dict[str, Any]]:
    """Probe the gpu through pynvml, or return None when that is not possible."""
    try:
        import pynvml
    except Exception:  # noqa: BLE001 the binding is optional
        return None

    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 no driver on this host
        return None

    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    record: Dict[str, Any] = {"source": "pynvml", "index": device_index}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        record["name"] = _text(pynvml.nvmlDeviceGetName(handle))
        try:
            record["driver_version"] = _text(pynvml.nvmlSystemGetDriverVersion())
        except Exception:  # noqa: BLE001
            record["driver_version"] = NOT_AVAILABLE
        try:
            raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
            record["cuda_driver_version"] = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:  # noqa: BLE001
            record["cuda_driver_version"] = NOT_AVAILABLE
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            record["total_memory_bytes"] = int(mem.total)
        except Exception:  # noqa: BLE001
            record["total_memory_bytes"] = None
        try:
            record["memory_bus_width_bits"] = int(
                pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            )
        except Exception:  # noqa: BLE001 not every binding version exposes this
            record["memory_bus_width_bits"] = None
        try:
            record["max_memory_clock_mhz"] = int(
                pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            )
        except Exception:  # noqa: BLE001
            record["max_memory_clock_mhz"] = None
        try:
            record["max_sm_clock_mhz"] = int(
                pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
        except Exception:  # noqa: BLE001
            record["max_sm_clock_mhz"] = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            record["compute_capability"] = f"{int(major)}.{int(minor)}"
        except Exception:  # noqa: BLE001
            record["compute_capability"] = NOT_AVAILABLE
        try:
            record["power_limit_watts"] = (
                float(pynvml.nvmlDeviceGetEnforcedPowerManagementLimit(handle)) / 1000.0
            )
        except Exception:  # noqa: BLE001
            record["power_limit_watts"] = None
    except Exception:  # noqa: BLE001 a partial probe is worse than no probe
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
        return None

    try:
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass

    record["available"] = True
    record["reason"] = ""
    record["peak_memory_bandwidth_gb_s"] = estimate_peak_bandwidth_gb_s(record)
    return record


def _smi_gpu_record(device_index: int = 0) -> Optional[Dict[str, Any]]:
    """Probe the gpu by parsing nvidia-smi, the fallback when pynvml is absent."""
    query = "name,driver_version,memory.total,power.limit"
    out = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ]
    )
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 3:
        return None

    def _number(text: str) -> Optional[float]:
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    total_mib = _number(parts[2])
    power_limit = _number(parts[3]) if len(parts) > 3 else None
    return {
        "source": "nvidia-smi",
        "index": device_index,
        "available": True,
        "reason": "",
        "name": parts[0] or NOT_AVAILABLE,
        "driver_version": parts[1] or NOT_AVAILABLE,
        "cuda_driver_version": NOT_AVAILABLE,
        "total_memory_bytes": int(total_mib * 1024 * 1024) if total_mib else None,
        "memory_bus_width_bits": None,
        "max_memory_clock_mhz": None,
        "max_sm_clock_mhz": None,
        "compute_capability": NOT_AVAILABLE,
        "power_limit_watts": power_limit,
        "peak_memory_bandwidth_gb_s": None,
    }


def estimate_peak_bandwidth_gb_s(record: Dict[str, Any]) -> Optional[float]:
    """Estimate peak device memory bandwidth from the NVML clock and bus width.

    The estimate is memory clock times bus width times two for the double data
    rate transfer, converted to gigabytes per second. NVML reports the memory
    clock in a way that already folds in part of the data rate on some GDDR and
    HBM parts, so this figure is an estimate and the report always labels it as
    one. It is good enough to say whether a kernel is close to the roof or an
    order of magnitude below it, which is the question the analysis asks.
    """
    clock_mhz = record.get("max_memory_clock_mhz")
    bus_bits = record.get("memory_bus_width_bits")
    if not clock_mhz or not bus_bits:
        return None
    bytes_per_second = float(clock_mhz) * 1e6 * 2.0 * (float(bus_bits) / 8.0)
    return bytes_per_second / 1e9


def probe_gpu(device_index: int = 0) -> Dict[str, Any]:
    """Return the gpu record from the best available source.

    The order is pynvml, then nvidia-smi, then a record that says not available
    and names the reason. This function never raises.
    """
    record = _nvml_gpu_record(device_index)
    if record is not None:
        return record
    record = _smi_gpu_record(device_index)
    if record is not None:
        return record

    if platform.system() == "Darwin":
        reason = (
            "this host is macOS, where there is no NVIDIA driver and no cuda "
            "runtime, so every gpu measurement is unavailable by construction"
        )
    else:
        reason = (
            "neither pynvml nor nvidia-smi could describe a gpu on this host, "
            "so there is no NVIDIA driver visible to this process"
        )
    return {
        "source": NOT_AVAILABLE,
        "index": device_index,
        "available": False,
        "reason": reason,
        "name": NOT_AVAILABLE,
        "driver_version": NOT_AVAILABLE,
        "cuda_driver_version": NOT_AVAILABLE,
        "total_memory_bytes": None,
        "memory_bus_width_bits": None,
        "max_memory_clock_mhz": None,
        "max_sm_clock_mhz": None,
        "compute_capability": NOT_AVAILABLE,
        "power_limit_watts": None,
        "peak_memory_bandwidth_gb_s": None,
    }


def collect_hardware_record(device_index: int = 0) -> Dict[str, Any]:
    """Collect the full provenance record embedded in every report and artifact.

    The record has four blocks. The host block names the machine and the
    operating system. The python block names the interpreter. The libraries
    block names every runtime version that could change a number. The gpu block
    names the accelerator or says why there is not one.
    """
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "cpu_model": cpu_model(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or NOT_AVAILABLE,
            "logical_cores": _logical_cores(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "libraries": {
            "numpy": _version_of("numpy"),
            "torch": torch_record(),
            "onnxruntime": onnxruntime_record(),
            "openvino": _version_of("openvino"),
            "tensorrt": tensorrt_record(),
            "pynvml": _version_of("pynvml"),
        },
        "gpu": probe_gpu(device_index),
    }


def _logical_cores() -> Any:
    """Return the logical core count, or the not available marker."""
    try:
        import os

        count = os.cpu_count()
    except Exception:  # noqa: BLE001
        return NOT_AVAILABLE
    return int(count) if count else NOT_AVAILABLE


def cpu_lane_label(record: Dict[str, Any]) -> str:
    """Return the short label used to tag every cpu measurement in a report."""
    host = record.get("host", {})
    return f"{host.get('cpu_model', NOT_AVAILABLE)} ({host.get('machine', NOT_AVAILABLE)})"


def gpu_lane_label(record: Dict[str, Any]) -> str:
    """Return the short label used to tag every gpu measurement in a report."""
    gpu = record.get("gpu", {})
    if not gpu.get("available"):
        return NOT_AVAILABLE
    name = gpu.get("name", NOT_AVAILABLE)
    driver = gpu.get("driver_version", NOT_AVAILABLE)
    return f"{name} (driver {driver})"


def gpu_unavailable_reason(record: Dict[str, Any]) -> str:
    """Return the sentence a report prints wherever a gpu number is missing."""
    gpu = record.get("gpu", {})
    if gpu.get("available"):
        return ""
    reason = gpu.get("reason") or "no NVIDIA gpu was visible to this process"
    trt = record.get("libraries", {}).get("tensorrt", {})
    trt_reason = trt.get("reason") or ""
    if trt_reason:
        return f"{reason}, and {trt_reason}"
    return reason


def hardware_markdown_lines(record: Dict[str, Any]) -> List[str]:
    """Render the provenance record as markdown table rows for a report."""
    host = record.get("host", {})
    py = record.get("python", {})
    libs = record.get("libraries", {})
    torch_info = libs.get("torch", {})
    ort_info = libs.get("onnxruntime", {})
    trt_info = libs.get("tensorrt", {})
    gpu = record.get("gpu", {})

    providers = ort_info.get("available_providers") or []
    provider_text = ", ".join(providers) if providers else NOT_AVAILABLE

    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Collected at (UTC) | {record.get('collected_at_utc', NOT_AVAILABLE)} |",
        f"| CPU | {host.get('cpu_model', NOT_AVAILABLE)} |",
        f"| Logical cores | {host.get('logical_cores', NOT_AVAILABLE)} |",
        f"| Platform | {host.get('platform', NOT_AVAILABLE)} |",
        f"| Architecture | {host.get('machine', NOT_AVAILABLE)} |",
        f"| Python | {py.get('version', NOT_AVAILABLE)} ({py.get('implementation', '')}) |",
        f"| NumPy | {libs.get('numpy', NOT_AVAILABLE)} |",
        f"| PyTorch | {torch_info.get('version', NOT_AVAILABLE)} |",
        f"| PyTorch cuda build | {torch_info.get('cuda_build_version', NOT_AVAILABLE)} |",
        f"| PyTorch cuda available | {bool(torch_info.get('cuda_available', False))} |",
        f"| ONNX Runtime | {ort_info.get('version', NOT_AVAILABLE)} |",
        f"| ONNX Runtime providers | {provider_text} |",
        f"| OpenVINO | {libs.get('openvino', NOT_AVAILABLE)} |",
        f"| TensorRT | {trt_info.get('version', NOT_AVAILABLE)} |",
        f"| GPU | {gpu.get('name', NOT_AVAILABLE)} |",
        f"| GPU driver | {gpu.get('driver_version', NOT_AVAILABLE)} |",
        f"| CUDA driver runtime | {gpu.get('cuda_driver_version', NOT_AVAILABLE)} |",
        f"| GPU memory | {human_bytes(gpu.get('total_memory_bytes'))} |",
        f"| GPU compute capability | {gpu.get('compute_capability', NOT_AVAILABLE)} |",
    ]
    peak_bw = gpu.get("peak_memory_bandwidth_gb_s")
    if peak_bw:
        lines.append(
            f"| GPU peak memory bandwidth (estimated) | {peak_bw:.0f} GB/s |"
        )
    else:
        lines.append(f"| GPU peak memory bandwidth (estimated) | {NOT_AVAILABLE} |")
    return lines


def print_hardware_record(record: Dict[str, Any]) -> None:
    """Print a compact one screen summary of the provenance record to stdout."""
    host = record.get("host", {})
    gpu = record.get("gpu", {})
    print("hardware record")
    print(f"  cpu       {host.get('cpu_model', NOT_AVAILABLE)}")
    print(f"  platform  {host.get('platform', NOT_AVAILABLE)}")
    print(f"  python    {record.get('python', {}).get('version', NOT_AVAILABLE)}")
    libs = record.get("libraries", {})
    print(f"  torch     {libs.get('torch', {}).get('version', NOT_AVAILABLE)}")
    print(f"  ort       {libs.get('onnxruntime', {}).get('version', NOT_AVAILABLE)}")
    print(f"  openvino  {libs.get('openvino', NOT_AVAILABLE)}")
    print(f"  tensorrt  {libs.get('tensorrt', {}).get('version', NOT_AVAILABLE)}")
    if gpu.get("available"):
        print(f"  gpu       {gpu.get('name')} with {human_bytes(gpu.get('total_memory_bytes'))}")
    else:
        print(f"  gpu       {NOT_AVAILABLE}, {gpu_unavailable_reason(record)}")


def cuda_lane_ready() -> Tuple[bool, str]:
    """Return whether a cuda lane can run at all, and the reason when it cannot.

    This is the single gate the backend registry consults before it even tries
    to construct a gpu backend. It keeps the failure message identical across
    every gpu runtime rather than letting each one invent its own wording.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return False, f"torch is not importable ({exc}), so no cuda lane can run"
    try:
        if not torch.cuda.is_available():
            return False, (
                "torch reports no cuda device on this host, so every gpu lane is "
                "unavailable"
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"the cuda probe failed ({exc}), so every gpu lane is unavailable"
    return True, ""
