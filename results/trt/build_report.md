# TensorRT Engine Build Report

No engine was built on this run.

the tensorrt python package is not installed (No module named 'tensorrt'). TensorRT engine builds need a NVIDIA gpu, the cuda runtime, and the tensorrt python package. None of those exist on Apple Silicon or on any cpu only host. Run this on a cuda machine through the gpu image this repository ships, with docker build -f docker/Dockerfile.tensorrt -t adrankbench-trt . and then docker run --gpus all adrankbench-trt. That image is built on nvcr.io/nvidia/tensorrt:25.01-py3, which carries TensorRT 10 and the cuda runtime already installed.

## Host

| Field | Value |
| --- | --- |
| Collected at (UTC) | 2026-08-28T00:34:04+00:00 |
| CPU | Apple M3 Pro |
| Logical cores | 12 |
| Platform | macOS-26.5.1-arm64-arm-64bit |
| Architecture | arm64 |
| Python | 3.12.7 (CPython) |
| NumPy | 2.5.2 |
| PyTorch | 2.8.0 |
| PyTorch cuda build | not available |
| PyTorch cuda available | False |
| ONNX Runtime | 1.22.0 |
| ONNX Runtime providers | CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider |
| OpenVINO | 2026.2.1-21919-ede283a88e3-releases/2026/2 |
| TensorRT | not available |
| GPU | not available |
| GPU driver | not available |
| CUDA driver runtime | not available |
| GPU memory | not available |
| GPU compute capability | not available |
| GPU peak memory bandwidth (estimated) | not available |
