# TensorRT Engine Build Report

Every engine below was compiled on the gpu named in the host table, for the batch range in the optimization profile, from the ONNX graph exported by this same script. A serialized plan is tied to the gpu architecture, the driver, and the TensorRT version that produced it, so these files are not portable and have to be rebuilt on the machine that will serve them.

The optimization profile covers batches from 1 to 4096 and is tuned for 256. The optimum is not the maximum on purpose, because tuning only for the largest batch would leave the online serving case at batch one running on a kernel that was chosen for a workload it never sees.

## Engines

| Model | Precision | Engine | Build time (s) | Size on disk | Status |
| --- | --- | --- | --- | --- | --- |
| DeepFM | int8 | deepfm_int8_bs4096.engine | 25.4 | 106.2 MB | built |

Build time is a deploy time cost and it is reported here rather than anywhere in the latency tables. An engine is built once and then scores batches for as long as the model is in production, so folding the build into a serving number would misrepresent both.

## Builder warnings

- DeepFM int8. warning ModelImporter.cpp:826: Make sure input cat has Int64 binding.
- DeepFM int8. built as a strongly typed network from an explicitly quantized qdq graph, because this TensorRT version removed the implicit quantization api

## INT8 calibration

Calibration rows are drawn from the validation split and never from the test split, so the INT8 accuracy the benchmark reports is measured on rows the quantizer has not seen. The scales are written to a cache file next to the engine, and a later build reads that cache instead of calibrating again, which is what makes an INT8 engine reproducible.

| Model | Cache | Batches | Batch size | Cache size |
| --- | --- | --- | --- | --- |
| DeepFM | deepfm_int8_calibration.cache | 0 | 256 | not available |

## Host

| Field | Value |
| --- | --- |
| Collected at (UTC) | 2026-08-28T06:35:53+00:00 |
| CPU | AMD EPYC 7742 64-Core Processor |
| Logical cores | 128 |
| Platform | Linux-6.8.0-124-generic-x86_64-with-glibc2.39 |
| Architecture | x86_64 |
| Python | 3.12.3 (CPython) |
| NumPy | 2.1.2 |
| PyTorch | 2.8.0+cu128 |
| PyTorch cuda build | 12.8 |
| PyTorch cuda available | True |
| ONNX Runtime | 1.29.0 |
| ONNX Runtime providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| OpenVINO | not available |
| TensorRT | 11.2.1.2 |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU driver | 580.159.04 |
| CUDA driver runtime | 13.0 |
| GPU memory | 80.0 GB |
| GPU compute capability | 8.0 |
| GPU peak memory bandwidth (estimated) | 2039 GB/s |
