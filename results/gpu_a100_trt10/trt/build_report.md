# TensorRT Engine Build Report

Every engine below was compiled on the gpu named in the host table, for the batch range in the optimization profile, from the ONNX graph exported by this same script. A serialized plan is tied to the gpu architecture, the driver, and the TensorRT version that produced it, so these files are not portable and have to be rebuilt on the machine that will serve them.

The optimization profile covers batches from 1 to 4096 and is tuned for 256. The optimum is not the maximum on purpose, because tuning only for the largest batch would leave the online serving case at batch one running on a kernel that was chosen for a workload it never sees.

## Engines

| Model | Precision | Engine | Build time (s) | Size on disk | Status |
| --- | --- | --- | --- | --- | --- |
| DeepFM | fp32 | deepfm_fp32_bs4096.engine | 0.0 | 86.8 MB | built |
| DeepFM | fp16 | deepfm_fp16_bs4096.engine | 0.0 | 86.8 MB | built |
| DeepFM | int8 | deepfm_int8_bs4096.engine | not available | not available | not available |

Build time is a deploy time cost and it is reported here rather than anywhere in the latency tables. An engine is built once and then scores batches for as long as the model is in production, so folding the build into a serving number would misrepresent both.

## Engines that were not built

- DeepFM int8. the onnx parser rejected the graph. In node 10 with name: deep_head.bias_DequantizeLinear and operator: DequantizeLinear (parseNode): INVALID_NODE: Invalid Node - deep_head.bias_DequantizeLinear
IDequantizeLayer::setPrecision: Error Code 3: API Usage Error (Parameter check failed, condition: isQuantized(dataType). A DequantizeLayer can only run in DataType::kINT8, DataType::kFP8, DataType::kFP4 or DataType::kINT4precision)
ITensor::getDimensions: Error Code 3: API Usage Error (deep_head.bias_DequantizeLinear: only activation types allowed as input to this layer.)

## Builder warnings

- DeepFM int8. warning ModelImporter.cpp:459: Make sure input cat has Int64 binding.
- DeepFM int8. warning onnxOpImporters.cpp:1499: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
- DeepFM int8. error IDequantizeLayer::setPrecision: Error Code 3: API Usage Error (Parameter check failed, condition: isQuantized(dataType). A DequantizeLayer can only run in DataType::kINT8, DataType::kFP8, DataType::kFP4 or DataType::kINT4precision)
- DeepFM int8. error ITensor::getDimensions: Error Code 3: API Usage Error (deep_head.bias_DequantizeLinear: only activation types allowed as input to this layer.)
- DeepFM int8. error In node 10 with name: deep_head.bias_DequantizeLinear and operator: DequantizeLinear (parseNode): INVALID_NODE: Invalid Node - deep_head.bias_DequantizeLinear
IDequantizeLayer::setPrecision: Error Code 3: API Usage Error (Parameter check failed, condition: isQuantized(dataType). A DequantizeLayer can only run in DataType::kINT8, DataType::kFP8, DataType::kFP4 or DataType::kINT4precision)
ITensor::getDimensions: Error Code 3: API Usage Error (deep_head.bias_DequantizeLinear: only activation types allowed as input to this layer.)

## INT8 calibration

Calibration rows are drawn from the validation split and never from the test split, so the INT8 accuracy the benchmark reports is measured on rows the quantizer has not seen. The scales are written to a cache file next to the engine, and a later build reads that cache instead of calibrating again, which is what makes an INT8 engine reproducible.

| Model | Cache | Batches | Batch size | Cache size |
| --- | --- | --- | --- | --- |
| DeepFM | deepfm_int8_calibration.cache | 0 | 256 | not available |

## Host

| Field | Value |
| --- | --- |
| Collected at (UTC) | 2026-08-28T05:06:30+00:00 |
| CPU | AMD EPYC 7742 64-Core Processor |
| Logical cores | 128 |
| Platform | Linux-6.8.0-59-generic-x86_64-with-glibc2.39 |
| Architecture | x86_64 |
| Python | 3.12.3 (CPython) |
| NumPy | 2.1.2 |
| PyTorch | 2.8.0+cu128 |
| PyTorch cuda build | 12.8 |
| PyTorch cuda available | True |
| ONNX Runtime | 1.29.0 |
| ONNX Runtime providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| OpenVINO | not available |
| TensorRT | 10.8.0.43 |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU driver | 570.195.03 |
| CUDA driver runtime | 12.8 |
| GPU memory | 80.0 GB |
| GPU compute capability | 8.0 |
| GPU peak memory bandwidth (estimated) | 2039 GB/s |
