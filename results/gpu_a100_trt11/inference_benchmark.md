# Inference Optimization Benchmark

This report compares every serving backend this project supports, running the same trained weights on the same held out test rows. The question it answers is how much serving latency drops when a trained ranker is exported and run through an inference optimized runtime instead of raw PyTorch, and what that costs in accuracy when the runtime also drops the numeric precision.

The sweep covers DeepFM across 9 backends that ran, at the batch sizes 1, 256, 1024, 4096. It scores 10000 held out test rows per cell with seed 42, and it times 5 batches over 5 passes after 5 warmup batches.

Batch one and the largest batch are in the sweep for different reasons. Batch one is the online serving case, where a single ad request is scored on its own inside a few milliseconds and there is no other work to hide the latency behind. The largest batch is the offline scoring case, where a whole candidate pool is scored in bulk and the only number that matters is throughput. A runtime can win one of those and lose the other, so the report shows the full curve.

## Hardware and software provenance

Every number below was produced on this machine with these library versions. An inference measurement without its hardware is not a result, so this table is embedded in the markdown report and in the json artifact next to every raw measurement.

| Field | Value |
| --- | --- |
| Collected at (UTC) | 2026-08-28T07:00:46+00:00 |
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

## Two lanes, never blended

The cpu lane ran on AMD EPYC 7742 64-Core Processor (x86_64). It holds PyTorch eager mode, ONNX Runtime on the cpu execution provider, and OpenVINO. OpenVINO is Intel's cpu inference toolkit and it belongs to this lane only. It is not a like for like comparison against any gpu backend and it is never reported as one, because a cpu runtime and a gpu runtime answer different deployment questions and putting their numbers in the same ranking would be meaningless.

The gpu lane ran on NVIDIA A100-SXM4-80GB (driver 580.159.04). It holds PyTorch eager on cuda, ONNX Runtime on the cuda and TensorRT execution providers, and the natively built TensorRT engines. Every timing in this lane is taken with the cuda stream synchronized, so it measures completed device work rather than how fast the host filled a queue.

## Summary at batch 1024

This is the headline table. Latency is the mean wall time per batch, p50 and p99 are the median and the tail, throughput is the steady state samples per second, and AUC and LogLoss are measured over every held out test row rather than over the timing subset.

| Model | Backend | Lane | Hardware | Latency (ms/batch) | p50 (ms) | p99 (ms) | Throughput (samples/s) | AUC | LogLoss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | PyTorch eager (CPU, fp32) | cpu | AMD EPYC 7742 64-Core Processor (x86_64) | 1899.723 | 1900.986 | 2262.788 | 539 | 0.7731 | 0.4796 |
| DeepFM | ONNX Runtime (CPU provider, fp32) | cpu | AMD EPYC 7742 64-Core Processor (x86_64) | 75.651 | 94.439 | 193.987 | 13,536 | 0.7731 | 0.4796 |
| DeepFM | PyTorch eager (CUDA, fp32) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 4.609 | 0.823 | 71.148 | 222,178 | 0.7731 | 0.4796 |
| DeepFM | PyTorch eager (CUDA, fp16 autocast) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 2.625 | 0.989 | 31.938 | 390,076 | 0.7731 | 0.4796 |
| DeepFM | ONNX Runtime (CUDA provider, fp32) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 131.936 | 105.330 | 196.969 | 7,761 | 0.7731 | 0.4796 |
| DeepFM | ONNX Runtime (TensorRT provider, fp16) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 148.062 | 102.111 | 380.043 | 6,916 | 0.7731 | 0.4796 |
| DeepFM | TensorRT native engine (fp32) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 0.375 | 0.376 | 0.386 | 2,728,896 | 0.7731 | 0.4796 |
| DeepFM | TensorRT native engine (fp16) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 0.362 | 0.361 | 0.383 | 2,825,700 | 0.7731 | 0.4796 |
| DeepFM | TensorRT native engine (int8) | gpu | NVIDIA A100-SXM4-80GB (driver 580.159.04) | 0.349 | 0.347 | 0.358 | 2,936,026 | 0.7734 | 0.4782 |

![Inference latency by backend](inference_latency.png)

## Backends that did not run

Each of these was asked for and could not be built. The reason is the exact one the registry reported, not a summary of it.

| Model | Backend | Lane | Reason |
| --- | --- | --- | --- |
| DeepFM | OpenVINO (CPU, fp32) | cpu | openvino is not installed (No module named 'openvino'), so this backend was skipped |
| DeepFM | ONNX Runtime (TensorRT provider, int8) | gpu | the int8 TensorRT provider needs a calibration table and none exists at results/trt/deepfm_int8_calibration.cache. Run scripts/build_trt_engines.py first so the scales are written. |

## Full sweep

One row per model, backend, and batch size. Engine build time is a deploy time cost and it is reported in its own column so it is never confused with a serving number.

| Model | Backend | Precision | Lane | Batch | Rows timed per batch | Mean (ms) | p50 (ms) | p99 (ms) | Throughput (samples/s) | AUC | LogLoss | Build (s) | Peak GPU memory | Model on disk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | PyTorch | fp32 | cpu | 1 | 1 | 0.316 | 0.315 | 0.327 | 3,166 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | PyTorch | fp32 | cpu | 256 | 256 | 1328.236 | 1306.267 | 1573.054 | 193 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | PyTorch | fp32 | cpu | 1024 | 1024 | 1899.723 | 1900.986 | 2262.788 | 539 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | PyTorch | fp32 | cpu | 4096 | 4096 | 2639.183 | 2651.168 | 2895.372 | 1,552 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | ONNX Runtime | fp32 | cpu | 1 | 1 | 0.077 | 0.076 | 0.084 | 13,034 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | ONNX Runtime | fp32 | cpu | 256 | 256 | 51.822 | 5.489 | 245.061 | 4,940 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | ONNX Runtime | fp32 | cpu | 1024 | 1024 | 75.651 | 94.439 | 193.987 | 13,536 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | ONNX Runtime | fp32 | cpu | 4096 | 4096 | 290.341 | 252.323 | 404.003 | 14,108 | 0.7731 | 0.4796 | not available | not available | 82.5 MB |
| DeepFM | PyTorch CUDA fp32 | fp32 | gpu | 1 | 1 | 0.729 | 0.731 | 0.739 | 1,371 | 0.7731 | 0.4796 | not available | 1.7 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp32 | fp32 | gpu | 256 | 256 | 0.764 | 0.764 | 0.780 | 335,233 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp32 | fp32 | gpu | 1024 | 1024 | 4.609 | 0.823 | 71.148 | 222,178 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp32 | fp32 | gpu | 4096 | 4096 | 0.967 | 0.960 | 1.113 | 4,234,491 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp16 | fp16 | gpu | 1 | 1 | 0.901 | 0.899 | 0.935 | 1,110 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp16 | fp16 | gpu | 256 | 256 | 0.943 | 0.940 | 0.968 | 271,477 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp16 | fp16 | gpu | 1024 | 1024 | 2.625 | 0.989 | 31.938 | 390,076 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | PyTorch CUDA fp16 | fp16 | gpu | 4096 | 4096 | 1.121 | 1.125 | 1.161 | 3,653,345 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT CUDA | fp32 | gpu | 1 | 1 | 0.075 | 0.074 | 0.083 | 13,286 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT CUDA | fp32 | gpu | 256 | 256 | 56.020 | 82.228 | 127.025 | 4,570 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT CUDA | fp32 | gpu | 1024 | 1024 | 131.936 | 105.330 | 196.969 | 7,761 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT CUDA | fp32 | gpu | 4096 | 4096 | 330.634 | 302.128 | 487.135 | 12,388 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT TRT fp16 | fp16 | gpu | 1 | 1 | 0.074 | 0.073 | 0.084 | 13,490 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT TRT fp16 | fp16 | gpu | 256 | 256 | 55.972 | 5.994 | 193.766 | 4,574 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT TRT fp16 | fp16 | gpu | 1024 | 1024 | 148.062 | 102.111 | 380.043 | 6,916 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | ORT TRT fp16 | fp16 | gpu | 4096 | 4096 | 440.190 | 405.544 | 591.430 | 9,305 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.5 MB |
| DeepFM | TensorRT fp32 | fp32 | gpu | 1 | 1 | 0.216 | 0.215 | 0.225 | 4,629 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.6 MB |
| DeepFM | TensorRT fp32 | fp32 | gpu | 256 | 256 | 0.266 | 0.263 | 0.298 | 961,029 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.6 MB |
| DeepFM | TensorRT fp32 | fp32 | gpu | 1024 | 1024 | 0.375 | 0.376 | 0.386 | 2,728,896 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.6 MB |
| DeepFM | TensorRT fp32 | fp32 | gpu | 4096 | 4096 | 1.191 | 0.920 | 2.715 | 3,440,149 | 0.7731 | 0.4796 | not available | 1.8 GB | 82.6 MB |
| DeepFM | TensorRT fp16 | fp16 | gpu | 1 | 1 | 0.211 | 0.210 | 0.225 | 4,732 | 0.7731 | 0.4796 | not available | 1.8 GB | 41.4 MB |
| DeepFM | TensorRT fp16 | fp16 | gpu | 256 | 256 | 0.262 | 0.261 | 0.272 | 975,389 | 0.7731 | 0.4796 | not available | 1.8 GB | 41.4 MB |
| DeepFM | TensorRT fp16 | fp16 | gpu | 1024 | 1024 | 0.362 | 0.361 | 0.383 | 2,825,700 | 0.7731 | 0.4796 | not available | 1.8 GB | 41.4 MB |
| DeepFM | TensorRT fp16 | fp16 | gpu | 4096 | 4096 | 0.859 | 0.856 | 0.882 | 4,769,481 | 0.7731 | 0.4796 | not available | 1.8 GB | 41.4 MB |
| DeepFM | TensorRT int8 | int8 | gpu | 1 | 1 | 0.213 | 0.207 | 0.262 | 4,701 | 0.7734 | 0.4782 | 25.4 | 1.8 GB | 101.3 MB |
| DeepFM | TensorRT int8 | int8 | gpu | 256 | 256 | 0.247 | 0.246 | 0.260 | 1,034,510 | 0.7734 | 0.4782 | 25.4 | 1.8 GB | 101.3 MB |
| DeepFM | TensorRT int8 | int8 | gpu | 1024 | 1024 | 0.349 | 0.347 | 0.358 | 2,936,026 | 0.7734 | 0.4782 | 25.4 | 1.8 GB | 101.3 MB |
| DeepFM | TensorRT int8 | int8 | gpu | 4096 | 4096 | 0.760 | 0.750 | 0.836 | 5,386,846 | 0.7734 | 0.4782 | 25.4 | 1.8 GB | 101.3 MB |

![Latency against batch size](inference_latency_vs_batch.png)

![Throughput against batch size](inference_throughput_vs_batch.png)

## Accuracy against the fp32 reference

Reduced precision is not free. An fp16 engine rounds every activation to half the mantissa and an int8 engine replaces the numbers entirely with a quantized approximation chosen by a calibrator. This table reports what that cost, measured against eager PyTorch fp32 on exactly the same test rows. The regression is reported whatever it is. Nothing here was tuned to make a lower precision engine look better.

| Model | Backend | Precision | AUC | AUC delta | LogLoss | LogLoss delta |
| --- | --- | --- | --- | --- | --- | --- |
| DeepFM | PyTorch | fp32 | 0.77313 | +0.00000 | 0.47956 | +0.00000 |
| DeepFM | ONNX Runtime | fp32 | 0.77313 | +0.00000 | 0.47956 | +0.00000 |
| DeepFM | PyTorch CUDA fp32 | fp32 | 0.77313 | +0.00000 | 0.47956 | -0.00000 |
| DeepFM | PyTorch CUDA fp16 | fp16 | 0.77313 | -0.00000 | 0.47956 | +0.00000 |
| DeepFM | ORT CUDA | fp32 | 0.77313 | +0.00000 | 0.47956 | +0.00000 |
| DeepFM | ORT TRT fp16 | fp16 | 0.77313 | +0.00000 | 0.47956 | +0.00000 |
| DeepFM | TensorRT fp32 | fp32 | 0.77313 | -0.00000 | 0.47956 | +0.00000 |
| DeepFM | TensorRT fp16 | fp16 | 0.77313 | -0.00000 | 0.47957 | +0.00001 |
| DeepFM | TensorRT int8 | int8 | 0.77341 | +0.00028 | 0.47818 | -0.00138 |

The largest accuracy cost on this run belongs to TensorRT fp16 at fp16, which moved AUC by -0.00000 against the fp32 reference.

![Test AUC against precision](inference_accuracy_vs_precision.png)

## Correctness gate

A matching AUC is a weak claim. AUC depends only on the ordering of the predictions, so a backend could shift every probability by a full percent and still report the same AUC to four decimals, which would leave the model badly calibrated while the table looked clean. Calibration is what ad pricing multiplies a bid by, so the gate here is the mean absolute difference between each backend's probabilities and the eager PyTorch fp32 probabilities on identical rows. It says how far a backend drifted rather than asserting that it did not.

| Model | Backend | Precision | Batch | Mean abs difference | Max abs difference |
| --- | --- | --- | --- | --- | --- |
| DeepFM | PyTorch | fp32 | 1024 | 0.000e+00 | 0.000e+00 |
| DeepFM | ONNX Runtime | fp32 | 1024 | 1.691e-08 | 1.187e-07 |
| DeepFM | PyTorch CUDA fp32 | fp32 | 1024 | 1.859e-08 | 1.318e-07 |
| DeepFM | PyTorch CUDA fp16 | fp16 | 1024 | 4.273e-05 | 3.310e-04 |
| DeepFM | ORT CUDA | fp32 | 1024 | 1.691e-08 | 1.187e-07 |
| DeepFM | ORT TRT fp16 | fp16 | 1024 | 1.691e-08 | 1.187e-07 |
| DeepFM | TensorRT fp32 | fp32 | 1024 | 2.742e-05 | 3.162e-04 |
| DeepFM | TensorRT fp16 | fp16 | 1024 | 7.162e-05 | 6.739e-04 |
| DeepFM | TensorRT int8 | int8 | 1024 | 5.148e-02 | 3.659e-01 |

## Power and energy

Latency says how fast. Power says what that speed costs, and across a serving fleet the second number is the one that sets the bill. Power is sampled through NVML on a background thread around the timing loop only, so the joules reported belong to a region of known work. Energy is the trapezoid integral of the sampled watt curve, and inferences per joule is the efficiency figure that compares two precisions without reference to wall clock.

| Model | Backend | Precision | Batch | Mean power (W) | Peak power (W) | Energy (J) | Inferences per joule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | PyTorch | fp32 | 1024 | not available | not available | not available | not available |
| DeepFM | ONNX Runtime | fp32 | 1024 | not available | not available | not available | not available |
| DeepFM | PyTorch CUDA fp32 | fp32 | 1024 | 84.6 | 84.8 | 10.07 | 2,543 |
| DeepFM | PyTorch CUDA fp16 | fp16 | 1024 | 84.7 | 84.7 | 5.23 | 4,894 |
| DeepFM | ORT CUDA | fp32 | 1024 | 83.9 | 84.6 | 343.57 | 75 |
| DeepFM | ORT TRT fp16 | fp16 | 1024 | 84.4 | 84.7 | 363.05 | 71 |
| DeepFM | TensorRT fp32 | fp32 | 1024 | 84.8 | 84.8 | 9.06 | 2,826 |
| DeepFM | TensorRT fp16 | fp16 | 1024 | 84.8 | 84.8 | 0.96 | 26,769 |
| DeepFM | TensorRT int8 | int8 | 1024 | 84.8 | 84.8 | 0.93 | 27,629 |

## Is this model memory bound or compute bound

A DeepFM or a DCN is one very large embedding table followed by a very small multilayer perceptron. The multiply accumulate work in the perceptron is small, while the embedding lookup drags scattered rows out of a table tens of megabytes wide with almost no locality. If that shape dominates then the wall clock is set by memory bandwidth, a narrower multiply buys much less than the factor of two that half precision seems to promise, and int8 trades accuracy for a speedup that was never on the table. This section computes the evidence rather than asserting the conclusion, and it reports the answer the numbers give.

### DeepFM

| Batch | FLOPs per row | Bytes per row | Gather share of bytes | Arithmetic intensity | Achieved GB/s | Achieved GFLOP/s | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 394,230 | 788,892 | 0.3 percent | 0.50 | 10.6 | 5.3 | memory bound |
| 256 | 394,230 | 9,500 | 25.8 percent | 41.50 | 9.8 | 407.8 | near the ridge point |
| 1024 | 394,230 | 7,208 | 34.0 percent | 54.69 | 21.2 | 1,157.5 | near the ridge point |
| 4096 | 394,230 | 6,635 | 36.9 percent | 59.42 | 35.7 | 2,123.7 | near the ridge point |

The arithmetic intensity moves with the batch size, which is the whole story for this model. At batch 1 the perceptron weights have to cross the memory bus for a single row, so the intensity is 0.50 operations per byte and the workload is memory bound. At batch 4096 those same weights amortize over the whole batch and the intensity rises to 59.42 operations per byte, which is near the ridge point. The online serving case and the offline scoring case therefore sit on opposite sides of the roofline, and an optimization that helps one can do nothing at all for the other.

At the headline batch size of 1024 the reading is near the ridge point, because the arithmetic intensity is 54.69 operations per byte, inside the 20 to 100 operations per byte band where the ridge point of a modern gpu sits, so neither bound dominates and the answer depends on the specific part.

The cost model at that batch size in full.

| Quantity | Value |
| --- | --- |
| Batch size the counts are taken at | 1024 |
| Embedded fields per row | 36 |
| Embedding dimension | 16 |
| Embedding table parameters | 21,420,000 |
| Embedding table size at fp32 | 85.7 MB |
| Floating point operations per row | 394,230 |
| Bytes moved per row | 7,208 |
| Share of bytes that is the embedding gather | 34.0 percent |
| Arithmetic intensity | 54.69 operations per byte |
| Reference ridge point band | 20 to 100 operations per byte |
| Achieved bandwidth | 21.2 GB/s |
| Device peak bandwidth (estimated) | 2,039 GB/s |
| Bandwidth utilization | 1.0 percent |
| Verdict | near the ridge point |

On the fp16 prediction, fp16 came in 1.04 times faster than fp32. The roofline reading was inconclusive, so this ratio is reported without a prediction attached to it.

#### Where the time goes inside the engine

The TensorRT layer profiler was attached for 20 iterations at batch 1024. Attaching a profiler adds overhead and suppresses some fusion, so these milliseconds are not the latency the tables report. What the profile is for is the shape of the distribution, which is where the time goes rather than how much of it there is.

| Bucket | Time per iteration (ms) | Share |
| --- | --- | --- |
| Embedding gather and data movement | 0.0000 | 0.0 percent |
| Matrix multiply and perceptron | 0.0829 | 37.6 percent |
| Everything else | 0.1374 | 62.4 percent |

The perceptron owns 37.6 percent of the engine time against 0.0 percent for the gather. That contradicts the memory bound prediction for this model at this batch size, and the measurement is what stands. The hash bucket space this project uses keeps the embedding tables small enough that the perceptron is the larger cost, which is a real difference from a production scale DLRM with tables in the tens of gigabytes.


## Reproducing this report

```bash
python scripts/build_trt_engines.py
python scripts/run_inference_benchmark.py --models deepfm --batch-sizes 1 256 1024 4096
```

The engine builder is a separate step because a TensorRT plan is tied to the gpu architecture, the driver, and the TensorRT version that produced it, so it has to be built on the machine that will serve it. On a host with no gpu the builder prints what is missing and exits cleanly, and this benchmark then reports every gpu row as not available with the same reason.
