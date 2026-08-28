"""Explicit int8 quantization of an ONNX graph, for TensorRT 11 and newer.

Why this module exists
----------------------

The original int8 plan for this project was the classic TensorRT recipe. Set
``BuilderFlag.INT8``, hand the builder an ``IInt8EntropyCalibrator2`` fed from a
held out slice of data, and let the builder work out a scale for every activation
tensor. That is implicit quantization, and it was how TensorRT worked for years.

TensorRT 11 removed all of it. On that version there is no ``BuilderFlag.INT8``,
no ``IInt8EntropyCalibrator2``, no ``IInt8Calibrator`` of any kind, and no
``set_calibration_profile``. Networks are now strongly typed, which means the
precision of every tensor is read off the graph rather than requested at build
time. A calibrator has nowhere to attach.

The replacement is explicit quantization. Instead of telling the builder to
quantize, the graph itself carries QuantizeLinear and DequantizeLinear node pairs
that say exactly where the tensor becomes int8 and with what scale. TensorRT then
honours what the graph asks for. The calibration step does not disappear, it
moves earlier, out of the builder and into a graph rewriting pass that has to run
before the builder ever sees the model.

This module is that pass. It uses the quantization tools that ship with ONNX
Runtime to run representative data through the graph, record activation ranges,
choose scales, and write a new graph with the Q and DQ pairs inserted.

The calibration rows still come from the validation split
-------------------------------------------------------

Nothing about the move to explicit quantization changes the leakage rule.
Calibration reads data and produces parameters, the per tensor scales, and those
parameters change the model's output. Fitting them on the test split would be
fitting parameters on the evaluation set, and the int8 accuracy loss that came
out would look smaller than it really is. The rows fed to the reader below come
from validation, the same split the trainer already uses for early stopping.

Entropy calibration is the default here because it is what
``IInt8EntropyCalibrator2`` did, which keeps this path as close as possible to
the one the older TensorRT would have taken. That matters for the comparison. The
point of the exercise is to measure what int8 costs this model, not to measure
the difference between two calibration algorithms.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.inference.common import NOT_AVAILABLE


def quantization_available() -> Tuple[bool, str]:
    """Return whether the ONNX Runtime quantization tools can be imported."""
    try:
        from onnxruntime.quantization import quantize_static  # noqa: F401
    except Exception as exc:  # noqa: BLE001 the tooling is optional
        return False, (
            "the onnxruntime quantization tools are not importable "
            f"({exc}). Install onnxruntime or onnxruntime-gpu to build an int8 "
            "graph."
        )
    return True, "the onnxruntime quantization tools are available."


class _ArrayCalibrationReader:
    """Feed fixed numpy batches to the ONNX Runtime calibrator, once each.

    The quantization pass expects an object with a ``get_next`` returning a feed
    dict or None when the data runs out. It is deliberately single pass, so the
    reader is rewound explicitly if it is reused.
    """

    def __init__(self, feeds: List[Dict[str, np.ndarray]]) -> None:
        self._feeds = feeds
        self._index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._index >= len(self._feeds):
            return None
        feed = self._feeds[self._index]
        self._index += 1
        return feed

    def rewind(self) -> None:
        self._index = 0


def build_calibration_feeds(
    numerical: np.ndarray,
    cat: np.ndarray,
    batch_size: int,
    max_batches: int,
    numerical_name: str = "numerical",
    cat_name: str = "cat",
) -> List[Dict[str, np.ndarray]]:
    """Slice validation arrays into the feed dicts the calibrator consumes.

    The arrays are the featurized validation split, so they are already in the
    exact form the model sees at inference. Only whole batches are used, because
    a short trailing batch would calibrate some tensors on fewer rows than the
    rest for no benefit.
    """
    feeds: List[Dict[str, np.ndarray]] = []
    total = min(len(numerical), len(cat))
    n_batches = min(max_batches, total // batch_size) if batch_size > 0 else 0
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        feeds.append({
            numerical_name: np.ascontiguousarray(numerical[start:end], dtype=np.float32),
            cat_name: np.ascontiguousarray(cat[start:end], dtype=np.int64),
        })
    return feeds


def quantize_onnx_int8(
    onnx_path: str,
    out_path: str,
    feeds: List[Dict[str, np.ndarray]],
    calibrate_method: str = "entropy",
    per_channel: bool = False,
) -> Dict[str, Any]:
    """Write an int8 QDQ copy of an onnx graph and describe what happened.

    Returns a record rather than raising, in the same style as the engine
    builder, so a failure becomes a reported not available row instead of a
    crash.

    per_channel is off by default, and that is a compromise forced by the
    TensorRT parser rather than a preference. A per channel scale lets each
    output channel keep its own range, which is normally the better choice
    because a single scale across a whole weight tensor is the usual reason a
    quantized model loses more accuracy than it needs to. The problem is that
    per channel quantization writes an axis attribute onto every quantize node,
    and this graph contains scalar constants with no dimensions at all. The
    parser rejects those with "axis must be in the range [0, nbDims (0)]" and
    the whole engine build fails. Per tensor scales carry no axis, so the graph
    parses. The cost is a coarser scale on the weight tensors, which is a real
    accuracy tradeoff and is reported rather than hidden.
    """
    record: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "int8_onnx_path": out_path,
        "source_onnx_path": onnx_path,
        "calibrate_method": calibrate_method,
        "per_channel": per_channel,
        "calibration_batches": len(feeds),
        "calibration_rows": int(sum(len(next(iter(f.values()))) for f in feeds)) if feeds else 0,
        "quantize_seconds": None,
        "size_bytes": None,
        "quantized_node_types": {},
    }

    ok, message = quantization_available()
    if not ok:
        record["message"] = message
        return record

    if not feeds:
        record["message"] = (
            "no calibration batches were supplied, so no int8 graph was written. "
            "Explicit quantization needs representative data to choose the "
            "activation scales."
        )
        return record

    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as exc:  # noqa: BLE001
        record["message"] = f"the quantization tools failed to import ({exc})."
        return record

    methods = {
        "entropy": getattr(CalibrationMethod, "Entropy", None),
        "minmax": getattr(CalibrationMethod, "MinMax", None),
        "percentile": getattr(CalibrationMethod, "Percentile", None),
    }
    method = methods.get(calibrate_method) or methods.get("minmax")
    if method is None:
        record["message"] = "no usable calibration method was found in this onnxruntime build."
        return record

    reader = _ArrayCalibrationReader(feeds)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    started = time.perf_counter()
    try:
        quantize_static(
            model_input=onnx_path,
            model_output=out_path,
            calibration_data_reader=reader,
            # QDQ is the format TensorRT understands. The alternative,
            # QOperator, folds quantization into fused integer operators that
            # the TensorRT parser does not read the same way.
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            per_channel=per_channel,
            # Gather reads an embedding row and does no arithmetic, so
            # quantizing it buys nothing and costs a rounding step on every
            # lookup. It is excluded on purpose and the report says so.
            # Only the matrix operations. int8 pays for itself where there is
            # a large dot product to accelerate, which here is the multilayer
            # perceptron. Add and Mul were in this list originally and had to
            # come out. They are frequently applied to scalar constants, and
            # quantizing a scalar produces a QuantizeLinear node on a zero
            # dimensional tensor carrying an axis attribute, which the TensorRT
            # parser rejects with "axis must be in the range [0, nbDims (0)]".
            # Those nodes also bought nothing, since a scalar multiply is not
            # where the time goes. Gather stays out for the same reason it
            # always was, an embedding lookup moves bytes rather than doing
            # arithmetic, so quantizing it adds a rounding step to every lookup
            # and accelerates nothing.
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={
                # TensorRT supports symmetric quantization only. It requires
                # the zero point on every QuantizeLinear and DequantizeLinear to
                # be zero, and it rejects the graph outright with "TensorRT only
                # supports symmetric quantization" otherwise. Asymmetric
                # activations are the ONNX Runtime default because they fit a
                # one sided distribution such as a post relu activation more
                # tightly, so this is a real accuracy tradeoff that TensorRT
                # imposes rather than a free switch. It also matches what the
                # old IInt8EntropyCalibrator2 did, since that path was symmetric
                # too, which keeps this comparable to the classic recipe.
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                # TensorRT rejects a graph whose bias tensors carry int32
                # quantize and dequantize pairs. Its DequantizeLinear accepts
                # int8, fp8, and fp4 only, so an int32 bias DQ node fails the
                # parser with "only activation types allowed as input to this
                # layer" and the whole engine build dies. ONNX Runtime quantizes
                # bias to int32 by default because that is what its own cpu
                # kernels want, so the two tools disagree and this switch is the
                # bridge. Leaving bias in fp32 costs almost nothing, since bias
                # is a vector next to a matrix.
                "QuantizeBias": False,
                # TensorRT reads weight scales off an explicit quantize and
                # dequantize pair rather than off a folded initializer, so this
                # keeps the pair in the graph where the parser can see it.
                "AddQDQPairToWeight": True,
            },
        )
    except Exception as exc:  # noqa: BLE001 surface the real reason
        record["message"] = f"the int8 quantization pass failed ({exc})."
        return record

    record["quantize_seconds"] = float(time.perf_counter() - started)

    try:
        record["size_bytes"] = int(os.path.getsize(out_path))
    except OSError:
        record["size_bytes"] = None

    record["quantized_node_types"] = _count_node_types(out_path)
    n_q = record["quantized_node_types"].get("QuantizeLinear", 0)
    record["ok"] = n_q > 0
    if n_q == 0:
        record["message"] = (
            "the quantization pass ran but inserted no QuantizeLinear nodes, so "
            "the graph is still effectively fp32 and no int8 engine is worth "
            "building from it."
        )
    else:
        record["message"] = (
            f"wrote an int8 QDQ graph with {n_q} QuantizeLinear nodes, "
            f"calibrated with the {calibrate_method} method on "
            f"{record['calibration_rows']} validation rows."
        )
    return record


def _count_node_types(onnx_path: str) -> Dict[str, int]:
    """Return a count of node types in a graph, for the quantization report."""
    try:
        import onnx
    except Exception:  # noqa: BLE001
        return {}
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:  # noqa: BLE001
        return {}
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def quantization_markdown(record: Dict[str, Any]) -> List[str]:
    """Render the quantization record as markdown lines for the build report."""
    lines: List[str] = []
    lines.append("### Int8 quantization")
    lines.append("")
    if not record.get("ok"):
        lines.append(
            "No int8 graph was produced. " + str(record.get("message") or NOT_AVAILABLE)
        )
        lines.append("")
        return lines

    counts = record.get("quantized_node_types") or {}
    lines.append(
        "TensorRT 11 removed the implicit quantization api, so int8 here is "
        "explicit. The graph carries QuantizeLinear and DequantizeLinear pairs "
        "that were inserted by a calibration pass before the builder ran, and "
        "the calibration rows come from the validation split rather than the "
        "test split."
    )
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Calibration method | {record.get('calibrate_method')} |")
    lines.append(f"| Calibration rows | {record.get('calibration_rows')} |")
    lines.append(f"| Calibration batches | {record.get('calibration_batches')} |")
    lines.append(f"| Per channel weight scales | {record.get('per_channel')} |")
    quant_s = record.get("quantize_seconds")
    lines.append(
        f"| Quantization pass seconds | {quant_s:.2f} |" if quant_s else
        f"| Quantization pass seconds | {NOT_AVAILABLE} |"
    )
    lines.append(f"| QuantizeLinear nodes | {counts.get('QuantizeLinear', 0)} |")
    lines.append(f"| DequantizeLinear nodes | {counts.get('DequantizeLinear', 0)} |")
    lines.append(f"| MatMul nodes | {counts.get('MatMul', 0)} |")
    lines.append(f"| Gather nodes (left in fp32 on purpose) | {counts.get('Gather', 0)} |")
    lines.append("")
    return lines
