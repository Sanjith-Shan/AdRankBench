"""Roofline style cost analysis for a DLRM shaped ranking model.

The interesting claim about this family of models is that they are not compute
bound. A DeepFM or a DCN is one very large embedding table plus a very small
multilayer perceptron. The multiply accumulate work in the perceptron is tiny,
a few hundred thousand operations per row, while the embedding lookup drags
scattered rows out of a table that is tens of megabytes wide and has no
locality worth speaking of. If that is true then the wall clock is set by memory
bandwidth, a lower precision multiply buys much less than the usual factor of
two, and INT8 costs accuracy for a speedup that was never available.

This module computes the numbers that decide the question rather than asserting
the answer. It counts the floating point operations and the bytes moved per row,
divides them into an arithmetic intensity, and compares the achieved bandwidth
against the device peak when the device peak is known. The report states
whichever way the measurement falls, including the case where the prediction is
wrong.

Nothing here needs a gpu. The counts come from the module definition, so they
are available on any machine, and the parts that need a device are the achieved
bandwidth and the peak, which degrade to not available on their own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.inference.common import NOT_AVAILABLE

# A rough reference band for where the ridge point of a modern data centre gpu
# sits, in floating point operations per byte of device memory traffic. It is a
# band and not a number because it moves with the precision and with the part.
# A workload well below this band is memory bound whatever the exact figure is.
RIDGE_POINT_REFERENCE = (20.0, 100.0)


def module_cost_model(
    module: Any,
    n_embed_fields: int,
    embed_dim: int,
    n_numerical: int,
    batch_size: int,
    element_bytes: int = 4,
) -> Dict[str, Any]:
    """Count floating point operations and bytes moved for one batch.

    The walk is generic over the module tree rather than special cased per
    architecture, so it is correct for DeepFM and for DCN without either of them
    knowing about this file.

    The byte count has four parts.

    The embedding gather is the dominant term. Each row pulls n_embed_fields
    rows out of the shared table, each of width embed_dim for the second order
    part and width one for the first order part. Those reads are scattered, so
    every one of them is a separate cache line at best and a separate memory
    transaction at worst, and the model gets no reuse across a batch unless two
    rows happen to share a field value.

    The weight traffic is the perceptron weights, read once per batch rather
    than once per row, because they stay resident while the batch streams past.
    That is why the weight term divides out as the batch grows and the gather
    term does not.

    The activation traffic is the intermediate tensors written and read between
    layers, counted as one write and one read per layer output.

    The index traffic is the int64 field indices read from the input.

    Returns a record with per batch and per row figures and the arithmetic
    intensity, which is total operations divided by total bytes.
    """
    import torch.nn as nn

    batch = max(1, int(batch_size))

    linear_flops = 0
    weight_bytes = 0
    activation_elements = 0
    embedding_parameters = 0
    n_linear = 0

    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            n_linear += 1
            # One multiply and one add per weight, plus the bias add.
            linear_flops += 2 * int(sub.in_features) * int(sub.out_features)
            linear_flops += int(sub.out_features)
            weight_bytes += (
                int(sub.in_features) * int(sub.out_features) * element_bytes
            )
            if sub.bias is not None:
                weight_bytes += int(sub.out_features) * element_bytes
            activation_elements += int(sub.out_features)
        elif isinstance(sub, nn.Embedding):
            embedding_parameters += int(sub.weight.numel())

    linear_flops *= batch

    # The factorization machine second order term and the cross network both
    # work over the flattened embedding vector, so they are linear in the number
    # of embedding elements per row rather than quadratic. Counting them as a
    # small constant multiple of the embedding width is close enough for a
    # roofline argument and is stated as an estimate in the report.
    interaction_flops = batch * n_embed_fields * embed_dim * 6

    total_flops = linear_flops + interaction_flops

    gather_elements = batch * n_embed_fields * (embed_dim + 1)
    gather_bytes = gather_elements * element_bytes
    index_bytes = batch * n_embed_fields * 8
    numerical_bytes = batch * n_numerical * element_bytes
    activation_bytes = 2 * batch * activation_elements * element_bytes
    output_bytes = batch * element_bytes

    total_bytes = (
        gather_bytes + index_bytes + numerical_bytes + weight_bytes
        + activation_bytes + output_bytes
    )

    intensity = (total_flops / total_bytes) if total_bytes else None

    return {
        "batch_size": batch,
        "element_bytes": element_bytes,
        "n_embed_fields": int(n_embed_fields),
        "embed_dim": int(embed_dim),
        "n_linear_layers": n_linear,
        "embedding_parameters": embedding_parameters,
        "embedding_table_bytes": embedding_parameters * element_bytes,
        "flops_per_batch": total_flops,
        "flops_per_row": total_flops / batch,
        "linear_flops_per_batch": linear_flops,
        "interaction_flops_per_batch": interaction_flops,
        "bytes_per_batch": total_bytes,
        "bytes_per_row": total_bytes / batch,
        "gather_bytes_per_batch": gather_bytes,
        "gather_share_of_bytes_pct": (
            gather_bytes / total_bytes * 100.0 if total_bytes else None
        ),
        "weight_bytes_per_batch": weight_bytes,
        "activation_bytes_per_batch": activation_bytes,
        "index_bytes_per_batch": index_bytes,
        "arithmetic_intensity_flops_per_byte": intensity,
    }


def achieved_bandwidth_gb_s(
    bytes_per_batch: float, mean_latency_ms: Optional[float]
) -> Optional[float]:
    """Convert a per batch byte count and a per batch latency into GB per second."""
    if not mean_latency_ms or mean_latency_ms <= 0:
        return None
    seconds = float(mean_latency_ms) / 1000.0
    return float(bytes_per_batch) / seconds / 1e9


def achieved_gflops(
    flops_per_batch: float, mean_latency_ms: Optional[float]
) -> Optional[float]:
    """Convert a per batch operation count and a per batch latency into GFLOP/s."""
    if not mean_latency_ms or mean_latency_ms <= 0:
        return None
    seconds = float(mean_latency_ms) / 1000.0
    return float(flops_per_batch) / seconds / 1e9


def bound_verdict(
    intensity: Optional[float],
    achieved_bw: Optional[float],
    peak_bw: Optional[float],
) -> Dict[str, Any]:
    """Decide, from the numbers, whether this workload is memory or compute bound.

    The verdict has two independent pieces of evidence. The arithmetic intensity
    says where the workload sits relative to the ridge point of a typical gpu,
    which is a property of the model and needs no device. The achieved fraction
    of peak bandwidth says whether the device was actually saturated, which
    needs a device and a peak figure and is not available otherwise.

    The function does not decide in advance. It returns the label the numbers
    support, including inconclusive when the evidence is not there.
    """
    low, high = RIDGE_POINT_REFERENCE
    record: Dict[str, Any] = {
        "arithmetic_intensity_flops_per_byte": intensity,
        "ridge_point_reference": list(RIDGE_POINT_REFERENCE),
        "achieved_bandwidth_gb_s": achieved_bw,
        "peak_bandwidth_gb_s": peak_bw,
        "bandwidth_utilization_pct": None,
        "verdict": "inconclusive",
        "explanation": (
            "there was not enough evidence on this run to call the workload "
            "memory bound or compute bound"
        ),
    }

    if achieved_bw is not None and peak_bw:
        record["bandwidth_utilization_pct"] = achieved_bw / peak_bw * 100.0

    if intensity is None:
        return record

    if intensity < low:
        record["verdict"] = "memory bound"
        record["explanation"] = (
            f"the arithmetic intensity is {intensity:.2f} operations per byte, "
            f"below the {low:.0f} to {high:.0f} operations per byte band where "
            "the ridge point of a modern gpu sits, so the wall clock is set by "
            "how fast the embedding rows can be pulled out of memory and not by "
            "how fast the multiply accumulate units can consume them"
        )
    elif intensity > high:
        record["verdict"] = "compute bound"
        record["explanation"] = (
            f"the arithmetic intensity is {intensity:.2f} operations per byte, "
            f"above the {low:.0f} to {high:.0f} operations per byte band where "
            "the ridge point of a modern gpu sits, so the wall clock is set by "
            "arithmetic throughput and a lower precision multiply should pay off"
        )
    else:
        record["verdict"] = "near the ridge point"
        record["explanation"] = (
            f"the arithmetic intensity is {intensity:.2f} operations per byte, "
            f"inside the {low:.0f} to {high:.0f} operations per byte band where "
            "the ridge point of a modern gpu sits, so neither bound dominates "
            "and the answer depends on the specific part"
        )
    return record


def speedup_prediction_check(
    fp32_latency_ms: Optional[float],
    fp16_latency_ms: Optional[float],
    verdict: str,
) -> Dict[str, Any]:
    """Compare the measured fp16 speedup against what the verdict predicts.

    A memory bound workload should show far less than the factor of two that a
    naive reading of half precision promises, because halving the width of the
    arithmetic does not halve the number of embedding rows that have to cross
    the memory bus. This function states the measured ratio and says plainly
    whether it supports the verdict or contradicts it.
    """
    record: Dict[str, Any] = {
        "fp32_latency_ms": fp32_latency_ms,
        "fp16_latency_ms": fp16_latency_ms,
        "measured_speedup": None,
        "supports_verdict": None,
        "statement": (
            "the fp16 speedup could not be measured on this run, so the "
            "prediction was neither confirmed nor contradicted"
        ),
    }
    if not fp32_latency_ms or not fp16_latency_ms or fp16_latency_ms <= 0:
        return record

    speedup = float(fp32_latency_ms) / float(fp16_latency_ms)
    record["measured_speedup"] = speedup

    if verdict == "memory bound":
        supports = speedup < 1.6
        record["supports_verdict"] = supports
        if supports:
            record["statement"] = (
                f"fp16 came in {speedup:.2f} times faster than fp32, well short "
                "of the factor of two a compute bound kernel would give, which "
                "is what a memory bound workload looks like and confirms the "
                "roofline reading"
            )
        else:
            record["statement"] = (
                f"fp16 came in {speedup:.2f} times faster than fp32, at or above "
                "the factor of two that a compute bound kernel would give. That "
                "contradicts the memory bound reading from the arithmetic "
                "intensity, so the prediction was wrong on this hardware and the "
                "measurement is what stands"
            )
    elif verdict == "compute bound":
        supports = speedup >= 1.6
        record["supports_verdict"] = supports
        record["statement"] = (
            f"fp16 came in {speedup:.2f} times faster than fp32, which "
            + ("matches" if supports else "falls short of")
            + " what a compute bound workload should give"
        )
    else:
        record["statement"] = (
            f"fp16 came in {speedup:.2f} times faster than fp32. The roofline "
            "reading was inconclusive, so this ratio is reported without a "
            "prediction attached to it"
        )
    return record


def cost_model_markdown(cost: Dict[str, Any], verdict: Dict[str, Any]) -> List[str]:
    """Render the cost model and the verdict as markdown lines."""

    def num(value, spec=",.0f"):
        if value is None:
            return NOT_AVAILABLE
        return format(value, spec)

    lines = [
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Batch size the counts are taken at | {cost.get('batch_size', NOT_AVAILABLE)} |",
        f"| Embedded fields per row | {cost.get('n_embed_fields', NOT_AVAILABLE)} |",
        f"| Embedding dimension | {cost.get('embed_dim', NOT_AVAILABLE)} |",
        f"| Embedding table parameters | {num(cost.get('embedding_parameters'))} |",
        f"| Embedding table size at fp32 | {num((cost.get('embedding_table_bytes') or 0) / 1e6, ',.1f')} MB |",
        f"| Floating point operations per row | {num(cost.get('flops_per_row'))} |",
        f"| Bytes moved per row | {num(cost.get('bytes_per_row'))} |",
        f"| Share of bytes that is the embedding gather | {num(cost.get('gather_share_of_bytes_pct'), '.1f')} percent |",
        f"| Arithmetic intensity | {num(cost.get('arithmetic_intensity_flops_per_byte'), '.2f')} operations per byte |",
        f"| Reference ridge point band | {RIDGE_POINT_REFERENCE[0]:.0f} to {RIDGE_POINT_REFERENCE[1]:.0f} operations per byte |",
        f"| Achieved bandwidth | {num(verdict.get('achieved_bandwidth_gb_s'), ',.1f')} GB/s |",
        f"| Device peak bandwidth (estimated) | {num(verdict.get('peak_bandwidth_gb_s'), ',.0f')} GB/s |",
        f"| Bandwidth utilization | {num(verdict.get('bandwidth_utilization_pct'), '.1f')} percent |",
        f"| Verdict | {verdict.get('verdict', NOT_AVAILABLE)} |",
    ]
    return lines
