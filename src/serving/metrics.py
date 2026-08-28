"""In process request metrics for the scoring service.

A serving system that reports only a mean is not observable. docs/INFERENCE.md
already argues the point for offline measurement, that a backend with a low mean
and a fat tail is worse in production than one with a slightly higher mean and a
tight distribution, and that only the tail column shows it. The same argument is
sharper online, because a real request fans out across components and the
slowest one sets the response time.

So the histogram is the primary structure and the counters hang off it. Buckets
are cumulative and fixed at build time, which is what makes a histogram
aggregatable across processes. Two histograms with the same buckets can be added
bucket by bucket and the sum still supports a quantile estimate, which is why
Prometheus histograms look the way they do and why the buckets here are not
chosen per deployment.

Three latencies are tracked rather than one. The end to end request latency is
what a caller experiences and what the service level objective is stated
against. The feature latency and the model latency are tracked separately
because on this model they are not close to each other, and a service that
reports only the total cannot tell an operator which half to go fix.

Alongside the histograms there is a bounded reservoir of recent latencies. A
histogram gives interpolated quantiles from bucket counts, which is the right
tradeoff for a metrics system and the wrong one for a report that wants an exact
p999. The reservoir holds the most recent observations and yields exact order
statistics over that window, and it is bounded so a long lived process cannot
grow without limit.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Sequence, Tuple

import numpy as np

# Cumulative bucket upper bounds in seconds. The range is dense between one
# millisecond and one hundred milliseconds because that is the band an ad
# ranking service lives or dies in, and it thins out above that because a
# request that took a second already failed its budget and the exact figure no
# longer changes a decision.
DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
    0.250,
    0.500,
    1.000,
    2.500,
    5.000,
)

# How many recent observations the exact quantile reservoir keeps.
RESERVOIR_SIZE: int = 20000


class Histogram:
    """A cumulative bucket histogram with a bounded exact quantile reservoir."""

    def __init__(self, name: str, buckets: Sequence[float] = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.buckets: Tuple[float, ...] = tuple(sorted(float(b) for b in buckets))
        self.counts: List[int] = [0] * len(self.buckets)
        self.inf_count: int = 0
        self.total: int = 0
        self.sum_seconds: float = 0.0
        self.reservoir: Deque[float] = deque(maxlen=RESERVOIR_SIZE)

    def observe(self, seconds: float) -> None:
        """Record one observation into every bucket it falls under."""
        value = float(seconds)
        self.total += 1
        self.sum_seconds += value
        self.reservoir.append(value)
        placed = False
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                placed = True
        if not placed:
            self.inf_count += 1

    def quantiles(self, qs: Sequence[float] = (50.0, 95.0, 99.0, 99.9)) -> Dict[str, float]:
        """Return exact order statistics over the recent observation window."""
        if not self.reservoir:
            return {f"p{q:g}": float("nan") for q in qs}
        arr = np.asarray(self.reservoir, dtype=np.float64)
        return {f"p{q:g}": float(np.percentile(arr, q)) for q in qs}

    def snapshot(self) -> Dict[str, Any]:
        """Return the json friendly view of this histogram."""
        mean = self.sum_seconds / self.total if self.total else float("nan")
        return {
            "count": self.total,
            "sum_seconds": self.sum_seconds,
            "mean_ms": mean * 1000.0,
            "quantiles_ms": {
                k: v * 1000.0 for k, v in self.quantiles().items()
            },
            "reservoir_size": len(self.reservoir),
            "buckets_seconds": list(self.buckets),
            "bucket_counts": list(self.counts),
            "inf_count": self.inf_count,
        }

    def prometheus_lines(self, help_text: str) -> List[str]:
        """Render this histogram in the Prometheus text exposition format."""
        lines = [f"# HELP {self.name} {help_text}", f"# TYPE {self.name} histogram"]
        # The stored counts are already cumulative, because observe increments
        # every bucket whose upper bound the value falls under.
        for bound, count in zip(self.buckets, self.counts):
            lines.append(f'{self.name}_bucket{{le="{bound}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.total}')
        lines.append(f"{self.name}_sum {self.sum_seconds:.9f}")
        lines.append(f"{self.name}_count {self.total}")
        return lines


class ServiceMetrics:
    """Every counter and histogram the scoring service exposes.

    All mutation happens under one lock. The service scores requests on a
    thread pool, so two threads can finish at the same instant, and a metrics
    structure that loses an observation under contention is worse than no
    metrics because it looks correct.
    """

    def __init__(self, buckets: Sequence[float] = DEFAULT_BUCKETS) -> None:
        self._lock = threading.Lock()
        self.requests_total: int = 0
        self.requests_failed_total: int = 0
        self.ads_scored_total: int = 0
        self.candidates_per_request_sum: int = 0
        self.request_latency = Histogram("adrank_request_latency_seconds", buckets)
        self.feature_latency = Histogram("adrank_feature_latency_seconds", buckets)
        self.model_latency = Histogram("adrank_model_latency_seconds", buckets)

    def observe_request(
        self,
        total_seconds: float,
        feature_seconds: float,
        model_seconds: float,
        n_candidates: int,
    ) -> None:
        """Record one successfully served request."""
        with self._lock:
            self.requests_total += 1
            self.ads_scored_total += int(n_candidates)
            self.candidates_per_request_sum += int(n_candidates)
            self.request_latency.observe(total_seconds)
            self.feature_latency.observe(feature_seconds)
            self.model_latency.observe(model_seconds)

    def observe_failure(self) -> None:
        """Record one request the service could not serve."""
        with self._lock:
            self.requests_total += 1
            self.requests_failed_total += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return the json view of every counter and histogram."""
        with self._lock:
            served = self.requests_total - self.requests_failed_total
            return {
                "requests_total": self.requests_total,
                "requests_failed_total": self.requests_failed_total,
                "requests_served_total": served,
                "ads_scored_total": self.ads_scored_total,
                "mean_candidates_per_request": (
                    self.candidates_per_request_sum / served if served else float("nan")
                ),
                "request_latency": self.request_latency.snapshot(),
                "feature_latency": self.feature_latency.snapshot(),
                "model_latency": self.model_latency.snapshot(),
            }

    def render_prometheus(self) -> str:
        """Render every metric in the Prometheus text exposition format.

        The format is written by hand rather than pulled from a client library,
        because it is a dozen lines of text and it keeps the serving image free
        of a dependency that exists only to print them.
        """
        with self._lock:
            lines: List[str] = [
                "# HELP adrank_requests_total Scoring requests received.",
                "# TYPE adrank_requests_total counter",
                f"adrank_requests_total {self.requests_total}",
                "# HELP adrank_requests_failed_total Scoring requests that could not be served.",
                "# TYPE adrank_requests_failed_total counter",
                f"adrank_requests_failed_total {self.requests_failed_total}",
                "# HELP adrank_ads_scored_total Candidate ads scored across all requests.",
                "# TYPE adrank_ads_scored_total counter",
                f"adrank_ads_scored_total {self.ads_scored_total}",
            ]
            lines += self.request_latency.prometheus_lines(
                "End to end scoring request latency in seconds."
            )
            lines += self.feature_latency.prometheus_lines(
                "Feature pipeline latency in seconds, part of the request latency."
            )
            lines += self.model_latency.prometheus_lines(
                "Model execution latency in seconds, part of the request latency."
            )
        return "\n".join(lines) + "\n"
