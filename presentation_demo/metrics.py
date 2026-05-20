"""Per-request performance metrics.

The formulas below are re-implemented to match ais_bench exactly so the demo
reports numbers using the same definitions a user would see from a real
ais_bench dataset run, but without paying the dataset-initialization cost
described in 需求描述.txt (4).

References:
- ais_bench/benchmark/summarizers/default_perf.py:150-181
    start_time / end_time / latency / ttft / tpot / itl / generate_tokens_speed
- ais_bench/benchmark/calculators/base_perf_metric_calculator.py:_calc_common_metrics
    Benchmark Duration / Request Throughput / Input Token Throughput /
    Output Token Throughput / Total Token Throughput / Prefill Token Throughput /
    Concurrency / etc., including the ms-unit conversion via SECOND_TO_MILLISECOND.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


SECOND_TO_MILLISECOND = 1000


@dataclass
class RequestMetrics:
    """Holds raw timing data for a single chat-completions stream and exposes
    derived metrics computed with ais_bench formulas.

    time_points semantics (matches ais_bench Output.time_points):
        time_points[0]   = perf_counter() right before sending the HTTP request
        time_points[1]   = perf_counter() when the first content delta arrives
        time_points[2:]  = perf_counter() when each subsequent content delta arrives
        time_points[-1]  = perf_counter() of the last content delta (end of stream)
    """

    time_points: List[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = False
    error_info: str = ""

    # ----- derived metrics (filled by finalize) -----
    start_time: float = 0.0
    end_time: float = 0.0
    latency: float = 0.0          # E2EL, seconds
    ttft: float = 0.0             # seconds
    tpot: float = 0.0             # seconds
    itl: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    generate_tokens_speed: float = 0.0   # tokens / second

    # Common (request-set level) metrics; for a single request many of these
    # are trivial but we keep them to mirror the ais_bench summary table.
    infer_time: float = 0.0
    request_throughput: float = 0.0          # req / s
    total_input_tokens: int = 0
    total_generated_tokens: int = 0
    prefill_token_throughput: Optional[float] = None  # tokens / s; None when input_tokens or ttft missing
    input_token_throughput: float = 0.0      # tokens / s
    output_token_throughput: float = 0.0     # tokens / s
    total_token_throughput: float = 0.0      # tokens / s
    concurrency: float = 0.0
    max_concurrency: int = 1

    def record_time_point(self) -> None:
        """Record a single perf_counter() sample.

        Called once just before the HTTP request is dispatched (start) and
        once for each non-empty content delta received from the server.
        """
        self.time_points.append(time.perf_counter())

    def finalize(self) -> None:
        """Compute derived metrics from raw time_points and token counts.

        Mirrors ais_bench's per-request calculation block (default_perf.py:167-180)
        and the common-metric pass in BasePerfMetricCalculator._calc_common_metrics.
        Safe to call only when at least 2 time points were captured; otherwise
        marks the request as unsuccessful (chunk-size guard mirrors ais_bench
        RequestOutput.get_metrics).
        """
        tp = self.time_points
        if len(tp) <= 1:
            self.success = False
            if not self.error_info:
                self.error_info = "chunk size is less than 2"
            return

        self.success = True
        self.start_time = tp[0]
        self.end_time = tp[-1]
        self.latency = tp[-1] - tp[0]
        self.ttft = tp[1] - tp[0]
        if self.output_tokens > 1:
            self.tpot = (self.latency - self.ttft) / (self.output_tokens - 1)
        else:
            self.tpot = 0.0
        if len(tp) > 2:
            self.itl = np.diff(np.asarray(tp[1:], dtype=np.float64))
        else:
            self.itl = np.array([], dtype=np.float64)
        if self.latency > 0:
            self.generate_tokens_speed = self.output_tokens / self.latency
        else:
            self.generate_tokens_speed = 0.0

        # ----- common metrics -----
        # For a single request, infer_time == latency (max(end_time) - min(start_time)).
        self.infer_time = self.latency
        self.total_input_tokens = self.input_tokens
        self.total_generated_tokens = self.output_tokens

        if self.infer_time > 0:
            self.request_throughput = 1.0 / self.infer_time
            self.input_token_throughput = self.input_tokens / self.infer_time
            self.output_token_throughput = self.output_tokens / self.infer_time
            self.total_token_throughput = (
                self.input_tokens + self.output_tokens
            ) / self.infer_time
        else:
            self.request_throughput = 0.0
            self.input_token_throughput = 0.0
            self.output_token_throughput = 0.0
            self.total_token_throughput = 0.0

        # ais_bench: Prefill Token Throughput = total_input_tokens / sum(ttft)
        # For a single request sum(ttft) == ttft.
        if self.input_tokens > 0 and self.ttft > 0:
            self.prefill_token_throughput = self.input_tokens / self.ttft
        else:
            self.prefill_token_throughput = None

        # ais_bench: Concurrency = sum(E2EL) / infer_time. With one request this
        # collapses to 1.0, but we compute it explicitly to stay faithful.
        if self.infer_time > 0:
            self.concurrency = self.latency / self.infer_time
        else:
            self.concurrency = 0.0
