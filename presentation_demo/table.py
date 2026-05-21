"""Pretty-print per-request metrics in the ais_bench table style.

Two tables are rendered with `tabulate` using `tablefmt="fancy_grid"`, the
exact format ais_bench's DefaultPerfSummarizer._output_to_screen uses.

Table 1 (Request level):
    columns: Performance Parameters | Stage | Average | Min | Max | Median | P75 | P90 | P99 | N
    rows:    TTFT, TPOT, ITL, InputTokens, OutputTokens, E2EL, OutputTokenThroughput

Table 2 (Common metric):
    columns: Common Metric | Value
    rows:    Benchmark Duration / Request Throughput /
             Total Input Tokens / Prefill Token Throughput / Total Generated Tokens /
             Input Token Throughput / Output Token Throughput / Total Token Throughput

Units match ais_bench exactly: time is ms (SECOND_TO_MILLISECOND), throughput
is `token/s` or `req/s`. For a single-request scenario all percentile columns
collapse to the same value (and N=1), matching how ais_bench would report a
single-row result.
"""

from __future__ import annotations

from typing import List

import numpy as np
from tabulate import tabulate

from presentation_demo.metrics import RequestMetrics, SECOND_TO_MILLISECOND


STAGE_NAME = "total"
STAT_COLUMNS = ["Average", "Min", "Max", "Median", "P75", "P90", "P99"]


def _fmt_ms(seconds: float) -> str:
    return f"{round(seconds * SECOND_TO_MILLISECOND, 4)} ms"


def _fmt_tokens_per_s(value: float) -> str:
    return f"{round(value, 4)} token/s"


def _fmt_req_per_s(value: float) -> str:
    return f"{round(value, 4)} req/s"


def _stats_row_constant(value: float, n: int, unit_fmt) -> List[str]:
    """Render seven stat cells for a single observation (all equal) + N."""
    cell = unit_fmt(value)
    return [cell] * len(STAT_COLUMNS) + [str(n)]


def _stats_row_array(arr: np.ndarray, n: int, unit_fmt) -> List[str]:
    """Render seven stat cells from a numpy array using the same percentile
    definitions as ais_bench BasePerfMetricCalculator._calculate_statistics."""
    if arr.size == 0:
        return [unit_fmt(0.0)] * len(STAT_COLUMNS) + [str(n)]
    values = {
        "Average": float(arr.mean()),
        "Min": float(arr.min()),
        "Max": float(arr.max()),
        "Median": float(np.percentile(arr, 50)),
        "P75": float(np.percentile(arr, 75)),
        "P90": float(np.percentile(arr, 90)),
        "P99": float(np.percentile(arr, 99)),
    }
    return [unit_fmt(values[c]) for c in STAT_COLUMNS] + [str(n)]


def _stats_row_int(value: int, n: int) -> List[str]:
    cell = str(value)
    return [cell] * len(STAT_COLUMNS) + [str(n)]


def build_request_table(m: RequestMetrics) -> List[List[str]]:
    headers = (
        ["Performance Parameters", "Stage"]
        + STAT_COLUMNS
        + ["N"]
    )
    rows: List[List[str]] = [headers]

    # E2EL / TTFT
    rows.append(["E2EL", STAGE_NAME] + _stats_row_constant(m.latency, 1, _fmt_ms))
    rows.append(["TTFT", STAGE_NAME] + _stats_row_constant(m.ttft, 1, _fmt_ms))

    # TPOT: ais_bench reports N = number of requests with decode latency
    # (output_tokens > 1). For a single request we use 1 when valid else 0.
    tpot_n = 1 if m.output_tokens > 1 else 0
    rows.append(["TPOT", STAGE_NAME] + _stats_row_constant(m.tpot, tpot_n, _fmt_ms))

    # ITL: per-token inter-arrival latencies. ais_bench treats this as an
    # array metric, so we report percentiles across the actual ITL samples.
    itl_n = 1 if m.itl.size > 0 else 0
    rows.append(["ITL", STAGE_NAME] + _stats_row_array(m.itl, itl_n, _fmt_ms))

    rows.append(["InputTokens", STAGE_NAME] + _stats_row_int(m.input_tokens, 1))
    rows.append(["OutputTokens", STAGE_NAME] + _stats_row_int(m.output_tokens, 1))
    rows.append(
        ["OutputTokenThroughput", STAGE_NAME]
        + _stats_row_constant(m.generate_tokens_speed, 1, _fmt_tokens_per_s)
    )
    return rows


def build_common_table(m: RequestMetrics) -> List[List[str]]:
    """Build the common-metric (key/value) table.

    Order and units follow ais_bench's _add_units_to_common_metrics. The Stage
    column and request-count / concurrency rows are omitted for a cleaner demo
    display.
    """
    rows: List[List[str]] = [["Common Metric", "Value"]]

    rows.append(["Benchmark Duration", _fmt_ms(m.infer_time)])
    rows.append(
        ["Request Throughput", _fmt_req_per_s(m.request_throughput)]
    )
    rows.append(["Total Input Tokens", str(m.total_input_tokens)])
    if m.prefill_token_throughput is not None:
        rows.append(
            [
                "Prefill Token Throughput",
                _fmt_tokens_per_s(m.prefill_token_throughput),
            ]
        )
    rows.append(
        ["Total Generated Tokens", str(m.total_generated_tokens)]
    )
    rows.append(
        [
            "Input Token Throughput",
            _fmt_tokens_per_s(m.input_token_throughput),
        ]
    )
    rows.append(
        [
            "Output Token Throughput",
            _fmt_tokens_per_s(m.output_token_throughput),
        ]
    )
    rows.append(
        [
            "Total Token Throughput",
            _fmt_tokens_per_s(m.total_token_throughput),
        ]
    )
    return rows


def render_table(rows: List[List[str]]) -> str:
    return tabulate(
        rows,
        headers="firstrow",
        tablefmt="fancy_grid",
        floatfmt=".2f",
        numalign="center",
        stralign="left",
        missingval="N/A",
    )


def render_metrics(m: RequestMetrics) -> str:
    """Return both tables concatenated, ready to print."""
    parts: List[str] = []
    parts.append("Performance Results (per-request, ais_bench formulas):")
    parts.append(render_table(build_request_table(m)))
    parts.append("")
    parts.append(render_table(build_common_table(m)))
    return "\n".join(parts)
