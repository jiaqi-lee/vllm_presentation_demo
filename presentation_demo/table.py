"""Pretty-print per-request metrics in the ais_bench table style.

A single Common Metric table is rendered with `tabulate` using
`tablefmt="fancy_grid"`, matching ais_bench's DefaultPerfSummarizer output
style for the common-metrics section.

Columns: Common Metric | Value

Rows: E2EL / TTFT / TPOT / Request Throughput / Total Input Tokens /
      Total Generated Tokens / Total Token Throughput

Units match ais_bench: time is ms (SECOND_TO_MILLISECOND), throughput is
`token/s` or `req/s`.
"""

from __future__ import annotations

from typing import List

from tabulate import tabulate

from presentation_demo.metrics import RequestMetrics, SECOND_TO_MILLISECOND


def _fmt_ms(seconds: float) -> str:
    return f"{round(seconds * SECOND_TO_MILLISECOND, 4)} ms"


def _fmt_tokens_per_s(value: float) -> str:
    return f"{round(value, 4)} token/s"


def _fmt_req_per_s(value: float) -> str:
    return f"{round(value, 4)} req/s"


def build_common_table(m: RequestMetrics) -> List[List[str]]:
    """Build the common-metric (key/value) table."""
    rows: List[List[str]] = [["Common Metric", "Value"]]

    # E2EL Average: end-to-end latency (time_points[-1] - time_points[0]).
    rows.append(["E2EL", _fmt_ms(m.latency)])
    rows.append(["TTFT", _fmt_ms(m.ttft)])
    rows.append(["TPOT", _fmt_ms(m.tpot)])
    rows.append(["Total Input Tokens", str(m.total_input_tokens)])
    rows.append(["Total Generated Tokens", str(m.total_generated_tokens)])
    rows.append(["Request Throughput", _fmt_req_per_s(m.request_throughput)])
    rows.append(
        ["Total Token Throughput", _fmt_tokens_per_s(m.total_token_throughput)]
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
    """Return the Common Metric table, ready to print."""
    return render_table(build_common_table(m))
