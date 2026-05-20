"""Interactive streaming demo for vllm-ascend presentation.

A REPL-like terminal that forwards user prompts to a running vLLM server using
the OpenAI-compatible chat completions API in streaming mode, prints tokens as
they arrive, and finally reports per-request performance metrics using the
exact formulas adopted by the ais_bench benchmark suite.
"""

from presentation_demo.config import DemoConfig
from presentation_demo.metrics import RequestMetrics

# StreamingChatClient is imported lazily so consumers that only need the
# metrics layer (e.g. unit tests, offline analysis) do not have to install
# httpx just to import this package.
__all__ = ["DemoConfig", "RequestMetrics"]
