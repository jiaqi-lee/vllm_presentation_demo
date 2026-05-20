"""Entry point for the vllm-ascend streaming presentation demo.

Usage:
    python demo.py [--url http://127.0.0.1:8004] [--model minimax] \
                   [--max-tokens 200] [--temperature 0] [--enable-thinking]

The script enters an interactive REPL. Each line you type is forwarded to the
configured vLLM service as a streaming chat-completions request (exactly the
curl form documented in 需求描述.txt), tokens are printed to your terminal as
they arrive, and a per-request ais_bench-style performance table is printed
after the stream completes.
"""

from __future__ import annotations

import sys

from presentation_demo.repl import main


if __name__ == "__main__":
    sys.exit(main())
