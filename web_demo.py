"""Web entry point for the vllm-ascend streaming presentation demo.

Usage (on your local PC, forwarding to two remote vLLM servers):

    python web_demo.py --url-top http://199.147.1.2:8004 \
                       --url-bottom http://199.147.1.3:8004

Then open http://127.0.0.1:8765 in a browser on the same PC.

The terminal REPL remains available via ``python demo.py`` unchanged.
"""

from __future__ import annotations

import sys

from presentation_demo.web_server import main


if __name__ == "__main__":
    sys.exit(main())
