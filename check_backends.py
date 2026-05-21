"""Diagnose connectivity from this PC to the two remote vLLM backends.

Usage:
    python check_backends.py
    python check_backends.py --url-top http://199.147.1.2:8004 --url-bottom http://199.147.1.3:8004
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from presentation_demo.config import DemoConfig
from presentation_demo.health import probe_all
from presentation_demo.web_server import DEFAULT_URL_BOTTOM, DEFAULT_URL_TOP


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe remote vLLM /v1/models")
    parser.add_argument("--url-top", default=DEFAULT_URL_TOP)
    parser.add_argument("--url-bottom", default=DEFAULT_URL_BOTTOM)
    parser.add_argument("--model", default="minimax")
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Honor HTTP_PROXY (default: direct connection)",
    )
    args = parser.parse_args()
    trust_env = args.use_system_proxy

    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        print(f"System proxy env: {proxy!r}  trust_env={trust_env}\n")

    common = dict(model=args.model, trust_env=trust_env)
    panels = [
        ("top", DemoConfig(base_url=args.url_top, **common)),
        ("bottom", DemoConfig(base_url=args.url_bottom, **common)),
    ]
    results = asyncio.run(probe_all(panels))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    ok = all(r.get("ok") for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
