"""FastAPI web front-end for the streaming presentation demo.

Serves a dual-panel browser UI. The local ``web_demo.py`` process proxies
streaming requests to remote vLLM backends (configured via --url-top /
--url-bottom); the browser only talks to localhost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

import os

from presentation_demo.client import StreamingChatClient
from presentation_demo.config import DemoConfig, build_arg_parser
from presentation_demo.health import probe_all
from presentation_demo.table import render_metrics
from presentation_demo.web_sink import WebTokenSink

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Default remote backends for side-by-side comparison at exhibitions.
DEFAULT_URL_TOP = "http://199.147.1.2:8004"
DEFAULT_URL_BOTTOM = "http://199.147.1.3:8004"

PANEL_TOP = "top"
PANEL_BOTTOM = "bottom"

# Exhibition-facing panel titles (hide real backend host:port in the browser).
PANEL_LABEL_TOP = "不开启特性"
PANEL_LABEL_BOTTOM = "开启特性后"


@dataclass(frozen=True)
class PanelBackend:
    """One UI panel bound to a single vLLM base URL."""

    panel_id: str
    label: str
    config: DemoConfig


def _panel_summary(panel: PanelBackend) -> Dict[str, Any]:
    d = panel.config.to_display_dict()
    d["panel"] = panel.panel_id
    d["label"] = panel.label
    return d


def build_dual_panels(
    *,
    url_top: str,
    url_bottom: str,
    model: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout: float,
    trust_env: bool = False,
) -> List[PanelBackend]:
    common = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        timeout=timeout,
        trust_env=trust_env,
    )
    return [
        PanelBackend(
            panel_id=PANEL_TOP,
            label=PANEL_LABEL_TOP,
            config=DemoConfig(base_url=url_top, **common),
        ),
        PanelBackend(
            panel_id=PANEL_BOTTOM,
            label=PANEL_LABEL_BOTTOM,
            config=DemoConfig(base_url=url_bottom, **common),
        ),
    ]


def create_app(panels: List[PanelBackend]) -> FastAPI:
    app = FastAPI(
        title="vllm-ascend Presentation Demo",
        description="Dual-panel web UI for comparing two remote vLLM backends.",
    )
    app.state.panels = {p.panel_id: p for p in panels}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return {
            "mode": "dual",
            "panels": {
                pid: _panel_summary(p) for pid, p in app.state.panels.items()
            },
        }

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        panel_map: Dict[str, PanelBackend] = app.state.panels
        probes = await probe_all(
            [(pid, p.config) for pid, p in panel_map.items()]
        )
        return {"panels": probes}

    @app.websocket("/ws")
    async def chat_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        panel_map: Dict[str, PanelBackend] = app.state.panels
        active: Dict[str, asyncio.Task[Any]] = {}

        clients: Dict[str, StreamingChatClient] = {
            pid: StreamingChatClient(p.config) for pid, p in panel_map.items()
        }
        try:
            async def send_json(payload: Dict[str, Any]) -> None:
                await websocket.send_text(
                    json.dumps(payload, ensure_ascii=False)
                )

            async def run_panel(panel_id: str, prompt: str) -> None:
                panel = panel_map[panel_id]
                client = clients[panel_id]

                async def send_token(text: str) -> None:
                    await send_json(
                        {"type": "token", "panel": panel_id, "text": text}
                    )

                sink = WebTokenSink(send_token)
                try:
                    metrics = await client.stream(prompt, out=sink)
                except asyncio.CancelledError:
                    await send_json({"type": "cancelled", "panel": panel_id})
                    return
                except Exception as exc:  # noqa: BLE001
                    await send_json(
                        {
                            "type": "error",
                            "panel": panel_id,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    return

                if not metrics.success:
                    await send_json(
                        {
                            "type": "error",
                            "panel": panel_id,
                            "message": metrics.error_info or "request failed",
                        }
                    )
                    return

                await send_json(
                    {
                        "type": "done",
                        "panel": panel_id,
                        "table": render_metrics(metrics),
                        "metrics": {
                            "ttft_ms": round(metrics.ttft * 1000, 4),
                            "tpot_ms": round(metrics.tpot * 1000, 4),
                            "e2el_ms": round(metrics.latency * 1000, 4),
                            "input_tokens": metrics.input_tokens,
                            "output_tokens": metrics.output_tokens,
                            "request_throughput": round(
                                metrics.request_throughput, 4
                            ),
                            "total_token_throughput": round(
                                metrics.total_token_throughput, 4
                            ),
                        },
                    }
                )

            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await send_json({"type": "error", "message": "invalid JSON"})
                    continue

                action = msg.get("action", "chat")

                if action == "cancel":
                    for task in list(active.values()):
                        if not task.done():
                            task.cancel()
                    active.clear()
                    continue

                if action == "config":
                    await send_json(
                        {
                            "type": "config",
                            "data": {
                                "mode": "dual",
                                "panels": {
                                    pid: _panel_summary(p)
                                    for pid, p in panel_map.items()
                                },
                            },
                        }
                    )
                    continue

                if action != "chat":
                    await send_json(
                        {
                            "type": "error",
                            "message": f"unknown action: {action}",
                        }
                    )
                    continue

                prompt = str(msg.get("prompt", "")).strip()
                if not prompt:
                    await send_json({"type": "error", "message": "empty prompt"})
                    continue

                if any(not t.done() for t in active.values()):
                    await send_json(
                        {
                            "type": "error",
                            "message": "a request is already in progress",
                        }
                    )
                    continue

                for panel_id, panel in panel_map.items():
                    await send_json(
                        {
                            "type": "start",
                            "panel": panel_id,
                            "prompt": prompt,
                            "label": panel.label,
                        }
                    )
                    active[panel_id] = asyncio.create_task(
                        run_panel(panel_id, prompt)
                    )

                results = await asyncio.gather(
                    *active.values(), return_exceptions=True
                )
                active.clear()
                for exc in results:
                    if isinstance(exc, Exception):
                        await send_json(
                            {
                                "type": "error",
                                "message": f"{type(exc).__name__}: {exc}",
                            }
                        )

        finally:
            for task in active.values():
                if not task.done():
                    task.cancel()
            for client in clients.values():
                await client.aclose()

    return app


def build_web_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.prog = "web_demo"
    parser.description = (
        "Dual-panel web UI for comparing two remote vLLM backends. "
        "Run on your local PC; the browser connects to localhost while "
        "this process forwards requests to --url-top and --url-bottom."
    )
    parser.set_defaults(base_url=None)  # unused in dual mode
    parser.add_argument(
        "--url-top",
        default=DEFAULT_URL_TOP,
        help=f"vLLM base URL for the upper panel (default: {DEFAULT_URL_TOP}).",
    )
    parser.add_argument(
        "--url-bottom",
        default=DEFAULT_URL_BOTTOM,
        help=(
            f"vLLM base URL for the lower panel (default: {DEFAULT_URL_BOTTOM})."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for the local web server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local web server (default: 8765).",
    )
    return parser


def _print_proxy_notice(trust_env: bool) -> None:
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy and not trust_env:
        print(
            f"  [note] Detected system proxy {proxy!r} — ignored (direct LAN access).\n"
            f"         If you need the proxy, restart with --use-system-proxy.\n"
        )
    elif proxy and trust_env:
        print(f"  [note] Using system proxy: {proxy!r}\n")


async def _print_startup_health(panels: List[PanelBackend]) -> None:
    print("  Checking remote backends (/v1/models)...\n")
    probes = await probe_all([(p.panel_id, p.config) for p in panels])
    for panel in panels:
        r = probes[panel.panel_id]
        label = panel.label
        if r.get("ok"):
            models = r.get("models") or []
            print(f"  [OK]   {label}  ({r.get('latency_ms')} ms)  models={models[:3]}")
            if r.get("hint"):
                print(f"         warning: {r['hint']}")
        else:
            print(f"  [FAIL] {label}")
            print(f"         {r.get('message', 'unknown error')}")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_web_arg_parser()
    args = parser.parse_args(argv)
    trust_env = getattr(args, "use_system_proxy", False)
    panels = build_dual_panels(
        url_top=args.url_top,
        url_bottom=args.url_bottom,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        timeout=args.timeout,
        trust_env=trust_env,
    )

    import uvicorn

    app = create_app(panels)
    top = panels[0]
    bottom = panels[1]
    print(
        f"\n  vllm-ascend Web Demo (dual remote backends)\n"
        f"  Open http://127.0.0.1:{args.port} on this PC\n"
        f"  Upper panel -> {top.config.chat_completions_url}\n"
        f"  Lower panel -> {bottom.config.chat_completions_url}\n"
    )
    _print_proxy_notice(trust_env)
    asyncio.run(_print_startup_health(panels))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
