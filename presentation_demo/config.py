"""Runtime configuration for the streaming demo.

Defaults mirror the curl template in the requirements doc:

    curl http://127.0.0.1:8004/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "minimax",
            "messages": [{"role": "user", "content": "..."}],
            "stream": true,
            "temperature": 0,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": false}
        }'
"""

from __future__ import annotations

import argparse
import socket
from dataclasses import dataclass, field, asdict
from typing import Any, Dict
from urllib.parse import urljoin


#DEFAULT_BASE_URL = "http://127.0.0.1:8004"
CURRENT_IP = socket.gethostbyname(socket.gethostname())
DEFAULT_BASE_URL = f"http://{CURRENT_IP}:8004"
DEFAULT_MODEL = "minimax"
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_ENABLE_THINKING = False
DEFAULT_TIMEOUT = 600.0  # seconds; long enough for big outputs, short enough to not hang forever


@dataclass
class DemoConfig:
    """Mutable configuration object shared across REPL turns."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    timeout: float = DEFAULT_TIMEOUT
    extra_generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url if self.base_url.endswith("/") else self.base_url + "/"
        return urljoin(base, "v1/chat/completions")

    def build_payload(self, prompt: str) -> Dict[str, Any]:
        """Build the JSON body that matches the curl template exactly.

        The user's prompt is substituted into messages[0].content; everything
        else is taken from configuration so the request remains a faithful
        forward of the original curl call.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
            # stream_options.include_usage lets vLLM emit the final usage block
            # with prompt_tokens / completion_tokens in the last SSE chunk,
            # mirroring ais_bench's VLLMCustomAPIChat behaviour.
            "stream_options": {"include_usage": True},
        }
        payload.update(self.extra_generation_kwargs)
        return payload

    def to_display_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chat_completions_url"] = self.chat_completions_url
        return d

    SETTABLE_KEYS = {
        "base_url": str,
        "model": str,
        "max_tokens": int,
        "temperature": float,
        "enable_thinking": "bool",
        "timeout": float,
    }

    def update_field(self, key: str, raw_value: str) -> None:
        """Update a settable field by parsing the raw string value.

        Raises ValueError for unknown keys or unparsable values so the REPL
        can surface a clean error message to the user.
        """
        if key not in self.SETTABLE_KEYS:
            raise ValueError(
                f"unknown config key: {key!r}. Settable: {sorted(self.SETTABLE_KEYS)}"
            )
        kind = self.SETTABLE_KEYS[key]
        if kind == "bool":
            value: Any = _parse_bool(raw_value)
        else:
            value = kind(raw_value)
        setattr(self, key, value)


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean from {raw!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo",
        description=(
            "Interactive streaming demo that forwards each terminal prompt to a "
            "running vLLM service and prints tokens as they arrive, then reports "
            "per-request TTFT / TPOT / RPS / TPS using ais_bench formulas."
        ),
    )
    parser.add_argument(
        "--url",
        dest="base_url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the vLLM service (default: {DEFAULT_BASE_URL}). "
        "The demo appends '/v1/chat/completions' internally.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name passed to the API (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens for each request (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="Forward chat_template_kwargs.enable_thinking=true. "
        "Default is false, matching the original curl template.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP read timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    return DemoConfig(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        timeout=args.timeout,
    )
