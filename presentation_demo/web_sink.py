"""TextIO sink that forwards streamed tokens to a WebSocket (or similar) channel."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, TextIO


TokenSender = Callable[[str], Awaitable[None]]


class WebTokenSink(TextIO):
    """Minimal TextIO adapter used by StreamingChatClient.stream(out=...)."""

    def __init__(self, send: TokenSender) -> None:
        self._send = send
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def write(self, s: str) -> int:
        if not s:
            return 0
        loop = self._loop or asyncio.get_running_loop()
        self._loop = loop
        loop.create_task(self._send(s))
        return len(s)

    def flush(self) -> None:
        return None

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True
