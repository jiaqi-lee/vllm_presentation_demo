"""Asynchronous streaming client for the vLLM /v1/chat/completions endpoint.

The client is designed so the perceived speed of the model is *not* throttled
by terminal I/O:

* A producer coroutine reads SSE chunks from the server and, the moment it
  parses a non-empty content delta, records a perf_counter() timestamp into
  RequestMetrics and pushes the text into an unbounded asyncio.Queue using a
  non-blocking put_nowait. It never waits on stdout.
* A consumer coroutine drains the queue and writes to stdout. If stdout is
  slow it only delays the *display* of tokens, never the measurement of TTFT
  / TPOT / ITL.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional, TextIO

import httpx

from presentation_demo.config import DemoConfig
from presentation_demo.metrics import RequestMetrics


class StreamingError(RuntimeError):
    """Raised when the upstream HTTP request fails in a non-recoverable way."""


class StreamingChatClient:
    """Async streaming client; one instance can be reused across REPL turns."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        timeout = httpx.Timeout(
            connect=10.0,
            read=config.timeout,
            write=30.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "StreamingChatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def stream(
        self,
        prompt: str,
        out: Optional[TextIO] = None,
    ) -> RequestMetrics:
        """Send one streaming chat completion and return the final metrics.

        Args:
            prompt: User-provided prompt; substituted into messages[0].content.
            out: Output stream to write tokens to. Defaults to sys.stdout.

        Returns:
            A finalized RequestMetrics object. On failure, `success` is False
            and `error_info` describes the problem; the function does not
            raise for HTTP-level errors so the REPL stays alive.
        """
        sink = out if out is not None else sys.stdout
        metrics = RequestMetrics()
        payload = self.config.build_payload(prompt)
        url = self.config.chat_completions_url

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        producer_failed: list[BaseException] = []

        async def producer() -> None:
            try:
                metrics.record_time_point()  # time_points[0] = pre-send timestamp
                async with self._client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise StreamingError(
                            f"HTTP {response.status_code} from {url}: {body.strip()[:500]}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            # Comments, empty keep-alives, or non-SSE noise.
                            continue
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            # Skip malformed chunks rather than killing the stream.
                            continue
                        self._consume_chunk(obj, metrics, queue)
            except BaseException as exc:  # incl. CancelledError
                producer_failed.append(exc)
                raise
            finally:
                queue.put_nowait(None)

        async def consumer() -> None:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                sink.write(chunk)
                sink.flush()

        try:
            await asyncio.gather(producer(), consumer())
        except asyncio.CancelledError:
            metrics.success = False
            metrics.error_info = "cancelled by user"
            raise
        except StreamingError as exc:
            metrics.success = False
            metrics.error_info = str(exc)
            return metrics
        except httpx.HTTPError as exc:
            metrics.success = False
            metrics.error_info = f"{type(exc).__name__}: {exc}"
            return metrics
        except Exception as exc:  # noqa: BLE001 - keep REPL alive on unknowns
            metrics.success = False
            metrics.error_info = f"{type(exc).__name__}: {exc}"
            return metrics

        if producer_failed:
            # Should already have been re-raised above, but guard for safety.
            metrics.success = False
            metrics.error_info = str(producer_failed[0])
            return metrics

        metrics.finalize()
        return metrics

    @staticmethod
    def _consume_chunk(
        obj: dict,
        metrics: RequestMetrics,
        queue: "asyncio.Queue[Optional[str]]",
    ) -> None:
        """Parse one SSE chunk, recording timing + token counts.

        Matches the field layout used by vLLM's OpenAI-compatible streaming
        API and ais_bench's VLLMCustomAPIChat.parse_stream_response.
        """
        for choice in obj.get("choices", []) or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                # IMPORTANT: record the receive timestamp *before* the put,
                # so even a backlogged consumer cannot bias TTFT / ITL.
                metrics.record_time_point()
                queue.put_nowait(text)
            # Some servers stream reasoning content separately; surface it too
            # so the user can see what's happening, but do not count it twice
            # (token counts come from the final `usage` block).
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                metrics.record_time_point()
                queue.put_nowait(reasoning)
        usage = obj.get("usage")
        if usage:
            metrics.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            metrics.output_tokens = int(usage.get("completion_tokens", 0) or 0)
