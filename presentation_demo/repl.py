"""REPL main loop for the streaming demo.

Provides an interactive `>>>`-style prompt where each line of input is sent to
the configured vLLM service as a one-shot streaming chat completion. After the
stream completes, an ais_bench-style performance table is printed and the user
is returned to the prompt for the next question.

Control commands (lines starting with `/`):
    /help                Show available commands.
    /config              Print the current configuration.
    /set <key> <value>   Update a settable config field (e.g. /set max_tokens 500).
    /exit, /quit, /q     Exit the demo.

Ctrl+C during a streaming request cancels that single request and returns to
the prompt. A second Ctrl+C at the empty prompt exits the program.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Awaitable, Callable, List, Optional

from presentation_demo.client import StreamingChatClient
from presentation_demo.config import DemoConfig, build_arg_parser, config_from_args
from presentation_demo.metrics import RequestMetrics
from presentation_demo.table import render_metrics


PROMPT = ">>> "
BANNER = r"""
================================================================
 vllm-ascend Streaming Presentation Demo
 - type a question and press Enter to stream a response
 - type /help for commands, /exit to quit
================================================================
""".strip()


def _print_banner(cfg: DemoConfig) -> None:
    print(BANNER)
    print(f"Target: {cfg.chat_completions_url}  (model={cfg.model})")
    print(
        f"max_tokens={cfg.max_tokens}  temperature={cfg.temperature}  "
        f"enable_thinking={cfg.enable_thinking}  timeout={cfg.timeout}s"
    )
    print()


def _print_help() -> None:
    print(
        "Commands:\n"
        "  /help                Show this help text\n"
        "  /config              Print current configuration\n"
        "  /set <key> <value>   Update a config field. Keys: base_url, model, "
        "max_tokens, temperature, enable_thinking, timeout\n"
        "  /exit, /quit, /q     Exit\n"
        "Any other line is sent as a prompt to the vLLM service."
    )


def _print_config(cfg: DemoConfig) -> None:
    print(json.dumps(cfg.to_display_dict(), indent=2, ensure_ascii=False))


def _handle_set(cfg: DemoConfig, rest: str) -> None:
    parts = rest.strip().split(maxsplit=1)
    if len(parts) != 2:
        print("usage: /set <key> <value>")
        return
    key, value = parts[0], parts[1]
    try:
        cfg.update_field(key, value)
    except ValueError as exc:
        print(f"error: {exc}")
        return
    print(f"ok: {key} = {getattr(cfg, key)!r}")


def _is_exit_command(line: str) -> bool:
    return line.strip().lower() in {"/exit", "/quit", "/q"}


async def _run_one_prompt(client: StreamingChatClient, prompt: str) -> RequestMetrics:
    """Send one prompt, streaming output to stdout, and return finalized metrics.

    Cancellation (KeyboardInterrupt -> CancelledError) is propagated up so the
    REPL can surface a clean message and continue.
    """
    sys.stdout.write("\n")  # separate user input from streamed reply
    sys.stdout.flush()
    metrics = await client.stream(prompt)
    sys.stdout.write("\n\n")  # ensure the table starts on a fresh line
    sys.stdout.flush()
    return metrics


async def _async_main(cfg: DemoConfig) -> int:
    async with StreamingChatClient(cfg) as client:
        loop = asyncio.get_running_loop()
        _print_banner(cfg)

        while True:
            try:
                line = await loop.run_in_executor(None, _read_line, PROMPT)
            except EOFError:
                print()  # newline after ^D
                return 0
            except KeyboardInterrupt:
                # ^C at an empty prompt: exit cleanly.
                print("\n(received Ctrl+C at prompt, exiting)")
                return 0

            if line is None:
                return 0

            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("/"):
                cmd, _, rest = stripped.partition(" ")
                cmd_lower = cmd.lower()
                if cmd_lower in {"/exit", "/quit", "/q"}:
                    return 0
                if cmd_lower == "/help":
                    _print_help()
                    continue
                if cmd_lower == "/config":
                    _print_config(cfg)
                    continue
                if cmd_lower == "/set":
                    _handle_set(cfg, rest)
                    continue
                print(f"unknown command: {cmd}. Type /help for a list of commands.")
                continue

            # Otherwise treat the whole line as a prompt.
            task = asyncio.create_task(_run_one_prompt(client, stripped))
            try:
                metrics = await _await_with_ctrl_c(task)
            except KeyboardInterrupt:
                print("\n(request cancelled)")
                continue

            if not metrics.success:
                print(f"[error] request failed: {metrics.error_info}")
                continue
            print(render_metrics(metrics))


async def _await_with_ctrl_c(task: "asyncio.Task[RequestMetrics]") -> RequestMetrics:
    """Wait for task; on Ctrl+C, cancel it and propagate KeyboardInterrupt.

    Implemented by shielding the task and racing it against a future tied to
    the SIGINT handler. We use a plain try/except on KeyboardInterrupt because
    asyncio raises it into the coroutine on Windows when ^C is pressed.
    """
    try:
        return await task
    except KeyboardInterrupt:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, KeyboardInterrupt, Exception):
            pass
        raise


def _read_line(prompt: str) -> Optional[str]:
    """Synchronous readline that returns None on EOF.

    Run in an executor so the asyncio event loop stays responsive.
    """
    try:
        return input(prompt)
    except EOFError:
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    try:
        return asyncio.run(_async_main(cfg))
    except KeyboardInterrupt:
        # Final safety net so a ^C while asyncio is shutting down still
        # produces a clean exit code instead of a stack trace.
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
