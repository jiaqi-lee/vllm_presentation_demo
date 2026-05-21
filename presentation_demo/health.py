"""Lightweight connectivity probes for vLLM backends."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from presentation_demo.config import DemoConfig


def _models_url(cfg: DemoConfig) -> str:
    base = cfg.base_url if cfg.base_url.endswith("/") else cfg.base_url + "/"
    return urljoin(base, "v1/models")


def _hint_for_status(status: int, body: str) -> str:
    body_snip = (body or "").strip()[:200]
    if status == 502:
        return (
            "HTTP 502：网关/代理未拿到上游 vLLM 的有效响应。"
            "常见原因：① 远程 vLLM 进程未启动或已崩溃；"
            "② 前面有 Nginx/网关但后端未配置好；"
            "③ 本机开启了 HTTP 代理，请求被代理转发失败"
            "（请确认 web_demo 未加 --use-system-proxy，或清空 HTTP_PROXY 环境变量）。"
            + (f" 响应片段: {body_snip}" if body_snip else " 响应体为空。")
        )
    if status == 503:
        return f"HTTP 503：服务暂时不可用。{body_snip}"
    if status == 504:
        return f"HTTP 504：网关超时，vLLM 可能负载过高或未响应。{body_snip}"
    if status == 404:
        return (
            f"HTTP 404：路径不存在，请确认 base_url 指向 vLLM OpenAI 端口。"
            f" {body_snip}"
        )
    return f"HTTP {status}。{body_snip}" if body_snip else f"HTTP {status}，无响应体。"


def _hint_for_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc)
    if "ConnectTimeout" in name or "connect" in text.lower() and "timeout" in text.lower():
        return (
            f"{name}：无法在限定时间内连上 {text}。"
            "请检查 IP/端口、防火墙、VPN，以及两台机器是否已启动 vLLM。"
        )
    if "ConnectError" in name or "Connection refused" in text:
        return (
            f"{name}：连接被拒绝，目标端口可能没有服务在监听。"
            "请在远程机上确认 vLLM 已绑定 0.0.0.0:8004（而非仅 127.0.0.1）。"
        )
    if "ProxyError" in name:
        return (
            f"{name}：系统 HTTP 代理异常。可尝试清空 HTTP_PROXY/HTTPS_PROXY，"
            "或不要给 web_demo 加 --use-system-proxy。"
        )
    return f"{name}: {text}"


async def probe_backend(
    cfg: DemoConfig,
    *,
    connect_timeout: float = 10.0,
) -> Dict[str, Any]:
    """GET /v1/models to verify the exhibition PC can reach vLLM."""
    url = _models_url(cfg)
    timeout = httpx.Timeout(connect=connect_timeout, read=connect_timeout, write=10.0, pool=5.0)
    t0 = time.perf_counter()
    result: Dict[str, Any] = {
        "ok": False,
        "url": url,
        "base_url": cfg.base_url,
        "latency_ms": None,
        "status_code": None,
        "models": [],
        "message": "",
        "hint": "",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=cfg.trust_env) as client:
            resp = await client.get(url)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["status_code"] = resp.status_code
            body = resp.text
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    models_raw = data.get("data") or []
                    result["models"] = [
                        m.get("id", m) if isinstance(m, dict) else str(m)
                        for m in models_raw
                    ]
                except Exception:
                    result["models"] = []
                result["ok"] = True
                result["message"] = "reachable"
                if result["models"] and cfg.model not in result["models"]:
                    result["hint"] = (
                        f"已连通，但配置的 model={cfg.model!r} 不在 /v1/models 列表中："
                        f"{result['models'][:8]}"
                    )
                return result
            result["message"] = _hint_for_status(resp.status_code, body)
            result["hint"] = result["message"]
            return result
    except httpx.HTTPError as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["message"] = _hint_for_exception(exc)
        result["hint"] = result["message"]
        return result
    except Exception as exc:  # noqa: BLE001
        result["message"] = _hint_for_exception(exc)
        result["hint"] = result["message"]
        return result


async def probe_all(
    panels: List[tuple[str, DemoConfig]],
) -> Dict[str, Any]:
    import asyncio

    async def one(panel_id: str, cfg: DemoConfig) -> tuple[str, Dict[str, Any]]:
        r = await probe_backend(cfg)
        r["panel"] = panel_id
        return panel_id, r

    pairs = await asyncio.gather(*(one(pid, cfg) for pid, cfg in panels))
    return {pid: data for pid, data in pairs}
