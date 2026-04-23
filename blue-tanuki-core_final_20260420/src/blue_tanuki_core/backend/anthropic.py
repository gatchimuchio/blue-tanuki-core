"""Anthropic backend: /v1/messages.

- httpx.AsyncClient with a connection pool
- tenacity retry on 429 / 5xx / network; fail-fast on other 4xx
- maps Anthropic response to LLMCallResponse
- seed is not natively supported; we attach it to metadata for audit replay
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ..contracts import Message
from ..errors import LLMInvalidResponse, LLMProviderError, LLMTimeout
from .base import BaseBackend, LLMCallRequest, LLMCallResponse


_ANTHROPIC_VERSION = "2023-06-01"


class _Retryable(Exception):
    """Internal signal to retry."""


class AnthropicBackend(BaseBackend):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        *,
        anthropic_version: str = _ANTHROPIC_VERSION,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _to_anthropic_messages(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract leading system message(s); return (system, [{role, content}])."""
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = m.role
            if role not in ("user", "assistant"):
                # collapse tool/module into user-visible context
                role = "user"
            out.append({"role": role, "content": m.content})
        if not out or out[0]["role"] != "user":
            out.insert(0, {"role": "user", "content": ""})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, out

    async def call(
        self, req: LLMCallRequest, *, timeout_s: float,
    ) -> LLMCallResponse:
        system, msgs = self._to_anthropic_messages(req.messages)
        payload: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": msgs,
        }
        if system is not None:
            payload["system"] = system
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.stop:
            payload["stop_sequences"] = req.stop
        if req.seed is not None:
            payload.setdefault("metadata", {})["seed"] = req.seed
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        client = await self._get_client()

        last_retryable: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "/v1/messages",
                    json=payload, headers=headers, timeout=timeout_s,
                )
            except httpx.TimeoutException as e:
                raise LLMTimeout(self.provider, timeout_s) from e
            except httpx.HTTPError as e:
                last_retryable = _Retryable(f"network error: {e}")
            else:
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    last_retryable = _Retryable(f"retryable status {resp.status_code}")
                elif resp.status_code >= 400:
                    raise LLMProviderError(
                        self.provider,
                        f"{resp.status_code}: {resp.text[:500]}",
                        status=resp.status_code,
                    )
                else:
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    return self._parse(resp.json(), req.model, latency_ms)
            if attempt >= self.max_retries:
                break
            await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
        raise LLMProviderError(self.provider, str(last_retryable or "retries exhausted"))

    def _parse(
        self, body: dict[str, Any], model: str, latency_ms: int,
    ) -> LLMCallResponse:
        try:
            parts = body.get("content", [])
            text = "".join(
                p.get("text", "") for p in parts if p.get("type") == "text"
            )
            stop_reason = body.get("stop_reason", "stop")
            finish = {
                "end_turn": "stop", "stop_sequence": "stop",
                "max_tokens": "length", "tool_use": "stop",
            }.get(stop_reason, "stop")
            usage = body.get("usage", {})
            return LLMCallResponse(
                provider=self.provider,
                model=body.get("model", model),
                content=text,
                finish_reason=finish,  # type: ignore[arg-type]
                usage_prompt_tokens=int(usage.get("input_tokens", 0)),
                usage_completion_tokens=int(usage.get("output_tokens", 0)),
                latency_ms=latency_ms,
                raw=body,
            )
        except Exception as e:
            raise LLMInvalidResponse(self.provider, str(e)) from e

    async def healthcheck(self) -> bool:
        try:
            client = await self._get_client()
            r = await client.get(
                "/v1/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.anthropic_version,
                },
                timeout=5,
            )
            return r.status_code < 500
        except Exception:
            return False


__all__ = ["AnthropicBackend"]
