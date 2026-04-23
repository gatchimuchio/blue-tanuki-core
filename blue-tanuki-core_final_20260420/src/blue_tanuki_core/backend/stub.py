"""StubBackend: deterministic, seed-aware, zero-IO.

Contract:
- echoes the concatenation of user messages
- with audit_mode-compatible behavior: temperature=0 and seed fixed yields
  byte-for-byte identical output
- latency_ms is derived from len(content) to keep audit records stable
"""
from __future__ import annotations

import hashlib

from .base import BaseBackend, LLMCallRequest, LLMCallResponse


class StubBackend(BaseBackend):
    provider = "stub"

    def __init__(self, prefix: str = "[stub]"):
        self.prefix = prefix

    async def call(
        self, req: LLMCallRequest, *, timeout_s: float,
    ) -> LLMCallResponse:
        # echo every non-assistant message so tool/system context is visible
        visible = [m.content for m in req.messages if m.role != "assistant"]
        base = " | ".join(visible)
        seed_str = str(req.seed) if req.seed is not None else "none"
        # deterministic salt: audit-mode replay must be byte-equal
        digest = hashlib.sha256(
            f"{req.model}|{seed_str}|{req.temperature}|{base}".encode("utf-8")
        ).hexdigest()[:8]
        content = f"{self.prefix} {base} <{digest}>"
        return LLMCallResponse(
            provider=self.provider,
            model=req.model,
            content=content,
            finish_reason="stop",
            usage_prompt_tokens=sum(len(m.content) for m in req.messages) // 4,
            usage_completion_tokens=len(content) // 4,
            latency_ms=len(content),
        )

    async def healthcheck(self) -> bool:
        return True


__all__ = ["StubBackend"]
