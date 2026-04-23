"""LLMPipe: orchestrates backends with guards and failover.

Responsibilities (this layer):
- enforce audit-mode preconditions (temperature=0, seed set)
- enforce prompt size cap
- enforce per-turn budget (tokens / USD)
- provider failover (sequential, fail-fast on 4xx, retry on 5xx/429/network)
- telemetry hook (log only at this stage; metrics sink is out of scope)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .backend.base import BaseBackend, LLMCallRequest, LLMCallResponse
from .errors import (
    ConfigError, LLMBudgetExceeded, LLMPromptTooLarge,
    LLMProviderError, LLMTimeout,
)


# ──────────────────────────────────────────────────────────────────────
# Guards
# ──────────────────────────────────────────────────────────────────────

class PromptSizeGuard:
    def __init__(self, max_input_tokens: int):
        self.max_input_tokens = max_input_tokens

    def check(self, req: LLMCallRequest) -> None:
        # crude estimate: 4 chars per token. replace with tokenizer once wired.
        estimate = sum(len(m.content) for m in req.messages) // 4
        if estimate > self.max_input_tokens:
            raise LLMPromptTooLarge(estimate, self.max_input_tokens)


class BudgetGuard:
    def __init__(
        self,
        max_tokens_per_turn: int | None = None,
        max_usd_per_turn: float | None = None,
        usd_per_1k_prompt: float = 0.0,
        usd_per_1k_completion: float = 0.0,
    ):
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_usd_per_turn = max_usd_per_turn
        self.usd_per_1k_prompt = usd_per_1k_prompt
        self.usd_per_1k_completion = usd_per_1k_completion
        self._tokens_used = 0
        self._usd_spent = 0.0

    def reset_turn(self) -> None:
        self._tokens_used = 0
        self._usd_spent = 0.0

    def check_pre(self, req: LLMCallRequest) -> None:
        if self.max_tokens_per_turn is not None:
            remaining = self.max_tokens_per_turn - self._tokens_used
            if remaining <= 0:
                raise LLMBudgetExceeded(
                    f"tokens exhausted ({self._tokens_used}/{self.max_tokens_per_turn})"
                )

    def account(self, resp: LLMCallResponse) -> None:
        used = resp.usage_prompt_tokens + resp.usage_completion_tokens
        self._tokens_used += used
        self._usd_spent += (
            resp.usage_prompt_tokens * self.usd_per_1k_prompt / 1000
            + resp.usage_completion_tokens * self.usd_per_1k_completion / 1000
        )
        if self.max_tokens_per_turn is not None and self._tokens_used > self.max_tokens_per_turn:
            raise LLMBudgetExceeded(
                f"tokens {self._tokens_used} > cap {self.max_tokens_per_turn}"
            )
        if self.max_usd_per_turn is not None and self._usd_spent > self.max_usd_per_turn:
            raise LLMBudgetExceeded(
                f"usd {self._usd_spent:.4f} > cap {self.max_usd_per_turn}"
            )


# ──────────────────────────────────────────────────────────────────────
# Telemetry (minimal)
# ──────────────────────────────────────────────────────────────────────

class Telemetry(Protocol):
    def on_success(self, provider: str, resp: LLMCallResponse) -> None: ...
    def on_failure(self, provider: str, exc: Exception) -> None: ...


class NullTelemetry:
    def on_success(self, provider: str, resp: LLMCallResponse) -> None:
        return

    def on_failure(self, provider: str, exc: Exception) -> None:
        return


# ──────────────────────────────────────────────────────────────────────
# LLMPipe
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LLMPipe:
    backends: list[BaseBackend]
    budget: BudgetGuard
    size_guard: PromptSizeGuard
    telemetry: Telemetry = NullTelemetry()
    default_timeout_s: float = 60.0

    async def call(
        self,
        req: LLMCallRequest,
        *,
        audit_mode: bool = False,
        timeout_s: float | None = None,
    ) -> LLMCallResponse:
        if audit_mode:
            if req.temperature != 0:
                raise ConfigError(
                    "llm.audit_mode",
                    "audit_mode requires temperature=0",
                )
            if req.seed is None:
                raise ConfigError(
                    "llm.audit_mode",
                    "audit_mode requires seed to be set",
                )
        self.size_guard.check(req)
        self.budget.check_pre(req)

        to = timeout_s or self.default_timeout_s
        last_exc: Exception | None = None
        for backend in self.backends:
            start = time.perf_counter()
            try:
                resp = await backend.call(req, timeout_s=to)
                self.budget.account(resp)
                self.telemetry.on_success(backend.provider, resp)
                return resp
            except (LLMProviderError, LLMTimeout) as e:
                self.telemetry.on_failure(backend.provider, e)
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise LLMProviderError("pipe", "no backend configured")


__all__ = [
    "LLMPipe", "BudgetGuard", "PromptSizeGuard",
    "Telemetry", "NullTelemetry",
]
