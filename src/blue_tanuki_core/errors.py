"""Typed errors. Everything uncaught is still normalized at ControlPlane edge."""
from __future__ import annotations

from typing import Any

from .contracts import GateDecision


class BlueTanukiError(Exception):
    """Base for all typed errors."""
    pass


# ── Gate signals (not failures; they drive state transitions) ─────────

class GateStopSignal(BlueTanukiError):
    def __init__(self, decision: GateDecision):
        self.decision = decision
        super().__init__(decision.reason or "gate stop")


class GateSuspendSignal(BlueTanukiError):
    def __init__(self, decision: GateDecision, kind: str):
        self.decision = decision
        self.kind = kind  # "approval" | "clarify"
        super().__init__(decision.reason or f"gate suspend {kind}")


# ── Module errors ────────────────────────────────────────────────────

class ModuleRefused(BlueTanukiError):
    def __init__(self, module: str, reason: str, typed: str | None = None):
        self.module = module
        self.reason = reason
        self.typed = typed
        super().__init__(f"module {module} refused: {reason}")


class ModuleTimeout(BlueTanukiError):
    def __init__(self, module: str, timeout_s: float):
        self.module = module
        self.timeout_s = timeout_s
        super().__init__(f"module {module} timed out after {timeout_s}s")


class ModuleCrash(BlueTanukiError):
    def __init__(self, module: str, detail: str):
        self.module = module
        self.detail = detail
        super().__init__(f"module {module} crashed: {detail}")


class ModuleNotFound(BlueTanukiError):
    def __init__(self, module: str):
        self.module = module
        super().__init__(f"module {module} not registered")


# ── LLM errors ───────────────────────────────────────────────────────

class LLMBudgetExceeded(BlueTanukiError):
    def __init__(self, detail: str):
        super().__init__(f"budget exceeded: {detail}")


class LLMPromptTooLarge(BlueTanukiError):
    def __init__(self, size: int, cap: int):
        self.size = size
        self.cap = cap
        super().__init__(f"prompt size {size} exceeds cap {cap}")


class LLMProviderError(BlueTanukiError):
    def __init__(self, provider: str, detail: str, status: int | None = None):
        self.provider = provider
        self.status = status
        super().__init__(f"llm provider {provider} error: {detail}")


class LLMTimeout(BlueTanukiError):
    def __init__(self, provider: str, timeout_s: float):
        self.provider = provider
        self.timeout_s = timeout_s
        super().__init__(f"llm provider {provider} timed out after {timeout_s}s")


class LLMInvalidResponse(BlueTanukiError):
    def __init__(self, provider: str, detail: str):
        self.provider = provider
        super().__init__(f"llm provider {provider} invalid response: {detail}")


# ── Config / runtime ─────────────────────────────────────────────────

class ConfigError(BlueTanukiError):
    def __init__(self, key: str, detail: str = ""):
        self.key = key
        super().__init__(f"config error: {key}: {detail}".rstrip(": "))


class PendingNotFound(BlueTanukiError):
    def __init__(self, token: str):
        self.token = token
        super().__init__(f"pending token not found: {token}")


class PendingExpired(BlueTanukiError):
    def __init__(self, token: str):
        self.token = token
        super().__init__(f"pending token expired: {token}")


class PendingConflict(BlueTanukiError):
    """Raised when the same token is used twice (one-shot violation)."""
    def __init__(self, token: str):
        self.token = token
        super().__init__(f"pending token already consumed: {token}")


class AuditIntegrityError(BlueTanukiError):
    def __init__(self, detail: str):
        super().__init__(f"audit integrity error: {detail}")


__all__ = [
    "BlueTanukiError",
    "GateStopSignal", "GateSuspendSignal",
    "ModuleRefused", "ModuleTimeout", "ModuleCrash", "ModuleNotFound",
    "LLMBudgetExceeded", "LLMPromptTooLarge",
    "LLMProviderError", "LLMTimeout", "LLMInvalidResponse",
    "ConfigError",
    "PendingNotFound", "PendingExpired", "PendingConflict",
    "AuditIntegrityError",
]
