"""LLM backend base class and stub implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..contracts import Message


FinishReason = Literal["stop", "length", "error", "filtered"]


class LLMCallRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float | None = None
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMCallResponse(BaseModel):
    provider: str
    model: str
    content: str
    finish_reason: FinishReason
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseBackend(ABC):
    provider: str = "base"

    @abstractmethod
    async def call(
        self, req: LLMCallRequest, *, timeout_s: float,
    ) -> LLMCallResponse: ...

    @abstractmethod
    async def healthcheck(self) -> bool: ...


__all__ = [
    "FinishReason",
    "LLMCallRequest", "LLMCallResponse",
    "BaseBackend",
]
