"""Typed contracts and ID helpers.

All types crossing layer boundaries are pydantic models. No layer passes
dict or raw str through the control plane surface.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

# ──────────────────────────────────────────────────────────────────────
# ID helpers
# ──────────────────────────────────────────────────────────────────────

def _hex12() -> str:
    return uuid.uuid4().hex[:12]


def new_session_id() -> str:
    return f"ses_{_hex12()}"


def new_request_id() -> str:
    return f"req_{_hex12()}"


def new_turn_id() -> str:
    return f"turn_{_hex12()}"


def new_call_id() -> str:
    return f"call_{_hex12()}"


def new_resume_token() -> str:
    return f"resume_{uuid.uuid4().hex}"


# ──────────────────────────────────────────────────────────────────────
# Role / Direction / Caller
# ──────────────────────────────────────────────────────────────────────

Role = Literal["user", "assistant", "system", "module", "tool"]
Direction = Literal[
    "inbound",
    "pre_module",
    "post_module",
    "pre_llm",
    "post_llm",
]
Caller = Literal["user", "director", "module", "llm"]


# ──────────────────────────────────────────────────────────────────────
# Message
# ──────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    model_config = ConfigDict(frozen=False)
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Module I/O
# ──────────────────────────────────────────────────────────────────────

class TypedPayload(BaseModel):
    """Structured payload used for module dispatch. LLM/user-facing surfaces use Message."""
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class CallContext(BaseModel):
    """Side information for module calls that Gate needs to see."""
    op: str | None = None
    resource: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModuleRequest(BaseModel):
    call_id: str = Field(default_factory=new_call_id)
    parent_call_id: str | None = None
    request_id: str
    session_id: str
    turn_id: str
    module: str
    payload: TypedPayload
    context: CallContext
    caller: Caller


class ModuleStatus(BaseModel):
    code: Literal["ok", "error", "refused", "suspended"]
    reason: str | None = None
    typed_error: str | None = None


class SideEffect(BaseModel):
    kind: str
    target: str
    bytes_in: int = 0
    bytes_out: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class ModuleResponse(BaseModel):
    call_id: str
    status: ModuleStatus
    result: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[SideEffect] = Field(default_factory=list)
    duration_ms: int = 0


# ──────────────────────────────────────────────────────────────────────
# Control I/O
# ──────────────────────────────────────────────────────────────────────

class ControlRequest(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    session_id: str
    turn_id: str = Field(default_factory=new_turn_id)
    message: Message


class ControlStatus(BaseModel):
    code: Literal["ok", "suspended", "stopped", "failed"]
    reason: str | None = None
    resume_token: str | None = None


# ──────────────────────────────────────────────────────────────────────
# Approval / Clarify
# ──────────────────────────────────────────────────────────────────────

class ApprovalRequest(BaseModel):
    token: str
    subject: dict[str, Any]
    reason: str | None = None


class ApprovalResponse(BaseModel):
    token: str
    decision: Literal["approved", "rejected"]
    note: str | None = None


class ClarifyRequest(BaseModel):
    token: str
    prompt: str


class ClarifyResponse(BaseModel):
    token: str
    answer: str


class Pending(BaseModel):
    kind: Literal["approval", "clarify"]
    token: str
    approval: ApprovalRequest | None = None
    clarify: ClarifyRequest | None = None


class ControlResponse(BaseModel):
    request_id: str
    status: ControlStatus
    output: list[Message] = Field(default_factory=list)
    pending: Pending | None = None


# ──────────────────────────────────────────────────────────────────────
# Gate I/O
# ──────────────────────────────────────────────────────────────────────

class GateSubject(BaseModel):
    direction: Direction
    session_id: str
    request_id: str
    turn_id: str
    caller: Caller
    module: str | None = None
    op: str | None = None
    resource: str | None = None
    payload_hash: str
    messages_preview: list[str] = Field(default_factory=list)
    side_effects_preview: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str | None = None
    payload_size: int = 0
    side_effect_bytes_out_total: int = 0
    content: str | None = None  # inbound のときのユーザ入力本文
    extra: dict[str, Any] = Field(default_factory=dict)


class GateDecision(BaseModel):
    action: Literal["pass", "stop", "suspend_approval", "suspend_clarify"]
    reason: str | None = None
    policy_id: str | None = None
    matched_rule_ids: list[str] = Field(default_factory=list)
    clarify_prompt: str | None = None
    approval_subject: dict[str, Any] | None = None


# ──────────────────────────────────────────────────────────────────────
# Canonical JSON / hashing helpers
# ──────────────────────────────────────────────────────────────────────

def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding for hashing. sort_keys + separators fixed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_hash(payload: Any) -> str:
    """sha256 hex of canonical JSON form of redacted payload."""
    return sha256_hex(canonical_json_bytes(payload))


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


__all__ = [
    "Role", "Direction", "Caller",
    "Message",
    "TypedPayload", "CallContext",
    "ModuleRequest", "ModuleStatus", "SideEffect", "ModuleResponse",
    "ControlRequest", "ControlStatus", "ControlResponse",
    "ApprovalRequest", "ApprovalResponse", "ClarifyRequest", "ClarifyResponse",
    "Pending",
    "GateSubject", "GateDecision",
    "new_session_id", "new_request_id", "new_turn_id",
    "new_call_id", "new_resume_token",
    "canonical_json_bytes", "sha256_hex", "payload_hash", "utcnow",
]
