"""Gate interface and policy-driven evaluator.

Evaluation happens at 5 hooks: inbound, pre_module, post_module, pre_llm, post_llm.
Composition rule: severity order stop > suspend_approval > suspend_clarify > pass.
All matched rule ids are recorded in the decision for audit.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field, ValidationError

from .contracts import Direction, GateDecision, GateSubject
from .errors import ConfigError


# ──────────────────────────────────────────────────────────────────────
# Gate protocol
# ──────────────────────────────────────────────────────────────────────

class Gate(Protocol):
    def evaluate(self, subject: GateSubject) -> GateDecision: ...


# ──────────────────────────────────────────────────────────────────────
# Reference implementations
# ──────────────────────────────────────────────────────────────────────

class AllowAllGate:
    """Always returns pass. For tests only."""

    def evaluate(self, subject: GateSubject) -> GateDecision:
        return GateDecision(action="pass", policy_id="allow_all")


class DenyAllGate:
    """Always returns stop. For tests only."""

    def evaluate(self, subject: GateSubject) -> GateDecision:
        return GateDecision(action="stop", reason="deny_all", policy_id="deny_all")


# ──────────────────────────────────────────────────────────────────────
# Policy DSL
# ──────────────────────────────────────────────────────────────────────

_ALLOWED_WHEN_KEYS = {
    "direction", "caller", "module", "op",
    "resource_match", "content_match",
    "side_effect_bytes_out_gt", "payload_size_gt",
    "model_match", "finish_reason",
}

_ALLOWED_ACTIONS = {"pass", "stop", "suspend_approval", "suspend_clarify"}

_SEVERITY = {"pass": 0, "suspend_clarify": 1, "suspend_approval": 2, "stop": 3}


class RawRule(BaseModel):
    id: str
    when: dict[str, Any] = Field(default_factory=dict)
    action: str
    reason: str | None = None
    clarify_prompt: str | None = None
    approval_subject: dict[str, Any] | None = None


class RawPolicy(BaseModel):
    version: int = 1
    defaults: dict[str, Any] = Field(default_factory=dict)
    rules: list[RawRule] = Field(default_factory=list)


class CompiledRule:
    __slots__ = (
        "id", "action", "reason", "clarify_prompt", "approval_subject",
        "direction", "caller", "module", "op",
        "resource_re", "content_re", "model_re",
        "side_effect_bytes_out_gt", "payload_size_gt", "finish_reason",
    )

    def __init__(self, raw: RawRule):
        if raw.action not in _ALLOWED_ACTIONS:
            raise ConfigError(f"rule {raw.id}", f"unknown action {raw.action}")
        unknown = set(raw.when.keys()) - _ALLOWED_WHEN_KEYS
        if unknown:
            raise ConfigError(
                f"rule {raw.id}",
                f"unknown when keys: {sorted(unknown)}",
            )
        self.id = raw.id
        self.action = raw.action
        self.reason = raw.reason
        self.clarify_prompt = raw.clarify_prompt
        self.approval_subject = raw.approval_subject
        w = raw.when
        self.direction = w.get("direction")
        self.caller = w.get("caller")
        self.module = w.get("module")
        self.op = w.get("op")
        self.resource_re = re.compile(w["resource_match"]) if "resource_match" in w else None
        self.content_re = re.compile(w["content_match"]) if "content_match" in w else None
        self.model_re = re.compile(w["model_match"]) if "model_match" in w else None
        self.side_effect_bytes_out_gt = w.get("side_effect_bytes_out_gt")
        self.payload_size_gt = w.get("payload_size_gt")
        self.finish_reason = w.get("finish_reason")

    def matches(self, s: GateSubject) -> bool:
        if self.direction is not None and s.direction != self.direction:
            return False
        if self.caller is not None and s.caller != self.caller:
            return False
        if self.module is not None and s.module != self.module:
            return False
        if self.op is not None and s.op != self.op:
            return False
        if self.resource_re is not None:
            if not s.resource or not self.resource_re.search(s.resource):
                return False
        if self.content_re is not None:
            targets: list[str] = []
            if s.content:
                targets.append(s.content)
            targets.extend(s.messages_preview)
            if not any(self.content_re.search(t) for t in targets):
                return False
        if self.model_re is not None:
            if not s.model or not self.model_re.search(s.model):
                return False
        if self.side_effect_bytes_out_gt is not None:
            if s.side_effect_bytes_out_total <= self.side_effect_bytes_out_gt:
                return False
        if self.payload_size_gt is not None:
            if s.payload_size <= self.payload_size_gt:
                return False
        if self.finish_reason is not None:
            if s.finish_reason != self.finish_reason:
                return False
        return True


class PolicyGate:
    def __init__(
        self,
        rules: list[CompiledRule],
        *,
        on_no_match: str = "pass",
        policy_id: str = "policy",
    ):
        if on_no_match not in _ALLOWED_ACTIONS:
            raise ConfigError("defaults.on_no_match", f"unknown action {on_no_match}")
        self.rules = rules
        self.on_no_match = on_no_match
        self.policy_id = policy_id

    def evaluate(self, subject: GateSubject) -> GateDecision:
        matched: list[CompiledRule] = [r for r in self.rules if r.matches(subject)]
        if not matched:
            return GateDecision(
                action=self.on_no_match,  # type: ignore[arg-type]
                policy_id=self.policy_id,
                matched_rule_ids=[],
            )
        # severity pick; ties broken by rule order (first wins)
        picked = max(matched, key=lambda r: _SEVERITY[r.action])
        # but ties: we want first-in-rules-order among max severity
        max_sev = _SEVERITY[picked.action]
        picked = next(r for r in matched if _SEVERITY[r.action] == max_sev)
        return GateDecision(
            action=picked.action,  # type: ignore[arg-type]
            reason=picked.reason,
            policy_id=self.policy_id,
            matched_rule_ids=[r.id for r in matched],
            clarify_prompt=picked.clarify_prompt,
            approval_subject=picked.approval_subject,
        )


def load_policy(path: Path) -> PolicyGate:
    if not path.exists():
        raise ConfigError("policy_path", f"{path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError("policy_path", f"yaml parse failed: {e}") from e
    try:
        parsed = RawPolicy.model_validate(raw or {})
    except ValidationError as e:
        raise ConfigError("policy_path", f"schema validation failed: {e}") from e
    compiled = [CompiledRule(r) for r in parsed.rules]
    on_no_match = parsed.defaults.get("on_no_match", "pass")
    return PolicyGate(compiled, on_no_match=on_no_match)


__all__ = [
    "Gate", "AllowAllGate", "DenyAllGate",
    "PolicyGate", "CompiledRule", "RawRule", "RawPolicy",
    "load_policy",
]
