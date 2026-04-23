"""Replay tool: re-execute llm_request events from audit log and diff.

Purpose:
- Third-party verifiability of "this input yielded this output at this time".
- Audit mode requires temperature=0 + seed, so stub backend reproduces exactly.
  Real providers are best-effort: we still record diffs and a match-rate.

Usage:
    from blue_tanuki_core.replay import replay_session
    report = await replay_session(
        audit_root=Path("~/.blue-tanuki/audit"),
        session_id="ses_...",
        backend=StubBackend(),  # or AnthropicBackend(...)
    )
    print(report.summary())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend.base import BaseBackend, LLMCallRequest, LLMCallResponse
from .contracts import Message


@dataclass
class ReplayDiff:
    call_id: str | None
    turn_id: str | None
    matched: bool
    original_content: str
    replayed_content: str
    notes: list[str] = field(default_factory=list)


@dataclass
class ReplayReport:
    session_id: str
    total: int = 0
    matched: int = 0
    skipped: int = 0
    errors: int = 0
    diffs: list[ReplayDiff] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.matched / self.total

    def summary(self) -> str:
        lines = [
            f"session={self.session_id}",
            f"llm_calls total={self.total} matched={self.matched} "
            f"errors={self.errors} skipped={self.skipped}",
            f"match_rate={self.match_rate:.1%}",
        ]
        for d in self.diffs:
            if not d.matched:
                lines.append(
                    f"  DIFF call={d.call_id} turn={d.turn_id}\n"
                    f"    orig:    {d.original_content[:100]!r}\n"
                    f"    replay:  {d.replayed_content[:100]!r}"
                )
        return "\n".join(lines)


def _iter_session_events(audit_root: Path, session_id: str):
    """Yield audit events in order, across rotated JSONL files."""
    sdir = audit_root / "sessions" / session_id
    if not sdir.exists():
        return
    for f in sorted(sdir.glob("events.*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _reconstruct_request(payload: dict[str, Any]) -> LLMCallRequest:
    msgs = [Message(**m) for m in payload.get("messages", [])]
    return LLMCallRequest(
        model=payload.get("model", "unknown"),
        messages=msgs,
        max_tokens=payload.get("max_tokens", 1024),
        temperature=payload.get("temperature"),
        seed=payload.get("seed"),
    )


async def replay_session(
    *,
    audit_root: Path,
    session_id: str,
    backend: BaseBackend,
    require_audit_mode: bool = True,
    timeout_s: float = 60,
) -> ReplayReport:
    """Walk the audit JSONL; for each llm_request, call backend and compare
    to the adjacent llm_response record (by call_id)."""
    report = ReplayReport(session_id=session_id)

    # index llm_response by call_id for lookup
    responses: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    for ev in _iter_session_events(audit_root, session_id):
        kind = ev.get("kind")
        if kind == "llm_request":
            requests.append(ev)
        elif kind == "llm_response":
            cid = ev.get("call_id")
            if cid:
                responses[cid] = ev

    report.total = len(requests)

    for req_ev in requests:
        call_id = req_ev.get("call_id")
        turn_id = req_ev.get("turn_id")
        payload = req_ev.get("payload") or {}
        orig_resp = responses.get(call_id or "")
        if orig_resp is None:
            report.skipped += 1
            report.diffs.append(ReplayDiff(
                call_id=call_id, turn_id=turn_id, matched=False,
                original_content="", replayed_content="",
                notes=["no corresponding llm_response in audit"],
            ))
            continue

        if require_audit_mode:
            t = payload.get("temperature")
            seed = payload.get("seed")
            if t != 0 or seed is None:
                report.skipped += 1
                report.diffs.append(ReplayDiff(
                    call_id=call_id, turn_id=turn_id, matched=False,
                    original_content=orig_resp.get("payload", {}).get("content", ""),
                    replayed_content="",
                    notes=[f"not audit-mode (t={t}, seed={seed})"],
                ))
                continue

        try:
            llm_req = _reconstruct_request(payload)
            new_resp: LLMCallResponse = await backend.call(
                llm_req, timeout_s=timeout_s,
            )
        except Exception as e:
            report.errors += 1
            report.diffs.append(ReplayDiff(
                call_id=call_id, turn_id=turn_id, matched=False,
                original_content=orig_resp.get("payload", {}).get("content", ""),
                replayed_content="",
                notes=[f"error: {type(e).__name__}: {e}"],
            ))
            continue

        orig_content = orig_resp.get("payload", {}).get("content", "")
        matched = orig_content == new_resp.content
        if matched:
            report.matched += 1
        report.diffs.append(ReplayDiff(
            call_id=call_id, turn_id=turn_id, matched=matched,
            original_content=orig_content, replayed_content=new_resp.content,
        ))

    return report


__all__ = ["replay_session", "ReplayReport", "ReplayDiff"]
