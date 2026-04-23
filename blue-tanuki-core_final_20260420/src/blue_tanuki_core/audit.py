"""AuditLog: JSONL append-only with per-session hash chain.

Invariants (see DESIGN.md §9):
- One chain per session
- Every event is redacted before hashing and writing
- First event of a session: prev_hash = "0" * 64
- hash = sha256(canonical_json(event without 'hash' field))
- head.hash / head.seq are updated atomically after fsync
- SQLite index insert is best-effort
"""
from __future__ import annotations

import asyncio
import json
import os
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .contracts import ControlRequest, canonical_json_bytes, sha256_hex, utcnow
from .errors import AuditIntegrityError
from .redactor import Redactor
from .sqlite_index import IndexRow, SQLiteIndex


AuditKind = Literal[
    "request_received",
    "gate_decision",
    "module_request",
    "module_response",
    "llm_request",
    "llm_response",
    "state_transition",
    "suspend",
    "resume",
    "resume_rejected",
    "suspend_expired",
    "exception",
    "typed_error",
    "response_sent",
    "rotation_marker",
    "session_opened",
    "integrity_mismatch",
]

ZERO_HASH = "0" * 64


@dataclass
class _SessionState:
    head_hash: str
    head_seq: int
    current_file: Path
    current_file_size: int
    lock: asyncio.Lock


class AuditLog:
    def __init__(
        self,
        root: Path,
        redactor: Redactor,
        index: SQLiteIndex | None = None,
        *,
        rotate_bytes: int = 100 * 1024 * 1024,
    ):
        self.root = root
        self.redactor = redactor
        self.index = index
        self.rotate_bytes = rotate_bytes
        self._sessions: dict[str, _SessionState] = {}
        self._global_lock = asyncio.Lock()

    # ── session lifecycle ────────────────────────────────────────────

    def _session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id

    def _current_file_path(self, session_id: str, idx: int) -> Path:
        return self._session_dir(session_id) / f"events.{idx:04d}.jsonl"

    async def open_session(self, session_id: str) -> None:
        async with self._global_lock:
            if session_id in self._sessions:
                return
            sdir = self._session_dir(session_id)
            sdir.mkdir(parents=True, exist_ok=True)
            head_hash_file = sdir / "head.hash"
            head_seq_file = sdir / "head.seq"
            if head_hash_file.exists() and head_seq_file.exists():
                head_hash = head_hash_file.read_text().strip() or ZERO_HASH
                head_seq = int(head_seq_file.read_text().strip() or "0")
            else:
                head_hash = ZERO_HASH
                head_seq = 0
            # locate current (highest numbered) file
            files = sorted(sdir.glob("events.*.jsonl"))
            if files:
                current_file = files[-1]
                current_size = current_file.stat().st_size
            else:
                current_file = self._current_file_path(session_id, 1)
                current_file.touch()
                current_size = 0
            self._sessions[session_id] = _SessionState(
                head_hash=head_hash,
                head_seq=head_seq,
                current_file=current_file,
                current_file_size=current_size,
                lock=asyncio.Lock(),
            )
        # write a session_opened event (outside global lock, per-session serialized)
        if head_seq == 0:
            await self.record(
                "session_opened",
                session_id=session_id,
                payload={"ts": utcnow().isoformat()},
            )

    # ── recording ────────────────────────────────────────────────────

    async def record(
        self,
        kind: AuditKind,
        *,
        session_id: str,
        payload: dict[str, Any],
        request_id: str | None = None,
        turn_id: str | None = None,
        call_id: str | None = None,
        parent_call_id: str | None = None,
    ) -> str:
        if session_id not in self._sessions:
            await self.open_session(session_id)
        sess = self._sessions[session_id]
        async with sess.lock:
            return await self._append_locked(
                sess, kind, session_id, payload,
                request_id, turn_id, call_id, parent_call_id,
            )

    async def _append_locked(
        self,
        sess: _SessionState,
        kind: str,
        session_id: str,
        payload: dict[str, Any],
        request_id: str | None,
        turn_id: str | None,
        call_id: str | None,
        parent_call_id: str | None,
    ) -> str:
        redacted = self.redactor.redact(payload)
        seq = sess.head_seq + 1
        ts = utcnow().isoformat()
        body = {
            "seq": seq,
            "ts": ts,
            "session_id": session_id,
            "request_id": request_id,
            "turn_id": turn_id,
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "kind": kind,
            "payload": redacted,
            "prev_hash": sess.head_hash,
        }
        h = sha256_hex(canonical_json_bytes(body))
        body["hash"] = h
        line = (json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

        # rotation check
        if sess.current_file_size + len(line) > self.rotate_bytes and sess.current_file_size > 0:
            await self._rotate_locked(sess, session_id)

        offset = sess.current_file_size
        # append (O_APPEND via open mode "ab")
        with open(sess.current_file, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        sess.head_hash = h
        sess.head_seq = seq
        sess.current_file_size += len(line)

        # atomic update of head.hash / head.seq
        self._atomic_write(self._session_dir(session_id) / "head.hash", h)
        self._atomic_write(self._session_dir(session_id) / "head.seq", str(seq))

        # best-effort index insert
        if self.index is not None:
            try:
                await self.index.insert_event(IndexRow(
                    hash=h,
                    session_id=session_id,
                    seq=seq,
                    ts=ts,
                    kind=kind,
                    request_id=request_id,
                    turn_id=turn_id,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                    file_path=str(sess.current_file),
                    file_offset=offset,
                    file_length=len(line),
                ))
            except Exception:
                # swallow; truth is in JSONL
                pass
        return h

    async def _rotate_locked(self, sess: _SessionState, session_id: str) -> None:
        # write a rotation_marker in the current file first
        seq = sess.head_seq + 1
        ts = utcnow().isoformat()
        body = {
            "seq": seq,
            "ts": ts,
            "session_id": session_id,
            "request_id": None,
            "turn_id": None,
            "call_id": None,
            "parent_call_id": None,
            "kind": "rotation_marker",
            "payload": {"closed_file": str(sess.current_file)},
            "prev_hash": sess.head_hash,
        }
        h = sha256_hex(canonical_json_bytes(body))
        body["hash"] = h
        line = (json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        with open(sess.current_file, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        sess.head_hash = h
        sess.head_seq = seq

        # new file
        match = sess.current_file.stem.rsplit(".", 1)
        idx = int(match[1]) + 1 if len(match) == 2 and match[1].isdigit() else 2
        sess.current_file = self._current_file_path(session_id, idx)
        sess.current_file.touch()
        sess.current_file_size = 0

        self._atomic_write(self._session_dir(session_id) / "head.hash", h)
        self._atomic_write(self._session_dir(session_id) / "head.seq", str(seq))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, path)

    # ── exception helper ─────────────────────────────────────────────

    async def record_exception(
        self,
        req: ControlRequest | None,
        exc: BaseException,
        *,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or (req.session_id if req else "unknown")
        payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": _tb.format_exc(),
        }
        return await self.record(
            "exception",
            session_id=sid,
            payload=payload,
            request_id=req.request_id if req else None,
            turn_id=req.turn_id if req else None,
        )

    # ── integrity ────────────────────────────────────────────────────

    async def verify_session(self, session_id: str) -> None:
        """Replay all JSONL files in order, recomputing the hash chain.
        Raise AuditIntegrityError on mismatch.
        """
        sdir = self._session_dir(session_id)
        if not sdir.exists():
            raise AuditIntegrityError(f"no session dir: {session_id}")
        files = sorted(sdir.glob("events.*.jsonl"))
        prev = ZERO_HASH
        last_seq = 0
        for f in files:
            for raw in f.read_text().splitlines():
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                if rec.get("prev_hash") != prev:
                    raise AuditIntegrityError(
                        f"prev_hash mismatch at seq {rec.get('seq')} in {f}"
                    )
                claimed = rec.pop("hash")
                recomputed = sha256_hex(canonical_json_bytes(rec))
                if claimed != recomputed:
                    raise AuditIntegrityError(
                        f"hash mismatch at seq {rec.get('seq')} in {f}"
                    )
                prev = claimed
                last_seq = rec["seq"]

    async def close(self) -> None:
        # head files are already up to date after every append
        return


__all__ = ["AuditLog", "AuditKind", "ZERO_HASH"]
