"""PendingStore implementations.

Default now includes a file-backed store so approval/clarify suspension survives
process restarts. In-memory remains available for tests and ephemeral runs.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import CallContext, Message, ModuleRequest, Pending, TypedPayload
from .errors import PendingConflict, PendingExpired, PendingNotFound


@dataclass
class RunState:
    """Snapshot of an in-flight turn, restorable on resume."""
    request_id: str
    session_id: str
    turn_id: str
    phase: str
    history: list[Message] = field(default_factory=list)
    pending_module: ModuleRequest | None = None
    pending_llm: Any | None = None
    audit_mode: bool = False
    approval_granted: bool = False
    approval_bypass_directions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class PendingStore(Protocol):
    async def put(self, token: str, pending: Pending, state: RunState, ttl_s: int) -> None: ...
    async def take(self, token: str) -> tuple[Pending, RunState]: ...
    async def cancel(self, token: str) -> None: ...
    async def sweep_expired(self) -> list[str]: ...


@dataclass
class _Entry:
    pending: Pending
    state: RunState
    expires_at: float
    consumed: bool = False


class InMemoryPendingStore:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def put(self, token: str, pending: Pending, state: RunState, ttl_s: int) -> None:
        loop = asyncio.get_running_loop()
        expires_at = loop.time() + ttl_s
        async with self._lock:
            if token in self._entries:
                raise PendingConflict(token)
            self._entries[token] = _Entry(pending=pending, state=state, expires_at=expires_at)

    async def take(self, token: str) -> tuple[Pending, RunState]:
        loop = asyncio.get_running_loop()
        async with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                raise PendingNotFound(token)
            if entry.consumed:
                raise PendingConflict(token)
            if loop.time() >= entry.expires_at:
                del self._entries[token]
                raise PendingExpired(token)
            entry.consumed = True
            del self._entries[token]
            return entry.pending, entry.state

    async def cancel(self, token: str) -> None:
        async with self._lock:
            self._entries.pop(token, None)

    async def sweep_expired(self) -> list[str]:
        loop = asyncio.get_running_loop()
        now = loop.time()
        expired: list[str] = []
        async with self._lock:
            for tok, entry in list(self._entries.items()):
                if now >= entry.expires_at:
                    expired.append(tok)
                    del self._entries[tok]
        return expired


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value




def _hydrate_run_state(data: dict[str, Any]) -> RunState:
    history = [m if isinstance(m, Message) else Message.model_validate(m) for m in data.get("history", [])]
    pending_module_raw = data.get("pending_module")
    pending_module = None
    if isinstance(pending_module_raw, dict):
        pending_module = ModuleRequest.model_validate(pending_module_raw)
    state = RunState(
        request_id=data["request_id"],
        session_id=data["session_id"],
        turn_id=data["turn_id"],
        phase=data["phase"],
        history=history,
        pending_module=pending_module,
        pending_llm=None,
        audit_mode=bool(data.get("audit_mode", False)),
        approval_granted=bool(data.get("approval_granted", False)),
        approval_bypass_directions=list(data.get("approval_bypass_directions", [])),
        extra=dict(data.get("extra", {})),
    )
    return state


class FilePendingStore:
    """JSON-backed one-shot pending store.

    Uses wall-clock expiration so entries remain valid across restart.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, token: str) -> Path:
        return self.root / f"{token}.json"

    async def put(self, token: str, pending: Pending, state: RunState, ttl_s: int) -> None:
        path = self._path(token)
        payload = {
            "token": token,
            "expires_at": time.time() + ttl_s,
            "pending": pending.model_dump(),
            "state": _to_jsonable(state),
        }
        async with self._lock:
            if path.exists():
                raise PendingConflict(token)
            tmp = path.with_suffix(".tmp")
            await asyncio.to_thread(tmp.write_text, json.dumps(payload, ensure_ascii=False, sort_keys=True))
            await asyncio.to_thread(tmp.replace, path)

    async def take(self, token: str) -> tuple[Pending, RunState]:
        path = self._path(token)
        async with self._lock:
            if not path.exists():
                raise PendingNotFound(token)
            raw = await asyncio.to_thread(path.read_text)
            obj = json.loads(raw)
            if time.time() >= float(obj.get("expires_at", 0)):
                await asyncio.to_thread(path.unlink)
                raise PendingExpired(token)
            await asyncio.to_thread(path.unlink)
        pending = Pending.model_validate(obj["pending"])
        state = _hydrate_run_state(obj["state"])
        return pending, state

    async def cancel(self, token: str) -> None:
        path = self._path(token)
        async with self._lock:
            if path.exists():
                await asyncio.to_thread(path.unlink)

    async def sweep_expired(self) -> list[str]:
        expired: list[str] = []
        async with self._lock:
            for path in self.root.glob("resume_*.json"):
                try:
                    raw = await asyncio.to_thread(path.read_text)
                    obj = json.loads(raw)
                    if time.time() >= float(obj.get("expires_at", 0)):
                        expired.append(obj.get("token") or path.stem)
                        await asyncio.to_thread(path.unlink)
                except Exception:
                    continue
        return expired


__all__ = ["PendingStore", "InMemoryPendingStore", "FilePendingStore", "RunState"]
