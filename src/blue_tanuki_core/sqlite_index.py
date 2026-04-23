"""SQLite index for audit events.

Derived view only; JSONL is the truth. Losing the index means a rebuild,
not data loss. All writes are best-effort — an index write failure must
not abort the JSONL append.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .async_sqlite import connect


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    hash TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    request_id TEXT,
    turn_id TEXT,
    call_id TEXT,
    parent_call_id TEXT,
    file_path TEXT NOT NULL,
    file_offset INTEGER NOT NULL,
    file_length INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_session_seq ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_turn ON events(session_id, turn_id);
CREATE INDEX IF NOT EXISTS ix_events_call ON events(call_id);
CREATE INDEX IF NOT EXISTS ix_events_request ON events(request_id);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    final_status TEXT,
    request_id_first TEXT NOT NULL
);
"""


@dataclass
class IndexRow:
    hash: str
    session_id: str
    seq: int
    ts: str
    kind: str
    request_id: str | None
    turn_id: str | None
    call_id: str | None
    parent_call_id: str | None
    file_path: str
    file_offset: int
    file_length: int


class SQLiteIndex:
    def __init__(self, path: Path):
        self.path = path
        self._conn = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await connect(self.path)
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def insert_event(self, row: IndexRow) -> None:
        assert self._conn is not None, "index not opened"
        async with self._lock:
            try:
                await self._conn.execute(
                    """
                    INSERT OR IGNORE INTO events
                      (hash, session_id, seq, ts, kind,
                       request_id, turn_id, call_id, parent_call_id,
                       file_path, file_offset, file_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.hash, row.session_id, row.seq, row.ts, row.kind,
                        row.request_id, row.turn_id, row.call_id, row.parent_call_id,
                        row.file_path, row.file_offset, row.file_length,
                    ),
                )
                await self._conn.commit()
            except Exception:
                # best-effort: do not raise — JSONL is truth
                pass

    async def mark_turn_start(
        self, turn_id: str, session_id: str, started_at: str, request_id_first: str,
    ) -> None:
        assert self._conn is not None
        async with self._lock:
            try:
                await self._conn.execute(
                    """
                    INSERT OR IGNORE INTO turns
                      (turn_id, session_id, started_at, request_id_first)
                    VALUES (?, ?, ?, ?)
                    """,
                    (turn_id, session_id, started_at, request_id_first),
                )
                await self._conn.commit()
            except Exception:
                pass

    async def mark_turn_end(
        self, turn_id: str, ended_at: str, final_status: str,
    ) -> None:
        assert self._conn is not None
        async with self._lock:
            try:
                await self._conn.execute(
                    """
                    UPDATE turns SET ended_at=?, final_status=? WHERE turn_id=?
                    """,
                    (ended_at, final_status, turn_id),
                )
                await self._conn.commit()
            except Exception:
                pass


__all__ = ["SQLiteIndex", "IndexRow"]
