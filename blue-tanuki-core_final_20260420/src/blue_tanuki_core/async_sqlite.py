"""Small async DB shim.

Prefers aiosqlite when available. Falls back to sqlite3 + asyncio.to_thread so the
core remains runnable even when optional wheels are absent.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised implicitly when installed
    import aiosqlite as _aiosqlite  # type: ignore
except Exception:  # pragma: no cover - fallback path is what CI here uses
    _aiosqlite = None


class AsyncCursor:
    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    async def fetchone(self):
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self):
        return await asyncio.to_thread(self._cursor.fetchall)

    async def close(self) -> None:
        await asyncio.to_thread(self._cursor.close)


class AsyncConnection:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = asyncio.Lock()

    async def executescript(self, script: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.executescript, script)

    async def execute(self, sql: str, params: tuple[Any, ...] = ()):
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, sql, params)
        return AsyncCursor(cur)

    async def commit(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.commit)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)


async def connect(path: str | Path):
    if _aiosqlite is not None:
        return await _aiosqlite.connect(str(path))
    conn = await asyncio.to_thread(
        sqlite3.connect,
        str(path),
        check_same_thread=False,
    )
    return AsyncConnection(conn)


__all__ = ["connect", "AsyncConnection", "AsyncCursor"]
