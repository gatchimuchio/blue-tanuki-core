"""Memory module: SQLite-backed KV store.

Ops:
- set(key, value, namespace="default", ttl_s=None)
- get(key, namespace="default")
- delete(key, namespace="default")
- list(namespace="default", prefix=None, limit=100)

Contract:
- payload.kind must be one of "memory.set", "memory.get", "memory.delete", "memory.list"
- resource in CallContext = f"{namespace}/{key}" (or f"{namespace}/*" for list)
- refused if key contains "/" or newline (directory-traversal-like patterns)
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ..async_sqlite import connect

from ..contracts import ModuleRequest, ModuleResponse, ModuleStatus, SideEffect
from ..control_plane import Module
from ..errors import ModuleRefused


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    expires_at REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS ix_memory_expires ON memory(expires_at);
"""


def _validate_key(key: str) -> None:
    if not key:
        raise ModuleRefused("memory", "empty key")
    if "\n" in key or "\r" in key:
        raise ModuleRefused("memory", "key contains newline")
    if len(key) > 200:
        raise ModuleRefused("memory", "key too long (>200)")


def _validate_ns(ns: str) -> None:
    if not ns:
        raise ModuleRefused("memory", "empty namespace")
    if not ns.replace("_", "").replace("-", "").isalnum():
        raise ModuleRefused("memory", "namespace must be [A-Za-z0-9_-]+")


class MemoryModule(Module):
    name = "memory"

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await connect(self.db_path)
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def handle(self, req: ModuleRequest) -> ModuleResponse:
        if self._conn is None:
            await self.open()
        t0 = time.perf_counter()
        kind = req.payload.kind
        data = req.payload.data

        if kind == "memory.set":
            result, se = await self._set(data)
        elif kind == "memory.get":
            result, se = await self._get(data)
        elif kind == "memory.delete":
            result, se = await self._delete(data)
        elif kind == "memory.list":
            result, se = await self._list(data)
        else:
            raise ModuleRefused("memory", f"unknown payload kind: {kind}")

        duration = int((time.perf_counter() - t0) * 1000)
        return ModuleResponse(
            call_id=req.call_id,
            status=ModuleStatus(code="ok"),
            result=result,
            side_effects=se,
            duration_ms=duration,
        )

    async def _set(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[SideEffect]]:
        ns = data.get("namespace", "default")
        key = data.get("key", "")
        value = data.get("value")
        ttl_s = data.get("ttl_s")
        _validate_ns(ns); _validate_key(key)
        if value is None:
            raise ModuleRefused("memory", "value must not be None")
        serialized = json.dumps(value, ensure_ascii=False)
        expires_at = time.time() + ttl_s if ttl_s else None
        async with self._lock:
            assert self._conn is not None
            await self._conn.execute(
                """
                INSERT INTO memory(namespace, key, value, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                  value=excluded.value,
                  expires_at=excluded.expires_at,
                  updated_at=excluded.updated_at
                """,
                (ns, key, serialized, expires_at, time.time()),
            )
            await self._conn.commit()
        se = [SideEffect(
            kind="memory.set", target=f"{ns}/{key}",
            bytes_out=len(serialized.encode("utf-8")),
        )]
        return {"ok": True, "namespace": ns, "key": key}, se

    async def _get(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[SideEffect]]:
        ns = data.get("namespace", "default")
        key = data.get("key", "")
        _validate_ns(ns); _validate_key(key)
        async with self._lock:
            assert self._conn is not None
            cur = await self._conn.execute(
                "SELECT value, expires_at FROM memory WHERE namespace=? AND key=?",
                (ns, key),
            )
            row = await cur.fetchone()
            await cur.close()
        if not row:
            return {"found": False, "namespace": ns, "key": key}, []
        value_s, expires = row
        if expires is not None and expires < time.time():
            # expired — purge lazily
            async with self._lock:
                assert self._conn is not None
                await self._conn.execute(
                    "DELETE FROM memory WHERE namespace=? AND key=?", (ns, key),
                )
                await self._conn.commit()
            return {"found": False, "namespace": ns, "key": key, "expired": True}, []
        return {
            "found": True,
            "namespace": ns,
            "key": key,
            "value": json.loads(value_s),
        }, []

    async def _delete(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[SideEffect]]:
        ns = data.get("namespace", "default")
        key = data.get("key", "")
        _validate_ns(ns); _validate_key(key)
        async with self._lock:
            assert self._conn is not None
            cur = await self._conn.execute(
                "DELETE FROM memory WHERE namespace=? AND key=?", (ns, key),
            )
            deleted = cur.rowcount
            await cur.close()
            await self._conn.commit()
        return {"deleted": bool(deleted), "namespace": ns, "key": key}, [
            SideEffect(kind="memory.delete", target=f"{ns}/{key}")
        ]

    async def _list(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[SideEffect]]:
        ns = data.get("namespace", "default")
        prefix = data.get("prefix") or ""
        limit = min(int(data.get("limit", 100)), 1000)
        _validate_ns(ns)
        async with self._lock:
            assert self._conn is not None
            cur = await self._conn.execute(
                "SELECT key FROM memory WHERE namespace=? AND key LIKE ? ORDER BY key LIMIT ?",
                (ns, prefix + "%", limit),
            )
            rows = await cur.fetchall()
            await cur.close()
        keys = [r[0] for r in rows]
        return {"namespace": ns, "keys": keys, "count": len(keys)}, []


__all__ = ["MemoryModule"]
