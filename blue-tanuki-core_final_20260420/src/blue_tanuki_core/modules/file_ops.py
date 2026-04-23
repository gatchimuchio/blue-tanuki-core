"""file_ops: sandboxed file read/write/list under a single workspace root.

Ops:
- read(path) -> text content
- write(path, content, overwrite=True)
- append(path, content)
- list(path=".", glob="*")
- exists(path)
- delete(path)

Contract:
- All paths are relative or absolute; all must resolve INSIDE `workspace_root`
  after realpath (symlink-following).
- Any path escaping the workspace raises ModuleRefused.
- Reads/writes above `max_bytes` raise ModuleRefused.
- Binary files are read as UTF-8 with errors='replace' (warning in result).
- Writes are atomic: temp file + rename.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ..contracts import ModuleRequest, ModuleResponse, ModuleStatus, SideEffect
from ..control_plane import Module
from ..errors import ModuleRefused


DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB per op


class FileOpsModule(Module):
    name = "file_ops"

    def __init__(self, workspace_root: Path, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = workspace_root.resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _resolve(self, p: str) -> Path:
        if not p:
            raise ModuleRefused("file_ops", "empty path")
        # reject explicit path traversal attempts early
        candidate = (self.root / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ModuleRefused("file_ops", f"path escapes workspace: {p}") from e
        return candidate

    async def handle(self, req: ModuleRequest) -> ModuleResponse:
        t0 = time.perf_counter()
        kind = req.payload.kind
        data = req.payload.data
        op_map = {
            "file_ops.read": self._read,
            "file_ops.write": self._write,
            "file_ops.append": self._append,
            "file_ops.list": self._list,
            "file_ops.exists": self._exists,
            "file_ops.delete": self._delete,
        }
        if kind not in op_map:
            raise ModuleRefused("file_ops", f"unknown payload kind: {kind}")
        result, side_effects = await op_map[kind](data)
        return ModuleResponse(
            call_id=req.call_id,
            status=ModuleStatus(code="ok"),
            result=result,
            side_effects=side_effects,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── ops ──────────────────────────────────────────────────────────

    async def _read(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", ""))
        if not p.exists():
            raise ModuleRefused("file_ops", f"not found: {p}")
        if not p.is_file():
            raise ModuleRefused("file_ops", f"not a regular file: {p}")
        size = p.stat().st_size
        if size > self.max_bytes:
            raise ModuleRefused(
                "file_ops", f"file too large: {size} > {self.max_bytes}",
            )
        async with self._lock:
            text = p.read_text(encoding="utf-8", errors="replace")
        return (
            {"path": str(p.relative_to(self.root)), "content": text, "bytes": size},
            [SideEffect(kind="file.read", target=str(p), bytes_in=size)],
        )

    async def _write(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", ""))
        content = data.get("content", "")
        overwrite = bool(data.get("overwrite", True))
        if p.exists() and not overwrite:
            raise ModuleRefused("file_ops", f"exists and overwrite=False: {p}")
        encoded = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        if len(encoded) > self.max_bytes:
            raise ModuleRefused(
                "file_ops", f"write too large: {len(encoded)} > {self.max_bytes}",
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            # atomic write: tmp in same dir + rename
            with tempfile.NamedTemporaryFile(
                "wb", dir=p.parent, delete=False, suffix=".tmp",
            ) as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
                tmp_path = Path(f.name)
            os.replace(tmp_path, p)
        return (
            {"path": str(p.relative_to(self.root)), "bytes": len(encoded)},
            [SideEffect(kind="file.write", target=str(p), bytes_out=len(encoded))],
        )

    async def _append(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", ""))
        content = data.get("content", "")
        encoded = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        current = p.stat().st_size if p.exists() else 0
        if current + len(encoded) > self.max_bytes:
            raise ModuleRefused(
                "file_ops",
                f"append would exceed cap: {current + len(encoded)} > {self.max_bytes}",
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            with open(p, "ab") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
        return (
            {"path": str(p.relative_to(self.root)), "appended_bytes": len(encoded)},
            [SideEffect(kind="file.append", target=str(p), bytes_out=len(encoded))],
        )

    async def _list(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", "."))
        if not p.exists():
            raise ModuleRefused("file_ops", f"not found: {p}")
        if not p.is_dir():
            raise ModuleRefused("file_ops", f"not a directory: {p}")
        pattern = data.get("glob", "*")
        entries: list[dict[str, Any]] = []
        for child in sorted(p.glob(pattern)):
            try:
                rel = str(child.relative_to(self.root))
            except ValueError:
                continue
            entries.append({
                "path": rel,
                "is_dir": child.is_dir(),
                "bytes": child.stat().st_size if child.is_file() else None,
            })
        return (
            {"root": str(p.relative_to(self.root)), "entries": entries,
             "count": len(entries)},
            [],
        )

    async def _exists(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", ""))
        return (
            {"path": str(p.relative_to(self.root)), "exists": p.exists()},
            [],
        )

    async def _delete(self, data: dict[str, Any]):
        p = self._resolve(data.get("path", ""))
        if not p.exists():
            return (
                {"path": str(p.relative_to(self.root)), "deleted": False},
                [],
            )
        if p.is_dir():
            raise ModuleRefused(
                "file_ops", "directory deletion not allowed in this version",
            )
        async with self._lock:
            p.unlink()
        return (
            {"path": str(p.relative_to(self.root)), "deleted": True},
            [SideEffect(kind="file.delete", target=str(p))],
        )


__all__ = ["FileOpsModule"]
