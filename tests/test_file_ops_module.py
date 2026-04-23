from pathlib import Path

import pytest

from blue_tanuki_core.contracts import (
    CallContext, ModuleRequest, TypedPayload, new_call_id,
)
from blue_tanuki_core.errors import ModuleRefused
from blue_tanuki_core.modules.file_ops import FileOpsModule


@pytest.fixture
def fops(tmp_path):
    return FileOpsModule(tmp_path / "workspace", max_bytes=4096)


def _req(kind: str, data: dict) -> ModuleRequest:
    return ModuleRequest(
        call_id=new_call_id(),
        request_id="req_t", session_id="ses_t", turn_id="turn_t",
        module="file_ops", caller="director",
        payload=TypedPayload(kind=kind, data=data),
        context=CallContext(op=kind.split(".")[-1], resource=data.get("path", "")),
    )


async def test_write_then_read(fops):
    r = await fops.handle(_req("file_ops.write",
                               {"path": "hello.txt", "content": "hi"}))
    assert r.status.code == "ok"
    r2 = await fops.handle(_req("file_ops.read", {"path": "hello.txt"}))
    assert r2.result["content"] == "hi"


async def test_overwrite_false_refuses(fops):
    await fops.handle(_req("file_ops.write",
                           {"path": "a.txt", "content": "1"}))
    with pytest.raises(ModuleRefused):
        await fops.handle(_req("file_ops.write",
                               {"path": "a.txt", "content": "2", "overwrite": False}))


async def test_path_escape_refused(fops):
    with pytest.raises(ModuleRefused):
        await fops.handle(_req("file_ops.write",
                               {"path": "../evil", "content": "x"}))


async def test_absolute_path_outside_refused(fops):
    with pytest.raises(ModuleRefused):
        await fops.handle(_req("file_ops.write",
                               {"path": "/etc/passwd", "content": "x"}))


async def test_list(fops):
    await fops.handle(_req("file_ops.write", {"path": "a.md", "content": "a"}))
    await fops.handle(_req("file_ops.write", {"path": "b.md", "content": "b"}))
    await fops.handle(_req("file_ops.write", {"path": "c.txt", "content": "c"}))
    r = await fops.handle(_req("file_ops.list", {"path": ".", "glob": "*.md"}))
    names = sorted(e["path"] for e in r.result["entries"])
    assert names == ["a.md", "b.md"]


async def test_append(fops):
    await fops.handle(_req("file_ops.write", {"path": "log.txt", "content": "a\n"}))
    await fops.handle(_req("file_ops.append", {"path": "log.txt", "content": "b\n"}))
    r = await fops.handle(_req("file_ops.read", {"path": "log.txt"}))
    assert r.result["content"] == "a\nb\n"


async def test_size_cap(fops):
    with pytest.raises(ModuleRefused):
        await fops.handle(_req("file_ops.write",
                               {"path": "big.txt", "content": "x" * 5000}))


async def test_delete(fops):
    await fops.handle(_req("file_ops.write", {"path": "del.txt", "content": "x"}))
    r = await fops.handle(_req("file_ops.delete", {"path": "del.txt"}))
    assert r.result["deleted"] is True
    r2 = await fops.handle(_req("file_ops.exists", {"path": "del.txt"}))
    assert r2.result["exists"] is False


async def test_symlink_escape_refused(fops, tmp_path):
    # create a symlink inside workspace pointing outside
    outside = tmp_path / "outside_target"
    outside.write_text("secret")
    link = fops.root / "link"
    link.symlink_to(outside)
    with pytest.raises(ModuleRefused):
        await fops.handle(_req("file_ops.read", {"path": "link"}))


async def test_side_effect_bytes_tracked(fops):
    r = await fops.handle(_req("file_ops.write",
                               {"path": "x.txt", "content": "hello"}))
    assert any(se.bytes_out == 5 and se.kind == "file.write"
               for se in r.side_effects)
