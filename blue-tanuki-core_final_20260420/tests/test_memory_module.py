import pytest

from blue_tanuki_core.contracts import (
    CallContext, ModuleRequest, TypedPayload, new_call_id,
)
from blue_tanuki_core.errors import ModuleRefused
from blue_tanuki_core.modules.memory import MemoryModule


@pytest.fixture
async def mem(tmp_path):
    m = MemoryModule(tmp_path / "memory.sqlite")
    await m.open()
    yield m
    await m.close()


def _req(kind: str, data: dict, op: str = "", resource: str = "") -> ModuleRequest:
    return ModuleRequest(
        call_id=new_call_id(),
        request_id="req_t", session_id="ses_t", turn_id="turn_t",
        module="memory", caller="director",
        payload=TypedPayload(kind=kind, data=data),
        context=CallContext(op=op, resource=resource),
    )


async def test_set_and_get(mem):
    r = await mem.handle(_req("memory.set",
                              {"namespace": "notes", "key": "k1", "value": "hello"}))
    assert r.status.code == "ok"
    r2 = await mem.handle(_req("memory.get", {"namespace": "notes", "key": "k1"}))
    assert r2.result["found"] is True
    assert r2.result["value"] == "hello"


async def test_get_missing(mem):
    r = await mem.handle(_req("memory.get", {"namespace": "notes", "key": "nope"}))
    assert r.result["found"] is False


async def test_delete(mem):
    await mem.handle(_req("memory.set", {"namespace": "notes", "key": "x", "value": 1}))
    r = await mem.handle(_req("memory.delete", {"namespace": "notes", "key": "x"}))
    assert r.result["deleted"] is True
    r2 = await mem.handle(_req("memory.get", {"namespace": "notes", "key": "x"}))
    assert r2.result["found"] is False


async def test_list_prefix(mem):
    for k in ["note_a", "note_b", "note_c", "other"]:
        await mem.handle(_req("memory.set",
                              {"namespace": "notes", "key": k, "value": 1}))
    r = await mem.handle(_req("memory.list",
                              {"namespace": "notes", "prefix": "note_"}))
    assert r.result["count"] == 3
    assert set(r.result["keys"]) == {"note_a", "note_b", "note_c"}


async def test_refuses_bad_key(mem):
    with pytest.raises(ModuleRefused):
        await mem.handle(_req("memory.set",
                              {"namespace": "notes", "key": "", "value": "v"}))
    with pytest.raises(ModuleRefused):
        await mem.handle(_req("memory.set",
                              {"namespace": "notes", "key": "bad\nkey", "value": "v"}))


async def test_refuses_bad_namespace(mem):
    with pytest.raises(ModuleRefused):
        await mem.handle(_req("memory.set",
                              {"namespace": "bad/ns", "key": "k", "value": "v"}))


async def test_side_effects_recorded(mem):
    r = await mem.handle(_req("memory.set",
                              {"namespace": "notes", "key": "k", "value": "hi"}))
    assert len(r.side_effects) == 1
    assert r.side_effects[0].kind == "memory.set"
    assert r.side_effects[0].bytes_out > 0


async def test_unknown_kind(mem):
    with pytest.raises(ModuleRefused):
        await mem.handle(_req("memory.wat", {}))
