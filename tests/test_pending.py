import asyncio
import pytest

from blue_tanuki_core.contracts import Pending
from blue_tanuki_core.errors import PendingConflict, PendingExpired, PendingNotFound
from blue_tanuki_core.pending import InMemoryPendingStore, RunState


def _state() -> RunState:
    return RunState(
        request_id="r", session_id="s", turn_id="t",
        phase="pre_llm",
    )


async def test_put_and_take_once():
    store = InMemoryPendingStore()
    p = Pending(kind="approval", token="tok1")
    await store.put("tok1", p, _state(), ttl_s=60)
    got, st = await store.take("tok1")
    assert got.token == "tok1"
    with pytest.raises(PendingNotFound):
        await store.take("tok1")


async def test_put_conflict_raises():
    store = InMemoryPendingStore()
    p = Pending(kind="approval", token="tok2")
    await store.put("tok2", p, _state(), ttl_s=60)
    with pytest.raises(PendingConflict):
        await store.put("tok2", p, _state(), ttl_s=60)


async def test_ttl_expires():
    store = InMemoryPendingStore()
    p = Pending(kind="clarify", token="tok3")
    await store.put("tok3", p, _state(), ttl_s=0)
    await asyncio.sleep(0.05)
    with pytest.raises(PendingExpired):
        await store.take("tok3")


async def test_cancel_removes():
    store = InMemoryPendingStore()
    p = Pending(kind="approval", token="tok4")
    await store.put("tok4", p, _state(), ttl_s=60)
    await store.cancel("tok4")
    with pytest.raises(PendingNotFound):
        await store.take("tok4")


async def test_sweep_returns_expired():
    store = InMemoryPendingStore()
    await store.put("a", Pending(kind="approval", token="a"), _state(), ttl_s=0)
    await store.put("b", Pending(kind="approval", token="b"), _state(), ttl_s=60)
    await asyncio.sleep(0.05)
    expired = await store.sweep_expired()
    assert expired == ["a"]
