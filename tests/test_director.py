import pytest

from blue_tanuki_core.contracts import Message
from blue_tanuki_core.control_plane import ModuleRegistry
from blue_tanuki_core.modules.director import RuleDirector
from blue_tanuki_core.modules.memory import MemoryModule
from blue_tanuki_core.modules.file_ops import FileOpsModule
from blue_tanuki_core.pending import RunState


@pytest.fixture
def registry(tmp_path):
    r = ModuleRegistry()
    r.register(MemoryModule(tmp_path / "m.sqlite"))
    r.register(FileOpsModule(tmp_path / "ws"))
    return r


def _state(text: str) -> RunState:
    return RunState(
        request_id="r", session_id="s", turn_id="t",
        phase="plan",
        history=[Message(role="user", content=text)],
    )


async def test_remember_that_routes_to_memory_set(registry):
    d = RuleDirector()
    plan = await d.plan(_state("remember that I like dark roast"), registry)
    assert plan.action == "modules_then_llm"
    assert plan.module_calls[0].module == "memory"
    assert plan.module_calls[0].payload.kind == "memory.set"
    assert plan.module_calls[0].payload.data["value"] == "I like dark roast"


async def test_remember_k_is_v(registry):
    d = RuleDirector()
    plan = await d.plan(_state("remember coffee is dark roast"), registry)
    assert plan.module_calls[0].payload.data["key"] == "coffee"
    assert plan.module_calls[0].payload.data["value"] == "dark roast"


async def test_forget_routes_to_delete(registry):
    d = RuleDirector()
    plan = await d.plan(_state("forget coffee"), registry)
    assert plan.module_calls[0].payload.kind == "memory.delete"


async def test_what_do_you_remember_is_modules_only(registry):
    d = RuleDirector()
    plan = await d.plan(_state("what do you remember?"), registry)
    assert plan.action == "modules_only"
    assert plan.module_calls[0].payload.kind == "memory.list"


async def test_read_file(registry):
    d = RuleDirector()
    plan = await d.plan(_state("read file notes.md"), registry)
    assert plan.module_calls[0].module == "file_ops"
    assert plan.module_calls[0].payload.kind == "file_ops.read"
    assert plan.module_calls[0].payload.data["path"] == "notes.md"


async def test_write_file(registry):
    d = RuleDirector()
    plan = await d.plan(_state("write file todo.txt: buy milk"), registry)
    assert plan.module_calls[0].payload.kind == "file_ops.write"
    assert plan.module_calls[0].payload.data["content"] == "buy milk"


async def test_list_files(registry):
    d = RuleDirector()
    plan = await d.plan(_state("list files"), registry)
    assert plan.module_calls[0].payload.kind == "file_ops.list"


async def test_fallthrough_llm_only(registry):
    d = RuleDirector()
    plan = await d.plan(_state("hello there"), registry)
    assert plan.action == "llm_only"
    assert not plan.module_calls


async def test_skips_module_if_not_registered(tmp_path):
    empty = ModuleRegistry()
    d = RuleDirector()
    plan = await d.plan(_state("remember that x"), empty)
    assert plan.action == "llm_only"
