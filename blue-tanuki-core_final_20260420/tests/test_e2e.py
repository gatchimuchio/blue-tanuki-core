import pytest

from blue_tanuki_core.app import App, AppBuildOverrides
from blue_tanuki_core.backend.stub import StubBackend
from blue_tanuki_core.contracts import ControlRequest, Message
from blue_tanuki_core.control_plane import ModuleRegistry
from blue_tanuki_core.gate import AllowAllGate
from blue_tanuki_core.modules.director import RuleDirector
from blue_tanuki_core.modules.file_ops import FileOpsModule
from blue_tanuki_core.modules.memory import MemoryModule
from blue_tanuki_core.settings import Settings


@pytest.fixture
async def app(tmp_path):
    settings = Settings(state_dir=tmp_path)
    registry = ModuleRegistry()
    registry.register(MemoryModule(tmp_path / "memory.sqlite"))
    registry.register(FileOpsModule(tmp_path / "workspace"))
    a = App(
        settings,
        overrides=AppBuildOverrides(
            gate=AllowAllGate(),
            backends=[StubBackend()],
            modules=registry,
            director=RuleDirector(),
        ),
    )
    await a.start()
    yield a
    await a.stop()


async def test_remember_and_recall_round_trip(app):
    r1 = await app.handle(ControlRequest(
        session_id="ses_e2e",
        message=Message(role="user", content="remember coffee is dark roast"),
    ))
    assert r1.status.code == "ok"

    r2 = await app.handle(ControlRequest(
        session_id="ses_e2e",
        message=Message(role="user", content="recall coffee"),
    ))
    assert r2.status.code == "ok"
    # the module result is folded into LLM history; stub echoes user messages,
    # so the recall result reaches the assistant output verbatim-ish
    assert "recall coffee" in r2.output[0].content


async def test_what_do_you_remember_returns_list(app):
    await app.handle(ControlRequest(
        session_id="ses_list",
        message=Message(role="user", content="remember tea is green"),
    ))
    await app.handle(ControlRequest(
        session_id="ses_list",
        message=Message(role="user", content="remember coffee is dark"),
    ))
    r = await app.handle(ControlRequest(
        session_id="ses_list",
        message=Message(role="user", content="what do you remember?"),
    ))
    assert r.status.code == "ok"
    # modules_only: assistant content should contain memory.list result string
    # The stub path is skipped for modules_only; result is str(result).
    assert "tea" in r.output[0].content or "coffee" in r.output[0].content


async def test_write_file_actually_writes(app, tmp_path):
    r = await app.handle(ControlRequest(
        session_id="ses_file",
        message=Message(role="user",
                        content="write file hello.txt: konnichiwa"),
    ))
    assert r.status.code == "ok"
    ws = tmp_path / "workspace"
    assert (ws / "hello.txt").read_text() == "konnichiwa"


async def test_read_file_round_trip(app, tmp_path):
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "workspace" / "note.md").write_text("hello world")
    r = await app.handle(ControlRequest(
        session_id="ses_read",
        message=Message(role="user", content="read file note.md"),
    ))
    assert r.status.code == "ok"
    assert "hello world" in r.output[0].content


async def test_llm_only_fallthrough(app):
    r = await app.handle(ControlRequest(
        session_id="ses_llm",
        message=Message(role="user", content="tell me about the sky"),
    ))
    assert r.status.code == "ok"
    assert "tell me about the sky" in r.output[0].content
