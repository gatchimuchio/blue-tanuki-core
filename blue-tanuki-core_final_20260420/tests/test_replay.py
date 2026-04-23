import pytest

from blue_tanuki_core.app import App, AppBuildOverrides
from blue_tanuki_core.backend.stub import StubBackend
from blue_tanuki_core.contracts import ControlRequest, Message
from blue_tanuki_core.control_plane import ModuleRegistry
from blue_tanuki_core.gate import AllowAllGate
from blue_tanuki_core.modules.director import RuleDirector
from blue_tanuki_core.modules.memory import MemoryModule
from blue_tanuki_core.replay import replay_session
from blue_tanuki_core.settings import Settings


@pytest.fixture
async def audit_app(tmp_path):
    settings = Settings(state_dir=tmp_path)
    registry = ModuleRegistry()
    registry.register(MemoryModule(tmp_path / "mem.sqlite"))
    app = App(
        settings,
        overrides=AppBuildOverrides(
            gate=AllowAllGate(),
            backends=[StubBackend()],
            modules=registry,
            director=RuleDirector(),
            audit_mode=True,
        ),
    )
    await app.start()
    yield app, settings
    await app.stop()


async def test_replay_matches_in_audit_mode(audit_app):
    app, settings = audit_app
    sid = "ses_replay_ok"
    for msg in ["hello", "tell me a joke", "bye"]:
        r = await app.handle(ControlRequest(
            session_id=sid, message=Message(role="user", content=msg),
        ))
        assert r.status.code == "ok"

    report = await replay_session(
        audit_root=settings.state_dir / "audit",
        session_id=sid,
        backend=StubBackend(),
    )
    assert report.total == 3
    assert report.matched == 3
    assert report.match_rate == 1.0


async def test_replay_skips_non_audit_mode(tmp_path):
    settings = Settings(state_dir=tmp_path)
    registry = ModuleRegistry()
    app = App(
        settings,
        overrides=AppBuildOverrides(
            gate=AllowAllGate(),
            backends=[StubBackend()],
            modules=registry,
            director=RuleDirector(),
            audit_mode=False,  # not audit mode
        ),
    )
    await app.start()
    try:
        sid = "ses_non_audit"
        await app.handle(ControlRequest(
            session_id=sid, message=Message(role="user", content="hi"),
        ))
    finally:
        await app.stop()

    report = await replay_session(
        audit_root=settings.state_dir / "audit",
        session_id=sid,
        backend=StubBackend(),
    )
    # require_audit_mode=True by default -> skipped
    assert report.total == 1
    assert report.skipped == 1
    assert report.matched == 0


async def test_replay_detects_backend_drift(audit_app):
    app, settings = audit_app
    sid = "ses_drift"
    await app.handle(ControlRequest(
        session_id=sid, message=Message(role="user", content="hello drift"),
    ))

    # Replay against a stub with a different prefix -> content must differ
    different = StubBackend(prefix="[different]")
    report = await replay_session(
        audit_root=settings.state_dir / "audit",
        session_id=sid,
        backend=different,
    )
    assert report.total == 1
    assert report.matched == 0
    assert report.match_rate == 0.0
    assert not report.diffs[0].matched
