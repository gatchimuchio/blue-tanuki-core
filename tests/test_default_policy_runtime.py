from blue_tanuki_core.app import App
from blue_tanuki_core.contracts import ApprovalResponse, ControlRequest, Message
from blue_tanuki_core.settings import Settings


async def test_default_policy_suspends_file_write(tmp_path):
    app = App(Settings(state_dir=tmp_path))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_default_write",
            message=Message(role="user", content="write file hello.txt: konnichiwa"),
        ))
        assert resp.status.code == "suspended"
        assert resp.pending is not None
        assert resp.pending.kind == "approval"
        assert resp.pending.approval is not None
        assert resp.pending.approval.reason == "file write requires approval"

        token = resp.status.resume_token
        assert token is not None
        resp2 = await app.resume(
            token, approval=ApprovalResponse(token=token, decision="approved"),
        )
        assert resp2.status.code == "ok"
        assert (tmp_path / "workspace" / "hello.txt").read_text() == "konnichiwa"
    finally:
        await app.stop()


async def test_default_policy_suspends_external_fetch(tmp_path):
    app = App(Settings(state_dir=tmp_path))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_default_fetch",
            message=Message(role="user", content="fetch https://example.com"),
        ))
        assert resp.status.code == "suspended"
        assert resp.pending is not None
        assert resp.pending.kind == "approval"
        assert resp.pending.approval is not None
        assert resp.pending.approval.reason == "external network request requires approval"
    finally:
        await app.stop()


async def test_default_policy_stops_sensitive_file_write(tmp_path):
    app = App(Settings(state_dir=tmp_path))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_default_stop",
            message=Message(role="user", content="write file .env: SECRET=1"),
        ))
        assert resp.status.code == "stopped"
        assert "credential-like" in (resp.status.reason or "")
    finally:
        await app.stop()


async def test_default_policy_suspends_large_llm_payload(tmp_path):
    app = App(Settings(state_dir=tmp_path))
    await app.start()
    try:
        big = "x" * 250000
        resp = await app.handle(ControlRequest(
            session_id="ses_default_big",
            message=Message(role="user", content=big),
        ))
        assert resp.status.code == "suspended"
        assert resp.pending is not None
        assert resp.pending.kind == "approval"
        assert resp.pending.approval is not None
        assert resp.pending.approval.reason == "large LLM payload requires approval"
    finally:
        await app.stop()
