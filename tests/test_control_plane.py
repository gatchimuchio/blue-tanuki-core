from pathlib import Path

import pytest

from blue_tanuki_core.contracts import (
    ApprovalResponse, ClarifyResponse, ControlRequest, Message,
)
from blue_tanuki_core.app import App, AppBuildOverrides
from blue_tanuki_core.gate import AllowAllGate, DenyAllGate, PolicyGate, CompiledRule
from blue_tanuki_core.gate import RawRule
from blue_tanuki_core.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path)


async def test_happy_path_ok(settings):
    app = App(settings, overrides=AppBuildOverrides(gate=AllowAllGate()))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_happy",
            message=Message(role="user", content="hello"),
        ))
        assert resp.status.code == "ok"
        assert resp.output[0].role == "assistant"
    finally:
        await app.stop()


async def test_deny_all_stops(settings):
    app = App(settings, overrides=AppBuildOverrides(gate=DenyAllGate()))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_deny",
            message=Message(role="user", content="hello"),
        ))
        assert resp.status.code == "stopped"
    finally:
        await app.stop()


async def test_suspend_clarify_then_resume_ok(settings):
    rules = [
        CompiledRule(RawRule(
            id="clarify_empty",
            when={"direction": "inbound", "content_match": "^$"},
            action="suspend_clarify",
            clarify_prompt="もう一度お願いします",
        )),
    ]
    app = App(settings, overrides=AppBuildOverrides(gate=PolicyGate(rules)))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_clar",
            message=Message(role="user", content=""),
        ))
        assert resp.status.code == "suspended"
        token = resp.status.resume_token
        assert token is not None
        assert resp.pending is not None
        assert resp.pending.kind == "clarify"
        assert resp.pending.clarify is not None
        assert resp.pending.clarify.prompt == "もう一度お願いします"

        resp2 = await app.resume(
            token,
            clarify=ClarifyResponse(token=token, answer="こんにちは"),
        )
        assert resp2.status.code == "ok"
        assert "こんにちは" in resp2.output[0].content
    finally:
        await app.stop()


async def test_suspend_approval_rejected_stops(settings):
    # approve at inbound for simplicity (pre_llm would work too)
    rules = [
        CompiledRule(RawRule(
            id="approve_all",
            when={"direction": "inbound"},
            action="suspend_approval",
        )),
    ]
    app = App(settings, overrides=AppBuildOverrides(gate=PolicyGate(rules)))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_app",
            message=Message(role="user", content="do it"),
        ))
        assert resp.status.code == "suspended"
        token = resp.status.resume_token
        resp2 = await app.resume(
            token,
            approval=ApprovalResponse(token=token, decision="rejected"),
        )
        assert resp2.status.code == "stopped"
    finally:
        await app.stop()


async def test_suspend_approval_approved_continues(settings):
    rules = [
        CompiledRule(RawRule(
            id="approve_all_inbound",
            when={"direction": "inbound"},
            action="suspend_approval",
        )),
    ]
    app = App(settings, overrides=AppBuildOverrides(gate=PolicyGate(rules)))
    await app.start()
    try:
        resp = await app.handle(ControlRequest(
            session_id="ses_app2",
            message=Message(role="user", content="do it"),
        ))
        assert resp.status.code == "suspended"
        token = resp.status.resume_token
        resp2 = await app.resume(
            token,
            approval=ApprovalResponse(token=token, decision="approved"),
        )
        # After approval, inbound gate will fire again on resume — but our policy
        # would loop. Accept either ok (if the second run's state skips inbound)
        # or stopped if the skeleton re-hits inbound.
        # In this skeleton, resume restores state.phase=pre_llm so inbound is skipped.
        assert resp2.status.code == "ok"
    finally:
        await app.stop()
