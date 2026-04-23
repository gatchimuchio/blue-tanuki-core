import pytest

from blue_tanuki_core.backend.stub import StubBackend
from blue_tanuki_core.backend.base import LLMCallRequest
from blue_tanuki_core.contracts import Message
from blue_tanuki_core.errors import ConfigError, LLMPromptTooLarge
from blue_tanuki_core.pipe import (
    BudgetGuard, LLMPipe, NullTelemetry, PromptSizeGuard,
)


def _pipe(max_input_tokens=100_000, tokens_cap=None) -> LLMPipe:
    return LLMPipe(
        backends=[StubBackend()],
        budget=BudgetGuard(max_tokens_per_turn=tokens_cap),
        size_guard=PromptSizeGuard(max_input_tokens=max_input_tokens),
        telemetry=NullTelemetry(),
    )


async def test_happy_path():
    pipe = _pipe()
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="hi")], max_tokens=64,
    )
    resp = await pipe.call(req)
    assert resp.provider == "stub"
    assert "hi" in resp.content


async def test_audit_mode_requires_t0():
    pipe = _pipe()
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=64,
        temperature=0.7, seed=1,
    )
    with pytest.raises(ConfigError):
        await pipe.call(req, audit_mode=True)


async def test_audit_mode_requires_seed():
    pipe = _pipe()
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=64,
        temperature=0,
    )
    with pytest.raises(ConfigError):
        await pipe.call(req, audit_mode=True)


async def test_audit_mode_is_deterministic():
    pipe = _pipe()
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="deterministic")],
        max_tokens=64, temperature=0, seed=42,
    )
    r1 = await pipe.call(req, audit_mode=True)
    r2 = await pipe.call(req, audit_mode=True)
    assert r1.content == r2.content


async def test_prompt_size_guard():
    pipe = _pipe(max_input_tokens=1)
    req = LLMCallRequest(
        model="m",
        messages=[Message(role="user", content="x" * 1000)],
        max_tokens=64,
    )
    with pytest.raises(LLMPromptTooLarge):
        await pipe.call(req)
