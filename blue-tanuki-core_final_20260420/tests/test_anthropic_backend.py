import httpx
import pytest

from blue_tanuki_core.backend.anthropic import AnthropicBackend
from blue_tanuki_core.backend.base import LLMCallRequest
from blue_tanuki_core.contracts import Message
from blue_tanuki_core.errors import LLMProviderError


def _success_body(text: str = "hello from claude") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


async def _run_with_handler(backend, handler, req):
    # inject mock transport
    backend._client = httpx.AsyncClient(
        base_url=backend.base_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        return await backend.call(req, timeout_s=5)
    finally:
        await backend.aclose()


async def test_happy_path():
    b = AnthropicBackend(api_key="sk-test")
    def handler(req):
        assert req.url.path == "/v1/messages"
        assert req.headers["x-api-key"] == "sk-test"
        return httpx.Response(200, json=_success_body())
    req = LLMCallRequest(
        model="claude-test",
        messages=[Message(role="user", content="hi")],
        max_tokens=64,
    )
    resp = await _run_with_handler(b, handler, req)
    assert resp.provider == "anthropic"
    assert resp.content == "hello from claude"
    assert resp.finish_reason == "stop"
    assert resp.usage_prompt_tokens == 10


async def test_4xx_raises_provider_error():
    b = AnthropicBackend(api_key="sk-test", max_retries=2)
    def handler(req):
        return httpx.Response(400, json={"error": "bad"})
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=8,
    )
    with pytest.raises(LLMProviderError):
        await _run_with_handler(b, handler, req)


async def test_5xx_then_success_retries():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=_success_body("ok"))
    b = AnthropicBackend(api_key="sk-test", max_retries=3)
    req = LLMCallRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=8,
    )
    resp = await _run_with_handler(b, handler, req)
    assert resp.content == "ok"
    assert calls["n"] == 2


async def test_system_message_folded():
    captured: dict = {}
    def handler(req):
        import json
        captured["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())
    b = AnthropicBackend(api_key="sk-test")
    req = LLMCallRequest(
        model="m", max_tokens=8,
        messages=[
            Message(role="system", content="you are concise"),
            Message(role="user", content="hi"),
        ],
    )
    await _run_with_handler(b, handler, req)
    assert captured["body"]["system"] == "you are concise"
    # first message must be user
    assert captured["body"]["messages"][0]["role"] == "user"


async def test_seed_stored_in_metadata():
    captured: dict = {}
    def handler(req):
        import json
        captured["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())
    b = AnthropicBackend(api_key="sk-test")
    req = LLMCallRequest(
        model="m", max_tokens=8, temperature=0, seed=42,
        messages=[Message(role="user", content="x")],
    )
    await _run_with_handler(b, handler, req)
    assert captured["body"]["metadata"]["seed"] == 42
    assert captured["body"]["temperature"] == 0
