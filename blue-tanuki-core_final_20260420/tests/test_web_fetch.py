import httpx
import pytest

from blue_tanuki_core.contracts import (
    CallContext, ModuleRequest, TypedPayload, new_call_id,
)
from blue_tanuki_core.errors import ModuleRefused
from blue_tanuki_core.modules.web_fetch import WebFetchModule


def _req(url: str) -> ModuleRequest:
    return ModuleRequest(
        call_id=new_call_id(),
        request_id="r", session_id="s", turn_id="t",
        module="web_fetch", caller="director",
        payload=TypedPayload(kind="web_fetch.get", data={"url": url}),
        context=CallContext(op="get", resource=url),
    )


def _make(handler, **kw) -> WebFetchModule:
    m = WebFetchModule(allow_private_network=True, **kw)
    m._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return m


async def test_happy_text():
    def handler(req):
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
    m = _make(handler)
    r = await m.handle(_req("https://example.com/x"))
    assert r.status.code == "ok"
    assert r.result["content"] == "hello"
    assert r.result["bytes"] == 5
    assert r.result["status"] == 200


async def test_redirect_followed():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        if req.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        return httpx.Response(200, text="final", headers={"content-type": "text/plain"})
    m = _make(handler)
    r = await m.handle(_req("https://example.com/a"))
    assert r.result["content"] == "final"
    assert r.result["url"].endswith("/b")
    assert calls["n"] == 2


async def test_too_many_redirects():
    def handler(req):
        return httpx.Response(302, headers={"location": "/loop"})
    m = _make(handler, max_redirects=3)
    with pytest.raises(ModuleRefused):
        await m.handle(_req("https://example.com/a"))


async def test_invalid_scheme_refused():
    m = WebFetchModule(allow_private_network=True)
    with pytest.raises(ModuleRefused):
        await m.handle(_req("ftp://example.com/x"))


async def test_empty_url_refused():
    m = WebFetchModule(allow_private_network=True)
    with pytest.raises(ModuleRefused):
        await m.handle(_req(""))


async def test_allowlist_blocks_nonmatching():
    def handler(req):
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})
    m = _make(handler, hostname_allowlist=["allowed.com"])
    with pytest.raises(ModuleRefused):
        await m.handle(_req("https://forbidden.com/x"))


async def test_allowlist_wildcard():
    def handler(req):
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})
    m = _make(handler, hostname_allowlist=["*.example.com"])
    r = await m.handle(_req("https://api.example.com/v1"))
    assert r.status.code == "ok"


async def test_size_cap_truncates():
    def handler(req):
        return httpx.Response(200, text="x" * 5000,
                              headers={"content-type": "text/plain"})
    m = _make(handler, max_bytes=1000)
    r = await m.handle(_req("https://example.com/big"))
    assert r.result["bytes"] == 1000
    assert r.result["truncated"] is True


async def test_ssrf_private_ip_blocked():
    # allow_private_network=False (default) with a real private hostname
    m = WebFetchModule(allow_private_network=False)
    with pytest.raises(ModuleRefused):
        await m.handle(_req("http://127.0.0.1:8080/x"))


async def test_binary_content_base64_encoded():
    def handler(req):
        return httpx.Response(
            200, content=b"\x89PNG\r\n\x1a\n",
            headers={"content-type": "image/png"},
        )
    m = _make(handler)
    r = await m.handle(_req("https://example.com/img.png"))
    # not text/* or json -> base64 encoded string
    import base64
    assert base64.b64decode(r.result["content"]).startswith(b"\x89PNG")
