"""web_fetch: HTTP GET with strict defaults.

Contract:
- payload.kind = "web_fetch.get"
- data: {"url": str, "timeout_s": float (optional)}
- result: {"url": final_url, "status": int, "content": str, "bytes": int,
           "content_type": str, "truncated": bool}

Safety defaults:
- Scheme must be http or https
- Host must resolve to a public IP (no loopback / private / link-local)
- Optional hostname allowlist (exact or wildcard "*.example.com")
- Redirects followed; each hop re-validated
- Response size capped; text/* and json decoded, else base64-truncated string
- Timeout per request (default 15s)
"""
from __future__ import annotations

import base64
import ipaddress
import socket
import time
from typing import Any

import httpx

from ..contracts import ModuleRequest, ModuleResponse, ModuleStatus, SideEffect
from ..control_plane import Module
from ..errors import ModuleRefused


DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_REDIRECTS = 5


def _is_public_ip(host: str) -> bool:
    """Resolve host and verify all resolved addresses are globally routable."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


def _host_matches_allowlist(host: str, patterns: list[str]) -> bool:
    if not patterns:
        return True  # no allowlist = permit (SSRF check still applies)
    host = host.lower()
    for pat in patterns:
        pat = pat.lower()
        if pat == host:
            return True
        if pat.startswith("*.") and host.endswith(pat[1:]):
            return True
    return False


class WebFetchModule(Module):
    name = "web_fetch"

    def __init__(
        self,
        *,
        hostname_allowlist: list[str] | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
        allow_private_network: bool = False,
    ):
        self.allowlist = hostname_allowlist or []
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.default_timeout_s = default_timeout_s
        self.allow_private_network = allow_private_network
        self._client: httpx.AsyncClient | None = None

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                max_redirects=0,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _check_url(self, url: str) -> httpx.URL:
        try:
            u = httpx.URL(url)
        except Exception as e:
            raise ModuleRefused("web_fetch", f"invalid url: {url}") from e
        if u.scheme not in ("http", "https"):
            raise ModuleRefused("web_fetch", f"unsupported scheme: {u.scheme}")
        if not u.host:
            raise ModuleRefused("web_fetch", "missing host")
        if not _host_matches_allowlist(u.host, self.allowlist):
            raise ModuleRefused(
                "web_fetch",
                f"host not in allowlist: {u.host}",
            )
        if not self.allow_private_network and not _is_public_ip(u.host):
            raise ModuleRefused(
                "web_fetch",
                f"host resolves to non-public address: {u.host}",
            )
        return u

    async def handle(self, req: ModuleRequest) -> ModuleResponse:
        t0 = time.perf_counter()
        kind = req.payload.kind
        data = req.payload.data
        if kind != "web_fetch.get":
            raise ModuleRefused("web_fetch", f"unknown payload kind: {kind}")

        url = str(data.get("url") or "")
        if not url:
            raise ModuleRefused("web_fetch", "empty url")
        timeout_s = float(data.get("timeout_s") or self.default_timeout_s)

        self._check_url(url)

        client = await self._client_get()
        current = url
        hops = 0
        final_resp: httpx.Response | None = None
        final_url = current
        while True:
            self._check_url(current)
            try:
                r = await client.get(current, timeout=timeout_s)
            except httpx.TimeoutException as e:
                raise ModuleRefused("web_fetch", f"timeout: {e}") from e
            except httpx.HTTPError as e:
                raise ModuleRefused("web_fetch", f"http error: {e}") from e
            final_resp = r
            final_url = current
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    break
                hops += 1
                if hops > self.max_redirects:
                    raise ModuleRefused("web_fetch", "too many redirects")
                current = str(httpx.URL(current).join(loc))
                continue
            break

        assert final_resp is not None
        ctype = final_resp.headers.get("content-type", "")
        raw = final_resp.content
        truncated = False
        if len(raw) > self.max_bytes:
            raw = raw[: self.max_bytes]
            truncated = True

        if ctype.startswith("text/") or "json" in ctype:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = base64.b64encode(raw).decode("ascii")
        else:
            text = base64.b64encode(raw).decode("ascii")

        result: dict[str, Any] = {
            "url": final_url,
            "status": final_resp.status_code,
            "content": text,
            "bytes": len(raw),
            "content_type": ctype,
            "truncated": truncated,
        }
        se = [SideEffect(
            kind="web.fetch", target=final_url,
            bytes_in=len(raw),
            detail={"status": final_resp.status_code, "content_type": ctype},
        )]
        return ModuleResponse(
            call_id=req.call_id,
            status=ModuleStatus(code="ok"),
            result=result,
            side_effects=se,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )


__all__ = ["WebFetchModule"]
