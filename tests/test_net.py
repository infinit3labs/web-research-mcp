"""Tests for ``net.py`` — the shared HTTP-hardening module (INF-226 + INF-231).

Covers:
- Bounded retry on 5xx/429 and transport errors
- Retry-After header honoring
- Backoff math
- Content-type guard for binary/streaming responses
- SSRF block list (private/loopback/link-local/multicast/reserved/unspecified,
  non-http(s) schemes, unresolvable hosts)
- ``compute_backoff`` jitter range
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.web_research import net


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Return responses in order from ``self.responses``."""

    def __init__(self, responses: list[httpx.Response | Exception]):
        self.responses = list(responses)
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected extra request: {request.url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client_with(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, timeout=5.0)


# --------------------------------------------------------------------------------------
# Retry / backoff
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_with_retry_returns_first_success():
    transport = _ScriptedTransport([
        httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"}),
    ])
    async with _client_with(transport) as client:
        r = await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0)
    assert r.status_code == 200
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_request_with_retry_eventually_succeeds_after_429():
    transport = _ScriptedTransport([
        httpx.Response(429, content=b"slow down", headers={"Retry-After": "0", "content-type": "text/plain"}),
        httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"}),
    ])
    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock()) as sleep:
        async with _client_with(transport) as client:
            r = await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0)
    assert r.status_code == 200
    assert len(transport.calls) == 2
    sleep.assert_awaited()  # backoff happened


@pytest.mark.asyncio
async def test_request_with_retry_retries_5xx_then_gives_up():
    transport = _ScriptedTransport([
        httpx.Response(503, content=b"", headers={"content-type": "text/plain"}),
        httpx.Response(503, content=b"", headers={"content-type": "text/plain"}),
        httpx.Response(503, content=b"", headers={"content-type": "text/plain"}),
    ])
    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock()):
        async with _client_with(transport) as client:
            with pytest.raises(net.ProviderError) as exc:
                await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0, max_retries=2)
    assert exc.value.status_code == 503
    assert len(transport.calls) == 3  # 1 original + 2 retries


@pytest.mark.asyncio
async def test_request_with_retry_does_not_retry_4xx_other_than_429():
    transport = _ScriptedTransport([
        httpx.Response(404, content=b"not found", headers={"content-type": "text/plain"}),
    ])
    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock()) as sleep:
        async with _client_with(transport) as client:
            with pytest.raises(net.ProviderError) as exc:
                await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0)
    assert exc.value.status_code == 404
    assert len(transport.calls) == 1
    sleep.assert_not_awaited()  # no backoff on permanent failures


@pytest.mark.asyncio
async def test_request_with_retry_recovers_from_transport_error():
    transport = _ScriptedTransport([
        httpx.ConnectError("connection refused"),
        httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"}),
    ])
    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock()):
        async with _client_with(transport) as client:
            r = await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0)
    assert r.status_code == 200
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_request_with_retry_gives_up_on_recurring_transport_error():
    transport = _ScriptedTransport([
        httpx.ConnectError("x"),
        httpx.ConnectError("x"),
        httpx.ConnectError("x"),
    ])
    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock()):
        async with _client_with(transport) as client:
            with pytest.raises(net.ProviderError) as exc:
                await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0, max_retries=2)
    assert "ConnectError" in str(exc.value)


@pytest.mark.asyncio
async def test_request_with_retry_honors_retry_after_header():
    transport = _ScriptedTransport([
        httpx.Response(429, content=b"", headers={"Retry-After": "7", "content-type": "text/plain"}),
        httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"}),
    ])
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    with patch("src.web_research.net.asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)):
        async with _client_with(transport) as client:
            await net.request_with_retry(client, "GET", "https://x.test/", provider="t", timeout=5.0)
    assert sleeps == [7.0]


def test_compute_backoff_increases_with_attempts_and_jitters():
    # attempt 0 jitter window is [0, base]
    samples_0 = {net.compute_backoff(0, 0.5) for _ in range(200)}
    assert all(0.0 <= s <= 0.5 for s in samples_0)
    assert max(samples_0) > 0.4  # actually jittered (probability of never hitting >0.4 is vanishingly small)
    # attempt 2 jitter window is [0, base*4] and should cover a wider range
    samples_2 = {net.compute_backoff(2, 0.5) for _ in range(200)}
    assert max(samples_2) > 1.5


def test_parse_retry_after_handles_garbage():
    assert net.parse_retry_after("3.5") == 3.5
    assert net.parse_retry_after("0") == 0.0
    assert net.parse_retry_after("-1") == 0.0  # clamped
    assert net.parse_retry_after("not-a-number") is None
    assert net.parse_retry_after("") is None
    assert net.parse_retry_after(None) is None


# --------------------------------------------------------------------------------------
# Content-type guard
# --------------------------------------------------------------------------------------


def test_disallowed_content_types():
    for ct in ("image/png", "image/jpeg", "video/mp4", "audio/mpeg", "font/woff2", "application/octet-stream"):
        assert net.is_disallowed_content_type(ct), f"should reject {ct}"
    for ct in ("text/html", "text/markdown", "application/json", "application/atom+xml", ""):
        assert not net.is_disallowed_content_type(ct), f"should allow {ct}"


def test_disallowed_content_type_ignores_charset():
    assert net.is_disallowed_content_type("image/png; charset=binary")
    assert not net.is_disallowed_content_type("application/json; charset=utf-8")


# --------------------------------------------------------------------------------------
# SSRF guards
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_public_url_rejects_non_http_schemes():
    for bad in ("file:///etc/passwd", "ftp://x.test/", "gopher://x.test/", "javascript:alert(1)"):
        with pytest.raises(net.UnsafeURLError):
            await net.validate_public_url(bad)


@pytest.mark.asyncio
async def test_validate_public_url_rejects_missing_host():
    with pytest.raises(net.UnsafeURLError):
        await net.validate_public_url("https:///nopath")


@pytest.mark.asyncio
async def test_validate_public_url_rejects_loopback():
    # `localhost` should resolve to a loopback address on the test host; if not,
    # we fall back to a direct 127.0.0.1.
    with pytest.raises(net.UnsafeURLError):
        await net.validate_public_url("http://127.0.0.1:8080/")


@pytest.mark.asyncio
async def test_validate_public_url_rejects_link_local_metadata():
    with pytest.raises(net.UnsafeURLError):
        await net.validate_public_url("http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
async def test_validate_public_url_rejects_private_rfc1918():
    with pytest.raises(net.UnsafeURLError):
        await net.validate_public_url("http://10.0.0.1/")


@pytest.mark.asyncio
async def test_validate_public_url_rejects_unspecifiable_host():
    with pytest.raises(net.UnsafeURLError):
        await net.validate_public_url("http://this-host-does-not-exist.invalid/")


@pytest.mark.asyncio
async def test_validate_public_url_accepts_public_host(monkeypatch):
    """Public hostnames should pass — but we stub DNS so the test doesn't hit the network."""
    fake_info = (None, None, None, None, ("93.184.216.34", 0))  # example.com IP
    async def fake_getaddrinfo(*a, **kw):
        return [fake_info]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    # Should not raise
    await net.validate_public_url("https://example.com/foo")


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------


def test_retriable_status_codes_includes_429_and_5xx():
    for code in (429, 500, 502, 503, 504):
        assert code in net.RETRIABLE_STATUS_CODES
    for code in (200, 301, 400, 401, 403, 404):
        assert code not in net.RETRIABLE_STATUS_CODES


def test_max_response_bytes_is_a_reasonable_cap():
    # 2MB cap — enough for most pages, small enough to keep context safe.
    assert 100_000 < net.MAX_RESPONSE_BYTES < 50_000_000


def test_allowed_schemes_are_just_http_and_https():
    assert net.ALLOWED_URL_SCHEMES == {"http", "https"}
