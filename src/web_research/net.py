"""Shared HTTP hardening: bounded retry/backoff and SSRF-safe URL validation.

Used by every provider in ``providers.py`` so retry/backoff/timeout behavior
and outbound-URL safety checks live in one place instead of being
reimplemented per source.
"""

from __future__ import annotations

import asyncio
import ipaddress
import random
from urllib.parse import urlparse

import httpx


class ProviderError(Exception):
    """A classified provider-level failure (as opposed to a raw exception)."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        rate_limited: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.rate_limited = rate_limited


class UnsafeURLError(Exception):
    """Raised when a URL fails SSRF safety validation."""


# Status codes worth retrying: 429 (rate limit) and 5xx (transient upstream failure).
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}

ALLOWED_URL_SCHEMES = {"http", "https"}

# Response size cap enforced while streaming, before any post-hoc truncation.
MAX_RESPONSE_BYTES = 2_000_000

_DISALLOWED_CONTENT_TYPE_PREFIXES = ("image/", "video/", "audio/", "font/")
_DISALLOWED_CONTENT_TYPES = {"application/octet-stream"}


def is_disallowed_content_type(content_type: str) -> bool:
    """True if a response Content-Type is binary/non-text and should be rejected."""
    ct = content_type.split(";", 1)[0].strip().lower()
    if not ct:
        return False
    if ct in _DISALLOWED_CONTENT_TYPES:
        return True
    return any(ct.startswith(p) for p in _DISALLOWED_CONTENT_TYPE_PREFIXES)


def compute_backoff(attempt: int, base: float) -> float:
    """Full-jitter exponential backoff delay for retry attempt `attempt` (0-indexed)."""
    return random.uniform(0, base * (2**attempt))


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value (seconds form only) into a float delay."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider: str,
    timeout: float,
    max_retries: int = 2,
    backoff_base: float = 0.5,
    **kwargs,
) -> httpx.Response:
    """Issue an HTTP request with bounded retries and backoff.

    Retries on connection/timeout errors and on 429/5xx responses (honoring
    Retry-After when present). Any other 4xx is a permanent failure and is
    raised immediately without retrying.
    """
    attempt = 0
    while True:
        try:
            response = await client.request(method, url, timeout=timeout, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt >= max_retries:
                raise ProviderError(provider, f"{type(e).__name__}: {e}") from e
            await asyncio.sleep(compute_backoff(attempt, backoff_base))
            attempt += 1
            continue

        if response.status_code in RETRIABLE_STATUS_CODES:
            if attempt >= max_retries:
                raise ProviderError(
                    provider,
                    f"HTTP {response.status_code} after {attempt + 1} attempt(s)",
                    status_code=response.status_code,
                    rate_limited=response.status_code == 429,
                )
            delay = parse_retry_after(response.headers.get("Retry-After"))
            if delay is None:
                delay = compute_backoff(attempt, backoff_base)
            await asyncio.sleep(delay)
            attempt += 1
            continue

        if response.status_code >= 400:
            raise ProviderError(
                provider,
                f"HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        return response


async def validate_public_url(url: str) -> None:
    """Raise UnsafeURLError unless `url` is a safe, public http(s) target.

    Blocks non-http(s) schemes and any hostname that resolves (via all
    A/AAAA records) only to private, loopback, link-local (including the
    169.254.169.254 cloud metadata address), multicast, reserved, or
    unspecified addresses. This runs before the URL is ever embedded in an
    outbound request, so it applies equally to direct fetches and to
    proxy-style fetchers (e.g. Jina Reader).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"unsupported URL scheme {parsed.scheme!r}; only http/https are allowed")
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError("URL is missing a hostname")

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, None)
    except OSError as e:
        raise UnsafeURLError(f"could not resolve host {hostname!r}: {e}") from e
    if not infos:
        raise UnsafeURLError(f"could not resolve host {hostname!r}")

    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise UnsafeURLError(f"refusing to fetch {hostname!r}: resolves to non-public address {ip}")
