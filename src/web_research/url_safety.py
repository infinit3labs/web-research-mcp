"""URL safety policy for outbound page fetches."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


METADATA_HOSTNAMES = frozenset(
    {
        "metadata",
        "metadata.azure.com",
        "metadata.google.internal",
        "metadata.google.com",
        "instance-data.ec2.internal",
    }
)


class UrlSafetyError(ValueError):
    """Raised when a URL violates the outbound fetch policy."""


def validate_url(url: str, *, resolve: bool = True) -> None:
    if not isinstance(url, str) or not url or any(ord(char) < 0x20 for char in url):
        raise UrlSafetyError("blocked malformed URL")

    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError as exc:
        raise UrlSafetyError("blocked malformed URL") from exc

    if scheme not in {"http", "https"}:
        raise UrlSafetyError("blocked URL scheme; only http and https are allowed")
    if not hostname:
        raise UrlSafetyError("blocked URL without a hostname")
    if username or password:
        raise UrlSafetyError("blocked URL with embedded credentials")
    if port is not None and not 1 <= port <= 65535:
        raise UrlSafetyError("blocked URL with an invalid port")

    host = hostname.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UrlSafetyError("blocked URL with an invalid hostname") from exc
    if host.lower() in METADATA_HOSTNAMES:
        raise UrlSafetyError("blocked cloud metadata hostname")
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        _ensure_public_ip(literal_ip)
    elif resolve:
        try:
            addresses = resolve_host_ips(host)
        except OSError as exc:
            raise UrlSafetyError("blocked URL because its hostname could not be resolved") from exc
        if not addresses:
            raise UrlSafetyError("blocked URL because its hostname has no address")
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise UrlSafetyError("blocked URL because its hostname resolved to an invalid address") from exc
            _ensure_public_ip(parsed_address)


def _ensure_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not address.is_global:
        raise UrlSafetyError(f"blocked URL to non-public address {address}")


def resolve_host_ips(host: str) -> tuple[str, ...]:
    results = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(result[4][0] for result in results))
