"""Small, dependency-free observability helpers for provider calls.

Events are intentionally written to stderr because stdout is reserved for the
MCP JSON-RPC stream. Query and URL values are represented by short hashes so
logs remain useful for correlation without retaining user content.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any


_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "web_research_correlation_id", default=None
)
_provider_status: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "web_research_provider_status", default=None
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def correlation_id() -> str:
    """Return the current request id, creating one for standalone calls."""
    current = _correlation_id.get()
    if current is None:
        current = uuid.uuid4().hex
        _correlation_id.set(current)
    return current


@contextlib.contextmanager
def request_context() -> Iterator[str]:
    """Give one MCP tool invocation a correlation id."""
    token = _correlation_id.set(uuid.uuid4().hex)
    try:
        yield _correlation_id.get() or "unknown"
    finally:
        _correlation_id.reset(token)


def _estimated_cost(provider: str) -> float:
    name = f"WEB_RESEARCH_COST_USD_{provider.upper().replace('-', '_')}"
    try:
        return max(0.0, float(os.environ.get(name, "0")))
    except ValueError:
        return 0.0


def emit(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "timestamp": time.time(),
        "correlation_id": correlation_id(),
        **fields,
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str), file=sys.stderr, flush=True)


def provider_started(provider: str, operation: str, target: str) -> float:
    """Record the start time and a redacted target for a provider operation."""
    _provider_status.set(None)
    emit(
        "provider.started",
        provider=provider,
        operation=operation,
        target_fingerprint=_fingerprint(target),
    )
    return time.perf_counter()


def provider_finished(
    provider: str,
    operation: str,
    started: float,
    result_count: int,
    *,
    partial: bool = False,
    status: str | None = None,
) -> str:
    resolved_status = status or _provider_status.get() or ("partial" if partial else "ok")
    emit(
        "provider.finished",
        provider=provider,
        operation=operation,
        status=resolved_status,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        result_count=result_count,
        partial=partial,
        estimated_cost_usd=_estimated_cost(provider),
    )
    _provider_status.set(None)
    return resolved_status


def provider_failed(provider: str, operation: str, exc: BaseException) -> None:
    """Emit safe failure health data without serializing exception details."""
    response = getattr(exc, "response", None)
    rate_limited = getattr(response, "status_code", None) == 429
    status = "rate_limited" if rate_limited else "error"
    _provider_status.set(status)
    retry_after = None
    if rate_limited:
        raw_retry_after = getattr(response, "headers", {}).get("Retry-After")
        try:
            retry_after = max(0.0, float(raw_retry_after)) if raw_retry_after is not None else None
        except (TypeError, ValueError):
            retry_after = None
    emit(
        "provider.failed",
        provider=provider,
        operation=operation,
        status=status,
        error_type=type(exc).__name__,
        **({"retry_after_seconds": retry_after} if retry_after is not None else {}),
        estimated_cost_usd=_estimated_cost(provider),
    )
