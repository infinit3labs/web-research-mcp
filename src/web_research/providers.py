"""Multi-source search providers.

Each provider exposes a single async function: ``async def search(query, max_results, client) -> list[Result]``

Providers gracefully degrade on failure (return [], log to stderr) so a single
broken source never sinks the whole research run.
"""

from __future__ import annotations

import copy
import os
import re
import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol, TypedDict, overload
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from . import observability
from .cache import TTLCache
from .url_safety import UrlSafetyError, validate_url


USER_AGENT = "research-agent/0.1 (+https://github.com/local/web-research-mcp; mailto:research@local)"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
FETCH_MAX_BYTES = 5 * 1024 * 1024
FETCH_MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 45.0
FETCH_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/xhtml+xml",
        "application/xml",
        "text/html",
        "text/markdown",
        "text/plain",
        "text/xml",
    }
)


class ProviderRateLimitError(httpx.HTTPStatusError):
    """Retain the final 429 response so callers can report retry guidance."""

    def __init__(self, response: Any):
        request = getattr(response, "request", None) or httpx.Request("GET", "https://provider.invalid")
        super().__init__("provider rate limit exhausted", request=request, response=response)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a bounded float setting without allowing invalid env to break calls."""
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    """Read a bounded integer setting without allowing invalid env to break calls."""
    try:
        value = max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
    return min(value, maximum) if maximum is not None else value


def _provider_env_name(prefix: str, provider: str) -> str:
    return f"{prefix}_{provider.upper().replace('-', '_')}_SECONDS"


def _provider_float(prefix: str, provider: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a provider-specific setting, falling back to the global setting."""
    provider_name = _provider_env_name(prefix, provider)
    if provider_name in os.environ:
        return _env_float(provider_name, default, minimum=minimum)
    return _env_float(f"{prefix}_SECONDS", default, minimum=minimum)


def _provider_int(prefix: str, provider: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    provider_name = f"{prefix}_{provider.upper().replace('-', '_')}"
    if provider_name in os.environ:
        return _env_int(provider_name, default, minimum=minimum, maximum=maximum)
    return _env_int(f"{prefix}", default, minimum=minimum, maximum=maximum)


def _request_timeout(provider: str, default: float) -> float:
    # Configuration may tighten a provider's budget, but cannot remove the
    # built-in upper bound that keeps a single upstream from stalling a run.
    return min(_provider_float("WEB_RESEARCH_TIMEOUT", provider, default, minimum=0.1), default)


def _retry_after(response: httpx.Response, fallback: float) -> float:
    """Return a safe delay from Retry-After, supporting seconds and HTTP dates."""
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return fallback
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except ValueError:
            return fallback


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider: str,
    timeout: float,
    **kwargs: Any,
) -> httpx.Response:
    """Make a bounded, rate-limit-aware HTTP request.

    Provider functions still own parsing and public error behavior; this helper only
    adds retries for transient transport/server failures and HTTP 429 responses.
    """
    stream_response = bool(kwargs.pop("_stream_response", False))
    max_bytes = kwargs.pop("_max_bytes", None)
    allowed_content_types = kwargs.pop("_allowed_content_types", None)
    timeout_override = kwargs.pop("_timeout_override", None)
    retries = _provider_int("WEB_RESEARCH_MAX_RETRIES", provider, 2, minimum=0, maximum=5)
    backoff = _provider_float("WEB_RESEARCH_RETRY_BACKOFF", provider, 0.25, minimum=0.0)
    max_backoff = _provider_float("WEB_RESEARCH_MAX_BACKOFF", provider, 5.0, minimum=0.0)
    request_method = getattr(client, method.lower(), None)

    async def perform_request() -> httpx.Response:
        if not stream_response or not hasattr(client, "stream"):
            if request_method is None:
                raise TypeError(f"HTTP client does not support {method}")
            request_timeout = timeout_override if timeout_override is not None else _request_timeout(provider, timeout)
            response = await request_method(url, timeout=request_timeout, **kwargs)
            if max_bytes is not None and not _is_redirect_response(response):
                _validate_response_headers(response, max_bytes, allowed_content_types, buffered=True)
            return response

        request_timeout = timeout_override if timeout_override is not None else _request_timeout(provider, timeout)
        async with client.stream(method, url, timeout=request_timeout, **kwargs) as response:
            # Redirect and retry responses only need their status and headers. Avoid
            # reading an attacker-controlled body before the next policy decision.
            if _is_redirect_response(response) or response.status_code == 429 or response.status_code >= 500:
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    request=getattr(response, "request", None) or httpx.Request(method, url),
                )

            _validate_response_headers(response, max_bytes, allowed_content_types, buffered=False)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if max_bytes is not None and len(body) + len(chunk) > max_bytes:
                    raise ValueError("response too large")
                body.extend(chunk)
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=getattr(response, "request", None) or httpx.Request(method, url),
            )

    for attempt in range(retries + 1):
        try:
            response = await perform_request()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt >= retries:
                raise
            await asyncio.sleep(min(max_backoff, backoff * (2 ** attempt)))
            continue

        status_code = getattr(response, "status_code", 200)
        if status_code == 429 or 500 <= status_code <= 599:
            if attempt < retries:
                delay = _retry_after(response, min(max_backoff, backoff * (2 ** attempt)))
                await asyncio.sleep(min(max_backoff, delay))
                continue
            if status_code == 429:
                raise ProviderRateLimitError(response)
        return response

    raise RuntimeError("request retry loop exhausted")


def _is_redirect_response(response: httpx.Response) -> bool:
    return 300 <= getattr(response, "status_code", 0) < 400


def _validate_response_headers(
    response: httpx.Response,
    max_bytes: int | None,
    allowed_content_types: frozenset[str] | None,
    *,
    buffered: bool,
) -> None:
    if max_bytes is not None:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise ValueError("invalid response length") from exc
            if declared_length < 0 or declared_length > max_bytes:
                raise ValueError("response too large")

        # Test doubles may expose a fully buffered body even when no streaming API
        # is available. Retain the same limit on that compatibility path.
        if buffered:
            content = getattr(response, "content", b"")
            if len(content) > max_bytes:
                raise ValueError("response too large")

    if allowed_content_types is not None:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in allowed_content_types:
            raise ValueError(f"unsupported response content type: {content_type or 'missing'}")


@dataclass
class Result:
    """A normalized search result across all providers."""

    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.0
    published: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "published": self.published,
            **({"extra": self.extra} if self.extra else {}),
        }


class FetchResult(TypedDict, total=False):
    """Normalized fetch response returned by every fetch provider.

    Fetch providers continue to return ordinary dictionaries at runtime for
    compatibility with the existing tool layer; this TypedDict makes the
    internal contract explicit without introducing a serialization change.
    """

    url: str
    title: str
    content: str
    truncated: bool
    length: int
    error: str


class Capability(str, Enum):
    """Operations and provider domains exposed by the registry."""

    SEARCH = "search"
    FETCH = "fetch"
    ACADEMIC = "academic"
    NEWS = "news"
    COMMUNITY = "community"
    GENERAL_SEARCH = "general_search"


SearchFunction = Callable[..., Awaitable[list[Result]]]
FetchFunction = Callable[..., Awaitable[FetchResult]]


@dataclass(frozen=True)
class ProviderConfig:
    """Discoverable configuration and availability requirements for a provider."""

    name: str
    capabilities: frozenset[Capability]
    description: str = ""
    api_key_env: str | None = None
    requires_api_key: bool = False

    def __post_init__(self) -> None:
        if self.requires_api_key and not self.api_key_env:
            raise ValueError(f"Provider {self.name!r} requires an API key environment variable")

    @property
    def configured(self) -> bool:
        """Whether the required configuration is present in the environment."""
        return not self.requires_api_key or bool(os.environ.get(self.api_key_env or ""))

    @property
    def available(self) -> bool:
        """Whether this provider can be selected with the current configuration."""
        return self.configured

    def to_dict(self) -> dict[str, Any]:
        """Return safe metadata suitable for diagnostics or discovery output."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": sorted(capability.value for capability in self.capabilities),
            "api_key_env": self.api_key_env,
            "requires_api_key": self.requires_api_key,
            "configured": self.configured,
            "available": self.available,
        }


class ProviderMetadata(Protocol):
    """Metadata shared by all capability-specific providers."""

    name: str
    capabilities: frozenset[Capability]
    configuration: ProviderConfig


class SearchProvider(ProviderMetadata, Protocol):
    """Typed contract for providers that support search."""

    async def search(
        self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any
    ) -> list[Result]: ...


class GeneralSearchProvider(SearchProvider, Protocol):
    """Typed contract for providers used by the general web-search tool."""


class AcademicProvider(SearchProvider, Protocol):
    """Typed contract for academic search providers."""


class NewsProvider(SearchProvider, Protocol):
    """Typed contract for news search providers."""


class CommunityProvider(SearchProvider, Protocol):
    """Typed contract for community search providers."""


class FetchProvider(ProviderMetadata, Protocol):
    """Typed contract for providers that support fetching."""

    async def fetch(self, url: str, client: httpx.AsyncClient) -> FetchResult: ...


Provider = SearchProvider | FetchProvider


@dataclass(frozen=True)
class _FunctionProvider:
    """Adapter that gives existing provider functions the typed provider shape."""

    name: str
    capabilities: frozenset[Capability]
    search_fn: SearchFunction | None = None
    fetch_fn: FetchFunction | None = None
    config: ProviderConfig | None = None

    @property
    def configuration(self) -> ProviderConfig:
        return self.config or ProviderConfig(name=self.name, capabilities=self.capabilities)

    async def search(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        if self.search_fn is None:
            raise TypeError(f"Provider {self.name!r} does not support search")
        # Resolve by name so existing monkeypatches and compatibility hooks that
        # replace a legacy provider function remain effective through the registry.
        search_fn = getattr(globals().get(self.search_fn.__name__), "__call__", None)
        if search_fn is None:
            search_fn = self.search_fn
        return await search_fn(query, max_results, client, **kwargs)

    async def fetch(self, url: str, client: httpx.AsyncClient) -> FetchResult:
        if self.fetch_fn is None:
            raise TypeError(f"Provider {self.name!r} does not support fetch")
        return await self.fetch_fn(url, client)


class ProviderRegistry:
    """Registry for selecting providers by stable name and capability."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._configs: dict[str, ProviderConfig] = {}

    def register(self, provider: Provider, config: ProviderConfig | None = None) -> Provider:
        if provider.name in self._providers:
            raise ValueError(f"Provider {provider.name!r} is already registered")
        config = config or getattr(provider, "configuration", None)
        if config is None:
            config = ProviderConfig(name=provider.name, capabilities=provider.capabilities)
        if config.name != provider.name:
            raise ValueError(f"Provider config name {config.name!r} does not match {provider.name!r}")
        if config.capabilities != provider.capabilities:
            raise ValueError(f"Provider config capabilities do not match {provider.name!r}")
        self._providers[provider.name] = provider
        self._configs[provider.name] = config
        return provider

    @overload
    def get(self, name: str, capability: Literal[Capability.SEARCH]) -> SearchProvider: ...

    @overload
    def get(self, name: str, capability: Literal[Capability.FETCH]) -> FetchProvider: ...

    @overload
    def get(
        self,
        name: str,
        capability: Literal[
            Capability.SEARCH,
            Capability.ACADEMIC,
            Capability.NEWS,
            Capability.COMMUNITY,
            Capability.GENERAL_SEARCH,
        ],
    ) -> SearchProvider: ...

    @overload
    def get(self, name: str, capability: None = None) -> Provider: ...

    def get(self, name: str, capability: Capability | None = None) -> Provider:
        try:
            provider = self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name}") from exc
        if capability is not None and capability not in provider.capabilities:
            raise ValueError(f"Provider {name!r} does not support {capability.value}")
        return provider

    @overload
    def providers_for(self, capability: Literal[Capability.SEARCH]) -> tuple[SearchProvider, ...]: ...

    @overload
    def providers_for(self, capability: Literal[Capability.FETCH]) -> tuple[FetchProvider, ...]: ...

    @overload
    def providers_for(
        self,
        capability: Literal[
            Capability.SEARCH,
            Capability.ACADEMIC,
            Capability.NEWS,
            Capability.COMMUNITY,
            Capability.GENERAL_SEARCH,
        ],
        *,
        available_only: bool = False,
    ) -> tuple[SearchProvider, ...]: ...

    def providers_for(self, capability: Capability, *, available_only: bool = False) -> tuple[Provider, ...]:
        return tuple(
            provider
            for provider in self._providers.values()
            if capability in provider.capabilities
            and (not available_only or self._configs[provider.name].available)
        )

    def configuration(self, name: str) -> ProviderConfig:
        """Return the discoverable configuration for one registered provider."""
        try:
            return self._configs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name}") from exc

    def discover(
        self, capability: Capability | None = None, *, available_only: bool = False
    ) -> tuple[ProviderConfig, ...]:
        """List provider configuration and availability, optionally by capability."""
        configs = tuple(self._configs.values())
        if capability is not None:
            configs = tuple(config for config in configs if capability in config.capabilities)
        if available_only:
            configs = tuple(config for config in configs if config.available)
        return configs


# --------------------------------------------------------------------------------------
# Caching and bounded concurrency
#
# Every search/fetch call that goes through the provider registry (both plain
# search_* tools and the deep-research pipeline) is funneled through
# cached_search/cached_fetch below, so both get the same TTL cache and the
# same per-provider concurrency cap for free.
#
# Cache keys are built only from provider name + query/URL + non-secret
# kwargs (e.g. Stack Exchange's `site`) — provider functions read API keys
# from the environment directly, so no credential ever enters a cache key.
#
# This is a simple cache, not a request-coalescing one: two concurrent calls
# that both miss on the same key will both hit the network. That's an
# accepted tradeoff for keeping this dependency-free and easy to reason about.
# --------------------------------------------------------------------------------------

_search_cache: TTLCache[list[Result]] = TTLCache(
    max_entries=_env_int("WEB_RESEARCH_CACHE_MAX_ENTRIES", 256, minimum=0),
    ttl_seconds=_env_float("WEB_RESEARCH_CACHE_TTL_SECONDS", 300.0, minimum=0.0),
)
_fetch_cache: TTLCache[FetchResult] = TTLCache(
    max_entries=_env_int("WEB_RESEARCH_CACHE_MAX_ENTRIES", 256, minimum=0),
    ttl_seconds=_env_float("WEB_RESEARCH_CACHE_TTL_SECONDS", 300.0, minimum=0.0),
)
_provider_semaphores: dict[str, asyncio.Semaphore] = {}


def reset_caches() -> None:
    """Clear cached search/fetch results. Primarily for test isolation."""
    _search_cache.clear()
    _fetch_cache.clear()


def _semaphore_for(provider: str) -> asyncio.Semaphore:
    """Return the per-provider semaphore bounding concurrent in-flight calls.

    Built lazily so the limit reflects WEB_RESEARCH_MAX_CONCURRENCY[_<PROVIDER>]
    at first use and stays stable afterward — mirrors the module's env-config
    pattern for timeouts/retries, but concurrency limits (unlike timeouts)
    need one persistent object for the life of the process.
    """
    sem = _provider_semaphores.get(provider)
    if sem is None:
        limit = _provider_int("WEB_RESEARCH_MAX_CONCURRENCY", provider, 4, minimum=1, maximum=32)
        sem = asyncio.Semaphore(limit)
        _provider_semaphores[provider] = sem
    return sem


def _search_cache_key(provider: str, query: str, max_results: int, kwargs: dict[str, Any]) -> str:
    extra = "&".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))
    return f"search|{provider}|{query}|{max_results}|{extra}"


def _fetch_cache_key(provider: str, url: str) -> str:
    return f"fetch|{provider}|{_canonical_url(url)}"


async def cached_search(
    provider: SearchProvider, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any
) -> list[Result]:
    """Cache-aware, concurrency-bounded wrapper around provider.search()."""
    key = _search_cache_key(provider.name, query, max_results, kwargs)
    cached = _search_cache.get(key)
    if cached is not None:
        observability.provider_cache_hit(provider.name, "search", query)
        return copy.deepcopy(cached)
    async with _semaphore_for(provider.name):
        results = await provider.search(query, max_results, client, **kwargs)
    # Providers degrade to [] on failure (rate limit, timeout, parse error) rather
    # than raising — caching that [] would silently suppress retries for the full
    # TTL even after the upstream recovers. Only cache genuine non-empty results.
    if results:
        _search_cache.set(key, copy.deepcopy(results))
    return results


async def cached_fetch(provider: FetchProvider, url: str, client: httpx.AsyncClient) -> FetchResult:
    """Cache-aware, concurrency-bounded wrapper around provider.fetch()."""
    key = _fetch_cache_key(provider.name, url)
    cached = _fetch_cache.get(key)
    if cached is not None:
        observability.provider_cache_hit(provider.name, "fetch", url)
        return copy.deepcopy(cached)
    async with _semaphore_for(provider.name):
        result = await provider.fetch(url, client)
    if not result.get("error"):
        _fetch_cache.set(key, copy.deepcopy(result))
    return result


# --------------------------------------------------------------------------------------
# Brave Search (optional, requires BRAVE_API_KEY — best general web index)
# --------------------------------------------------------------------------------------

async def search_brave(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    try:
        r = await _request(client, "GET",
            "https://api.search.brave.com/res/v1/web/search",
            provider="brave",
            params={"q": query, "count": min(max_results, 20)},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("brave", "search", e)
        return []

    out: list[Result] = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        out.append(
            Result(
                title=item.get("title", "").strip(),
                url=item.get("url", "").strip(),
                snippet=item.get("description", "").strip(),
                source="brave",
                score=float(data.get("web", {}).get("results", []).index(item) + 1),
                extra={"age": item.get("age")},
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Tavily (optional, requires TAVILY_API_KEY — research-optimized, returns content)
# --------------------------------------------------------------------------------------

async def search_tavily(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = await _request(client, "POST",
            "https://api.tavily.com/search",
            provider="tavily",
            json={
                "api_key": key,
                "query": query,
                "max_results": min(max_results, 10),
                "include_answer": False,
                "search_depth": "advanced",
            },
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("tavily", "search", e)
        return []

    out: list[Result] = []
    for item in data.get("results", [])[:max_results]:
        out.append(
            Result(
                title=item.get("title", "").strip(),
                url=item.get("url", "").strip(),
                snippet=item.get("content", "").strip()[:600],
                source="tavily",
                score=float(item.get("score", 0)),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Wikipedia (no key — high-quality encyclopedic source)
# --------------------------------------------------------------------------------------

async def search_wikipedia(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    try:
        r = await _request(client, "GET",
            "https://en.wikipedia.org/w/api.php",
            provider="wikipedia",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(max_results, 10),
                "format": "json",
                "utf8": 1,
                "origin": "*",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("wikipedia", "search", e)
        return []

    out: list[Result] = []
    for item in data.get("query", {}).get("search", [])[:max_results]:
        title = item.get("title", "").strip()
        # Wikipedia search snippets contain HTML; strip it
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
        # Build the canonical page URL
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        out.append(
            Result(
                title=title,
                url=url,
                snippet=snippet,
                source="wikipedia",
                score=float(item.get("score", 0)) / 100.0,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# arXiv (no key — academic preprints)
# --------------------------------------------------------------------------------------

async def search_arxiv(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    try:
        r = await _request(client, "GET",
            "https://export.arxiv.org/api/query",
            provider="arxiv",
            params={
                "search_query": f"all:{query}",
                "max_results": min(max_results, 10),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=20.0,
        )
        r.raise_for_status()
        # Atom XML — robust parse with namespace handling
        root = ET.fromstring(r.text)
    except Exception as e:
        observability.provider_failed("arxiv", "search", e)
        return []

    out: list[Result] = []
    for entry in root.findall("a:entry", ATOM_NS)[:max_results]:
        title_el = entry.find("a:title", ATOM_NS)
        summary_el = entry.find("a:summary", ATOM_NS)
        id_el = entry.find("a:id", ATOM_NS)
        published_el = entry.find("a:published", ATOM_NS)
        title = (title_el.text or "").strip().replace("\n", " ").replace("  ", " ") if title_el is not None else ""
        summary = (summary_el.text or "").strip().replace("\n", " ").replace("  ", " ") if summary_el is not None else ""
        url = (id_el.text or "").strip() if id_el is not None else ""
        published = (published_el.text or "").strip()[:10] if published_el is not None else None
        if not title or not url:
            continue
        out.append(
            Result(
                title=title,
                url=url,
                snippet=summary[:600],
                source="arxiv",
                published=published,
                extra={"authors": [a.find("a:name", ATOM_NS).text for a in entry.findall("a:author", ATOM_NS) if a.find("a:name", ATOM_NS) is not None]},
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Hacker News Algolia (no key — tech/news discussions, very high signal)
# --------------------------------------------------------------------------------------

async def search_hn(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    try:
        r = await _request(client, "GET",
            "https://hn.algolia.com/api/v1/search",
            provider="hackernews",
            params={"query": query, "hitsPerPage": min(max_results, 20), "tags": "story"},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("hackernews", "search", e)
        return []

    out: list[Result] = []
    for hit in data.get("hits", [])[:max_results]:
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        # Prefer the HN comment count as a relevance proxy
        score = float(hit.get("points", 0)) or float(hit.get("num_comments", 0))
        out.append(
            Result(
                title=hit.get("title", "").strip() or hit.get("story_text", "")[:80],
                url=url,
                snippet=hit.get("_highlightResult", {}).get("comment_text", {}).get("value", "")
                       or hit.get("story_text", "")
                       or f"{hit.get('num_comments', 0)} comments on Hacker News",
                source="hackernews",
                score=score,
                published=hit.get("created_at", "")[:10] if hit.get("created_at") else None,
                extra={"points": hit.get("points"), "comments": hit.get("num_comments")},
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Stack Exchange (no key — high-quality Q&A across 180+ technical sites)
# --------------------------------------------------------------------------------------

async def search_stackexchange(query: str, max_results: int, client: httpx.AsyncClient, site: str = "stackoverflow") -> list[Result]:
    try:
        # Stack Exchange API 2.3 requires a `filter` param for body fields, but
        # the default `default` filter works fine and includes everything we need.
        # The custom safe-filter id we used previously was rejected (HTTP 400).
        r = await _request(client, "GET",
            "https://api.stackexchange.com/2.3/search/advanced",
            provider="stackexchange",
            params={
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": site,
                "pagesize": min(max_results, 10),
            },
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("stackexchange", "search", e)
        return []

    out: list[Result] = []
    for item in data.get("items", [])[:max_results]:
        out.append(
            Result(
                title=item.get("title", "").strip(),
                url=item.get("link", "").strip(),
                snippet=re.sub(r"<[^>]+>", "", item.get("excerpt", "")).strip(),
                source=f"stackexchange:{site}",
                score=float(item.get("score", 0)),
                extra={
                    "is_answered": item.get("is_answered"),
                    "answer_count": item.get("answer_count"),
                    "tags": item.get("tags", []),
                },
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Crossref (no key — scholarly metadata, DOIs, citations)
# --------------------------------------------------------------------------------------

async def search_crossref(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    try:
        r = await _request(client, "GET",
            "https://api.crossref.org/works",
            provider="crossref",
            params={"query": query, "rows": min(max_results, 10), "sort": "relevance"},
            headers={"User-Agent": USER_AGENT},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        observability.provider_failed("crossref", "search", e)
        return []

    out: list[Result] = []
    for item in data.get("message", {}).get("items", [])[:max_results]:
        title_list = item.get("title", [])
        title = (title_list[0] if title_list else "").strip()
        # DOI URL is canonical
        url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else "")
        abstract = re.sub(r"<[^>]+>", "", item.get("abstract", "")).strip()
        if abstract:
            snippet = abstract[:500]
        else:
            # Fallback to container title
            container = item.get("container-title", [""])[0] if item.get("container-title") else ""
            snippet = f"Published in: {container}" if container else ""
        published_parts = item.get("published-print", item.get("published-online", item.get("issued", {}))).get("date-parts", [[None]])[0]
        published = f"{published_parts[0]}-{published_parts[1]:02d}-{published_parts[2]:02d}" if published_parts and published_parts[0] else None
        out.append(
            Result(
                title=title,
                url=url,
                snippet=snippet,
                source="crossref",
                published=published,
                extra={"doi": item.get("DOI"), "type": item.get("type"), "citations": item.get("is-referenced-by-count", 0)},
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Fetch via Jina Reader (returns clean markdown — bypasses JS-only sites & bot walls)
# --------------------------------------------------------------------------------------

async def fetch_jina(url: str, client: httpx.AsyncClient) -> FetchResult:
    """Fetch any URL and return clean markdown. Jina handles JS rendering, anti-bot, and clean extraction."""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/markdown"}
    key = os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    fetch_timeout = min(_request_timeout("jina", FETCH_TIMEOUT_SECONDS), FETCH_TIMEOUT_SECONDS)

    async def fetch_with_policy() -> str:
        # Validate the user-controlled target before handing it to the indirect
        # reader, then validate the reader endpoint immediately before each hop.
        validate_url(url)
        current_url = jina_url
        for redirect_count in range(FETCH_MAX_REDIRECTS + 1):
            validate_url(current_url)
            r = await _request(
                client,
                "GET",
                current_url,
                provider="jina",
                headers=headers,
                timeout=fetch_timeout,
                follow_redirects=False,
                _stream_response=True,
                _max_bytes=FETCH_MAX_BYTES,
                _allowed_content_types=FETCH_CONTENT_TYPES,
                _timeout_override=fetch_timeout,
            )
            if r.is_redirect:
                location = r.headers.get("location")
                if not location:
                    raise UrlSafetyError("blocked redirect without a location")
                if redirect_count == FETCH_MAX_REDIRECTS:
                    raise UrlSafetyError("blocked redirect chain exceeding the limit")
                current_url = urljoin(current_url, location)
                continue
            r.raise_for_status()
            return r.text

        raise UrlSafetyError("blocked redirect chain")  # pragma: no cover

    try:
        content = await asyncio.wait_for(fetch_with_policy(), timeout=fetch_timeout)
    except asyncio.TimeoutError:
        return {"url": url, "error": f"fetch failed: fetch exceeded {FETCH_TIMEOUT_SECONDS:g}s timeout"}
    except Exception as e:
        return {"url": url, "error": f"fetch failed: {e}"}

    # Jina response includes a header block (Title:, URL Source:, etc.) then content
    # Split it out cleanly for the LLM
    parsed_url = url
    title = ""
    body = content
    if content.startswith("Title:"):
        lines = content.split("\n", 5)
        meta: dict[str, str] = {}
        body_start = 0
        for i, line in enumerate(lines):
            if ": " in line and not line.startswith(" "):
                k, _, v = line.partition(": ")
                meta[k.strip()] = v.strip()
                body_start = i + 1
            else:
                break
        title = meta.get("Title", "")
        parsed_url = meta.get("URL Source", parsed_url)
        try:
            validate_url(parsed_url)
        except UrlSafetyError as exc:
            return {"url": url, "error": f"fetch failed: {exc}"}
        # Skip the blank separator line if present
        if body_start < len(lines) and lines[body_start].strip() == "":
            body_start += 1
        body = "\n".join(lines[body_start:])

    # Truncate very long pages to avoid blowing context windows (default 20k chars ≈ 5k tokens)
    max_chars = 20_000
    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars] + f"\n\n[...truncated, full content was {len(content):,} chars]"

    return {
        "url": parsed_url,
        "title": title,
        "content": body,
        "truncated": truncated,
        "length": len(body),
    }


def _build_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    for name, capabilities, search_fn, description, api_key_env, requires_api_key in (
        ("brave", frozenset({Capability.SEARCH, Capability.GENERAL_SEARCH}), search_brave,
         "General web search via Brave", "BRAVE_API_KEY", True),
        ("tavily", frozenset({Capability.SEARCH, Capability.GENERAL_SEARCH}), search_tavily,
         "Research-optimized general web search via Tavily", "TAVILY_API_KEY", True),
        ("wikipedia", frozenset({Capability.SEARCH}), search_wikipedia,
         "Encyclopedic search via Wikipedia", None, False),
        ("arxiv", frozenset({Capability.SEARCH, Capability.ACADEMIC}), search_arxiv,
         "Academic preprint search via arXiv", None, False),
        ("hackernews", frozenset({Capability.SEARCH, Capability.NEWS}), search_hn,
         "Technology news and discussion search via Hacker News", None, False),
        ("stackexchange", frozenset({Capability.SEARCH, Capability.COMMUNITY}), search_stackexchange,
         "Community Q&A search via Stack Exchange", None, False),
        ("crossref", frozenset({Capability.SEARCH, Capability.ACADEMIC}), search_crossref,
         "Scholarly metadata search via Crossref", None, False),
    ):
        registry.register(
            _FunctionProvider(
                name,
                capabilities,
                search_fn=search_fn,
                config=ProviderConfig(
                    name=name,
                    capabilities=capabilities,
                    description=description,
                    api_key_env=api_key_env,
                    requires_api_key=requires_api_key,
                ),
            )
        )
    jina_capabilities = frozenset({Capability.FETCH})
    registry.register(
        _FunctionProvider(
            "jina",
            jina_capabilities,
            fetch_fn=fetch_jina,
            config=ProviderConfig(
                name="jina",
                capabilities=jina_capabilities,
                description="Clean page fetching via Jina Reader",
                api_key_env="JINA_API_KEY",
                requires_api_key=False,
            ),
        )
    )
    return registry


provider_registry = _build_provider_registry()


# --------------------------------------------------------------------------------------
# Deduplication & merging
# --------------------------------------------------------------------------------------

def _canonical_url(url: str) -> str:
    """Strip tracking params, fragments, and normalize for dedup."""
    try:
        p = urlparse(url)
        scheme = p.scheme.lower()
        hostname = (p.hostname or "").lower()
        if not hostname:
            return url
        try:
            port = p.port
        except ValueError:
            return url
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = hostname if port is None or default_port else f"{hostname}:{port}"
        # Drop common tracking params
        drop_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                       "fbclid", "gclid", "ref", "ref_src", "source"}
        qs = parse_qs(p.query)
        qs = {k: v for k, v in qs.items() if k.lower() not in drop_params}
        from urllib.parse import urlencode, urlunparse
        clean_query = urlencode(sorted(qs.items()), doseq=True)
        return urlunparse((scheme, netloc, p.path.rstrip("/"), "", clean_query, ""))
    except Exception:
        return url


def merge_results(*result_lists: list[Result], max_total: int = 30) -> list[Result]:
    """Merge results across sources, dedupe by canonical URL, prefer higher-scored entries."""
    seen: dict[str, Result] = {}
    for results in result_lists:
        for r in results:
            canon = _canonical_url(r.url)
            if canon in seen:
                # Keep the entry with the longer snippet (more informative)
                existing = seen[canon]
                if len(r.snippet) > len(existing.snippet):
                    existing.snippet = r.snippet
                # Append source tag if multi-source
                if r.source not in existing.extra.get("also_found_in", []):
                    existing.extra.setdefault("also_found_in", []).append(r.source)
                provenance = existing.extra.setdefault("provenance", [])
                if not any(item["source"] == r.source for item in provenance):
                    provenance.append({"source": r.source, "url": r.url, "score": r.score})
                    # Bump score only for independent cross-source agreement.
                    existing.score += r.score
            else:
                merged = Result(
                    title=r.title,
                    url=canon,
                    snippet=r.snippet,
                    source=r.source,
                    score=r.score,
                    published=r.published,
                    extra=dict(r.extra),
                )
                merged.extra["provenance"] = [{"source": r.source, "url": r.url, "score": r.score}]
                seen[canon] = merged

    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return merged[:max_total]
