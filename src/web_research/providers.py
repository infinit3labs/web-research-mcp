"""Multi-source search/fetch providers.

Each provider is a small class implementing either `BaseSearchProvider`
(``async def _search_impl(query, max_results, client, **kwargs) -> list[Result]``)
or `BaseFetchProvider` (``async def _fetch_impl(url, client) -> FetchResult``).
The base classes handle availability checks (API-key presence) and turn any
failure into a typed outcome (`ProviderOutcome` / `FetchResult.error`) instead
of a raised exception, so a single broken source never sinks a whole research
run. HTTP calls go through `net.request_with_retry`, which applies bounded
retries with backoff on timeouts, connection errors, 429s, and 5xxs.

Providers are registered by name/capability in the module-level `REGISTRY`
(`ProviderRegistry`); tool handlers look providers up by capability rather
than importing/calling each one directly, so adding a new source only means
registering it here.
"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from . import net


USER_AGENT = "research-agent/0.1 (+https://github.com/local/web-research-mcp; mailto:research@local)"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Global retry defaults, overridable without touching code.
DEFAULT_MAX_RETRIES = _env_int("WEB_RESEARCH_MAX_RETRIES", 2)
DEFAULT_BACKOFF_BASE = _env_float("WEB_RESEARCH_BACKOFF_BASE", 0.5)


def new_http_client() -> httpx.AsyncClient:
    """Shared client config used by both the MCP server and the deep-research pipeline."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


# --------------------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------------------

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


@dataclass
class FetchResult:
    """Normalized output of a fetch provider (e.g. Jina Reader)."""

    url: str
    title: str = ""
    content: str = ""
    truncated: bool = False
    length: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "truncated": self.truncated,
            "length": self.length,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ProviderOutcome:
    """Result of calling a search provider — always returned, never raised.

    `unavailable` means the provider was skipped because required
    configuration (an API key) is missing — expected and not worth alarming
    callers about. `error` means the provider was attempted and failed;
    `rate_limited` further flags that the failure was specifically a 429.
    """

    provider: str
    results: list[Result] = field(default_factory=list)
    error: str | None = None
    rate_limited: bool = False
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------------------

class Capability(str, Enum):
    """What kind of source a provider is — used for registry lookup/dispatch."""

    WEB_SEARCH = "web_search"
    REFERENCE = "reference"
    ACADEMIC = "academic"
    NEWS = "news"
    COMMUNITY_QA = "community_qa"
    SCHOLARLY_META = "scholarly_meta"
    FETCH = "fetch"


@dataclass(frozen=True)
class ProviderInfo:
    """Static config/metadata for a provider — what the registry surfaces for discovery."""

    name: str
    capability: Capability
    api_key_env: str | None = None
    timeout: float = 15.0
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE


class BaseSearchProvider(ABC):
    """A provider that turns a query into ranked `Result`s."""

    info: ProviderInfo

    def is_available(self) -> bool:
        """True if this provider's required API key (if any) is configured."""
        return not self.info.api_key_env or bool(os.environ.get(self.info.api_key_env))

    async def search(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> ProviderOutcome:
        if not self.is_available():
            return ProviderOutcome(provider=self.info.name, unavailable=True)
        try:
            results = await self._search_impl(query, max_results, client, **kwargs)
            return ProviderOutcome(provider=self.info.name, results=results)
        except net.ProviderError as e:
            print(f"[{self.info.name}] error: {e}", file=sys.stderr, flush=True)
            return ProviderOutcome(provider=self.info.name, error=str(e), rate_limited=e.rate_limited)
        except Exception as e:
            print(f"[{self.info.name}] error: {e}", file=sys.stderr, flush=True)
            return ProviderOutcome(provider=self.info.name, error=f"{type(e).__name__}: {e}")

    @abstractmethod
    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        ...


class BaseFetchProvider(ABC):
    """A provider that turns a URL into a `FetchResult`."""

    info: ProviderInfo

    def is_available(self) -> bool:
        return not self.info.api_key_env or bool(os.environ.get(self.info.api_key_env))

    async def fetch(self, url: str, client: httpx.AsyncClient) -> FetchResult:
        if not self.is_available():
            return FetchResult(url=url, error="not configured (missing API key)")
        try:
            return await self._fetch_impl(url, client)
        except net.UnsafeURLError as e:
            return FetchResult(url=url, error=f"unsafe URL: {e}")
        except net.ProviderError as e:
            print(f"[{self.info.name}] error: {e}", file=sys.stderr, flush=True)
            return FetchResult(url=url, error=str(e))
        except Exception as e:
            print(f"[{self.info.name}] error: {e}", file=sys.stderr, flush=True)
            return FetchResult(url=url, error=f"{type(e).__name__}: {e}")

    @abstractmethod
    async def _fetch_impl(self, url: str, client: httpx.AsyncClient) -> FetchResult:
        ...


class ProviderRegistry:
    """Capability-based lookup for all registered providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseSearchProvider | BaseFetchProvider] = {}

    def register(self, provider: BaseSearchProvider | BaseFetchProvider) -> None:
        if provider.info.name in self._providers:
            raise ValueError(f"provider {provider.info.name!r} already registered")
        self._providers[provider.info.name] = provider

    def get(self, name: str) -> BaseSearchProvider | BaseFetchProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"no provider registered as {name!r}") from None

    def by_capability(self, capability: Capability) -> list[BaseSearchProvider | BaseFetchProvider]:
        return [p for p in self._providers.values() if p.info.capability is capability]

    def describe(self) -> list[dict[str, Any]]:
        """Discoverable config/availability for every registered provider."""
        return [
            {
                "name": p.info.name,
                "capability": p.info.capability.value,
                "requires_api_key": p.info.api_key_env,
                "available": p.is_available(),
            }
            for p in self._providers.values()
        ]

    def __iter__(self):
        return iter(self._providers.values())


# --------------------------------------------------------------------------------------
# Brave Search (optional, requires BRAVE_API_KEY — best general web index)
# --------------------------------------------------------------------------------------

class BraveProvider(BaseSearchProvider):
    info = ProviderInfo(name="brave", capability=Capability.WEB_SEARCH, api_key_env="BRAVE_API_KEY", timeout=15.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        key = os.environ["BRAVE_API_KEY"]
        response = await net.request_with_retry(
            client, "GET", "https://api.search.brave.com/res/v1/web/search",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            params={"q": query, "count": min(max_results, 20)},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
        data = response.json()

        out: list[Result] = []
        for i, item in enumerate(data.get("web", {}).get("results", [])[:max_results]):
            out.append(
                Result(
                    title=item.get("title", "").strip(),
                    url=item.get("url", "").strip(),
                    snippet=item.get("description", "").strip(),
                    source="brave",
                    score=float(i + 1),
                    extra={"age": item.get("age")},
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Tavily (optional, requires TAVILY_API_KEY — research-optimized, returns content)
# --------------------------------------------------------------------------------------

class TavilyProvider(BaseSearchProvider):
    info = ProviderInfo(name="tavily", capability=Capability.WEB_SEARCH, api_key_env="TAVILY_API_KEY", timeout=20.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        key = os.environ["TAVILY_API_KEY"]
        response = await net.request_with_retry(
            client, "POST", "https://api.tavily.com/search",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            json={
                "api_key": key,
                "query": query,
                "max_results": min(max_results, 10),
                "include_answer": False,
                "search_depth": "advanced",
            },
        )
        data = response.json()

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

class WikipediaProvider(BaseSearchProvider):
    info = ProviderInfo(name="wikipedia", capability=Capability.REFERENCE, timeout=15.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        response = await net.request_with_retry(
            client, "GET", "https://en.wikipedia.org/w/api.php",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
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
        )
        data = response.json()

        out: list[Result] = []
        for item in data.get("query", {}).get("search", [])[:max_results]:
            title = item.get("title", "").strip()
            # Wikipedia search snippets contain HTML; strip it
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
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

class ArxivProvider(BaseSearchProvider):
    info = ProviderInfo(name="arxiv", capability=Capability.ACADEMIC, timeout=20.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        response = await net.request_with_retry(
            client, "GET", "https://export.arxiv.org/api/query",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            params={
                "search_query": f"all:{query}",
                "max_results": min(max_results, 10),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        # Atom XML — robust parse with namespace handling
        root = ET.fromstring(response.text)

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

class HackerNewsProvider(BaseSearchProvider):
    info = ProviderInfo(name="hackernews", capability=Capability.NEWS, timeout=15.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        response = await net.request_with_retry(
            client, "GET", "https://hn.algolia.com/api/v1/search",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            params={"query": query, "hitsPerPage": min(max_results, 20), "tags": "story"},
        )
        data = response.json()

        out: list[Result] = []
        for hit in data.get("hits", [])[:max_results]:
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
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

class StackExchangeProvider(BaseSearchProvider):
    info = ProviderInfo(name="stackexchange", capability=Capability.COMMUNITY_QA, timeout=15.0)

    async def _search_impl(
        self, query: str, max_results: int, client: httpx.AsyncClient, site: str = "stackoverflow", **kwargs: Any
    ) -> list[Result]:
        # Stack Exchange API 2.3 requires a `filter` param for body fields, but
        # the default `default` filter works fine and includes everything we need.
        response = await net.request_with_retry(
            client, "GET", "https://api.stackexchange.com/2.3/search/advanced",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            params={
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": site,
                "pagesize": min(max_results, 10),
            },
        )
        data = response.json()

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

class CrossrefProvider(BaseSearchProvider):
    info = ProviderInfo(name="crossref", capability=Capability.SCHOLARLY_META, timeout=15.0)

    async def _search_impl(self, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: Any) -> list[Result]:
        response = await net.request_with_retry(
            client, "GET", "https://api.crossref.org/works",
            provider=self.info.name, timeout=self.info.timeout, max_retries=self.info.max_retries,
            backoff_base=self.info.backoff_base,
            params={"query": query, "rows": min(max_results, 10), "sort": "relevance"},
            headers={"User-Agent": USER_AGENT},
        )
        data = response.json()

        out: list[Result] = []
        for item in data.get("message", {}).get("items", [])[:max_results]:
            title_list = item.get("title", [])
            title = (title_list[0] if title_list else "").strip()
            url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else "")
            abstract = re.sub(r"<[^>]+>", "", item.get("abstract", "")).strip()
            if abstract:
                snippet = abstract[:500]
            else:
                container = item.get("container-title", [""])[0] if item.get("container-title") else ""
                snippet = f"Published in: {container}" if container else ""
            published_parts = item.get("published-print", item.get("published-online", item.get("issued", {}))).get("date-parts", [[None]])[0]
            # date-parts may be a year-only list like [2023] — tolerate missing month/day
            if published_parts and published_parts[0]:
                y, m, d = (published_parts + [None, None, None])[:3]
                published = (
                    f"{y:04d}-{m:02d}-{d:02d}"
                    if m is not None and d is not None
                    else f"{y:04d}"
                )
            else:
                published = None
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

class JinaFetchProvider(BaseFetchProvider):
    """Fetch any URL via Jina Reader, which handles JS rendering, anti-bot, and extraction.

    The target URL is validated against `net.validate_public_url` before it's
    ever embedded in the outbound request, and the response is streamed with
    a byte cap and a content-type check — defense in depth even though the
    actual network fetch happens on Jina's infrastructure, not ours.
    """

    info = ProviderInfo(name="jina", capability=Capability.FETCH, api_key_env=None, timeout=45.0, max_retries=1)

    async def _fetch_impl(self, url: str, client: httpx.AsyncClient) -> FetchResult:
        await net.validate_public_url(url)

        jina_url = f"https://r.jina.ai/{url}"
        headers = {"User-Agent": USER_AGENT, "Accept": "text/markdown"}
        key = os.environ.get("JINA_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        raw = await self._stream_with_retry(client, jina_url, headers)
        content = raw.decode("utf-8", errors="replace")

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
            if body_start < len(lines) and lines[body_start].strip() == "":
                body_start += 1
            body = "\n".join(lines[body_start:])

        # Truncate very long pages to avoid blowing context windows (default 20k chars ≈ 5k tokens)
        max_chars = 20_000
        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars] + f"\n\n[...truncated, full content was {len(content):,} chars]"

        return FetchResult(url=parsed_url, title=title, content=body, truncated=truncated, length=len(body))

    async def _stream_with_retry(self, client: httpx.AsyncClient, jina_url: str, headers: dict[str, str]) -> bytes:
        attempt = 0
        while True:
            try:
                async with client.stream(
                    "GET", jina_url, headers=headers, timeout=self.info.timeout,
                    follow_redirects=True, max_redirects=5,
                ) as response:
                    if response.status_code in net.RETRIABLE_STATUS_CODES:
                        if attempt >= self.info.max_retries:
                            raise net.ProviderError(
                                self.info.name, f"HTTP {response.status_code} after {attempt + 1} attempt(s)",
                                status_code=response.status_code, rate_limited=response.status_code == 429,
                            )
                        delay = net.parse_retry_after(response.headers.get("Retry-After"))
                        if delay is None:
                            delay = net.compute_backoff(attempt, self.info.backoff_base)
                        attempt += 1
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code >= 400:
                        raise net.ProviderError(
                            self.info.name, f"HTTP {response.status_code}", status_code=response.status_code,
                        )

                    content_type = response.headers.get("content-type", "")
                    if net.is_disallowed_content_type(content_type):
                        raise net.ProviderError(self.info.name, f"disallowed content-type: {content_type}")

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > net.MAX_RESPONSE_BYTES:
                            raise net.ProviderError(
                                self.info.name, f"response exceeded {net.MAX_RESPONSE_BYTES:,} byte limit",
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= self.info.max_retries:
                    raise net.ProviderError(self.info.name, f"{type(e).__name__}: {e}") from e
                await asyncio.sleep(net.compute_backoff(attempt, self.info.backoff_base))
                attempt += 1
                continue


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

REGISTRY = ProviderRegistry()
for _provider in (
    BraveProvider(),
    TavilyProvider(),
    WikipediaProvider(),
    ArxivProvider(),
    HackerNewsProvider(),
    StackExchangeProvider(),
    CrossrefProvider(),
    JinaFetchProvider(),
):
    REGISTRY.register(_provider)
del _provider


# --------------------------------------------------------------------------------------
# Deduplication & merging
# --------------------------------------------------------------------------------------

def _canonical_url(url: str) -> str:
    """Strip tracking params, fragments, and normalize for dedup."""
    try:
        p = urlparse(url)
        # Drop common tracking params
        drop_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                       "fbclid", "gclid", "ref", "ref_src", "source"}
        qs = parse_qs(p.query)
        qs = {k: v for k, v in qs.items() if k.lower() not in drop_params}
        from urllib.parse import urlencode, urlunparse
        clean_query = urlencode(qs, doseq=True)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", clean_query, ""))
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
                # Bump score for cross-source agreement
                existing.score += r.score
            else:
                r.url = canon
                seen[canon] = r

    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return merged[:max_total]
