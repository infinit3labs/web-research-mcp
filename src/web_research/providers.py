"""Multi-source search providers.

Each provider exposes a single async function: ``async def search(query, max_results, client) -> list[Result]``

Providers gracefully degrade on failure (return [], log to stderr) so a single
broken source never sinks the whole research run.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx


USER_AGENT = "research-agent/0.1 (+https://github.com/local/web-research-mcp; mailto:research@local)"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


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


# --------------------------------------------------------------------------------------
# Brave Search (optional, requires BRAVE_API_KEY — best general web index)
# --------------------------------------------------------------------------------------

async def search_brave(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    try:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(max_results, 20)},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[brave] error: {e}", flush=True)
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
        r = await client.post(
            "https://api.tavily.com/search",
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
        print(f"[tavily] error: {e}", flush=True)
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
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
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
        print(f"[wikipedia] error: {e}", flush=True)
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
        r = await client.get(
            "https://export.arxiv.org/api/query",
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
        print(f"[arxiv] error: {e}", flush=True)
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
        r = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "hitsPerPage": min(max_results, 20), "tags": "story"},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[hn] error: {e}", flush=True)
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
        r = await client.get(
            "https://api.stackexchange.com/2.3/search/advanced",
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
        print(f"[stackexchange] error: {e}", flush=True)
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
        r = await client.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": min(max_results, 10), "sort": "relevance"},
            headers={"User-Agent": USER_AGENT},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[crossref] error: {e}", flush=True)
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

async def fetch_jina(url: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch any URL and return clean markdown. Jina handles JS rendering, anti-bot, and clean extraction."""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/markdown"}
    key = os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        r = await client.get(jina_url, headers=headers, timeout=45.0, follow_redirects=True)
        r.raise_for_status()
        content = r.text
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
