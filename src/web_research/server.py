"""MCP server entrypoint — registers all research tools and handles stdio transport.

Uses the modern mcp.server.mcpserver.MCPServer API (MCP SDK 1.x+).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from . import providers


app = MCPServer(
    "web-research",
    instructions=(
        "High-quality web research tools. Use search_web for general queries, fetch_url "
        "to read any page, and the *_meta tools for vertical-specific sources "
        "(Wikipedia, arXiv, Hacker News, Stack Exchange, Crossref)."
    ),
)


# Shared HTTP client — module-level so it can be reused across tool calls if the
# SDK keeps the event loop alive. For stdio transport each call still pays setup.
async def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": providers.USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------

@app.tool(
    name="search_web",
    description=(
        "Multi-source web search. Aggregates results from Brave Search and/or Tavily "
        "(whichever has a key configured) with dedup and cross-source scoring. "
        "Returns title, URL, snippet, and source for each result. "
        "Use as the default 'search the web' tool."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_web(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=30, default=10)] = 10,
    pro_mode: Annotated[
        bool,
        Field(description="If true, also fetch top 3 results via Jina Reader and append their content snippets."),
    ] = False,
) -> str:
    async with await _new_client() as client:
        brave_res, tavily_res = await asyncio.gather(
            providers.search_brave(query, max_results, client),
            providers.search_tavily(query, max_results, client),
        )

    merged = providers.merge_results(brave_res, tavily_res, max_total=max_results * 2)

    if pro_mode and merged:
        async with await _new_client() as client:
            top = merged[:3]
            fetched = await asyncio.gather(
                *(providers.fetch_jina(r.url, client) for r in top),
                return_exceptions=True,
            )
        for r, fetched_result in zip(top, fetched):
            if isinstance(fetched_result, Exception) or fetched_result.get("error"):
                continue
            content = fetched_result.get("content", "")
            if content:
                r.snippet = content[:1500].strip()

    if not merged:
        return (
            "No web results. This is likely because no API key is configured — set "
            "BRAVE_API_KEY or TAVILY_API_KEY in your web-research.env. "
            "(Wikipedia, arXiv, HN, Stack Exchange, Crossref, and Jina fetching still work keyless.)"
        )

    return _format_results(query, merged, "search_web")


@app.tool(
    name="fetch_url",
    description=(
        "Fetch any URL and return clean markdown content. Uses Jina Reader which handles "
        "JS rendering and bot detection on your behalf, returning readable text. "
        "Ideal for reading articles, papers, docs, or blog posts."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def fetch_url(
    url: Annotated[str, Field(description="HTTP(S) URL to fetch")],
) -> str:
    async with await _new_client() as client:
        result = await providers.fetch_jina(url, client)
    if result.get("error"):
        return f"Error: {result['error']}"

    title = result.get("title") or url
    content = result.get("content", "")
    truncated = result.get("truncated", False)
    length = result.get("length", len(content))

    parts = [
        f"# {title}",
        f"**Source:** {result['url']}",
        f"**Length:** {length:,} chars" + (" (truncated)" if truncated else ""),
        "",
        "---",
        "",
        content,
    ]
    return "\n".join(parts)


@app.tool(
    name="search_wikipedia",
    description="Search Wikipedia. Returns titles, URLs, and snippets. No API key required.",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_wikipedia(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await providers.search_wikipedia(query, max_results, client)
    return _format_results(query, res, "wikipedia") if res else f"No Wikipedia results for: {query}"


@app.tool(
    name="search_academic",
    description=(
        "Search arXiv for academic preprints across all fields (CS, physics, math, bio, etc.). "
        "Returns title, authors, abstract snippet, and PDF URL. No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_academic(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await providers.search_arxiv(query, max_results, client)
    return _format_results(query, res, "arxiv") if res else f"No arXiv results for: {query}"


@app.tool(
    name="search_news",
    description=(
        "Search Hacker News for tech news, discussions, and trending links. "
        "Returns title, URL, points, comments, and dates. No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_news(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=20, default=10)] = 10,
) -> str:
    async with await _new_client() as client:
        res = await providers.search_hn(query, max_results, client)
    return _format_results(query, res, "hackernews") if res else f"No Hacker News results for: {query}"


@app.tool(
    name="search_stackexchange",
    description=(
        "Search Stack Exchange sites (default: Stack Overflow) for high-quality technical Q&A. "
        "Set the 'site' parameter to search other SE communities (e.g. 'serverfault', "
        "'superuser', 'askubuntu', 'math', 'tex'). No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_stackexchange(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
    site: Annotated[str, Field(description="Stack Exchange site slug (default: stackoverflow)")] = "stackoverflow",
) -> str:
    async with await _new_client() as client:
        res = await providers.search_stackexchange(query, max_results, client, site=site)
    return _format_results(query, res, f"stackexchange:{site}") if res else f"No Stack Exchange/{site} results for: {query}"


@app.tool(
    name="search_scholar_meta",
    description=(
        "Search Crossref for scholarly metadata across publishers (Elsevier, Springer, Wiley, "
        "IEEE, ACM, etc.). Returns DOI, citation count, publication date, and abstract. "
        "No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_scholar_meta(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await providers.search_crossref(query, max_results, client)
    return _format_results(query, res, "crossref") if res else f"No Crossref results for: {query}"


# --------------------------------------------------------------------------------------
# Formatting helper
# --------------------------------------------------------------------------------------

def _format_results(query: str, results: list[providers.Result], source_label: str) -> str:
    lines = [
        f"# Results for: {query}",
        f"**Source:** {source_label} | **Total:** {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.title}")
        lines.append(f"- **URL:** {r.url}")
        if r.published:
            lines.append(f"- **Published:** {r.published}")
        if r.score:
            lines.append(f"- **Score:** {r.score:.2f}")
        if r.extra.get("also_found_in"):
            lines.append(f"- **Also found in:** {', '.join(r.extra['also_found_in'])}")
        for k, v in r.extra.items():
            if k == "also_found_in" or not v:
                continue
            lines.append(f"- **{k}:** {v}")
        lines.append("")
        lines.append(f"  > {r.snippet}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

async def _run() -> None:
    has_brave = bool(os.environ.get("BRAVE_API_KEY"))
    has_tavily = bool(os.environ.get("TAVILY_API_KEY"))
    has_jina = bool(os.environ.get("JINA_API_KEY"))
    print(
        f"[web-research-mcp] starting — brave={has_brave} tavily={has_tavily} jina={has_jina}",
        file=sys.stderr, flush=True,
    )
    await app.run_stdio_async()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
