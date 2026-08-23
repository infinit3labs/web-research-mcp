"""MCP server entrypoint — registers all research tools and handles stdio transport.

Uses the modern mcp.server.mcpserver.MCPServer API (MCP SDK 1.x+).

All search/fetch tools go through ``providers.REGISTRY``: each tool looks up
the providers it needs by name and calls ``.search(query, max_results, client)``
or ``.fetch(url, client)`` on them. Providers are responsible for
availability checks, retries, and SSRF safety; tool handlers just dispatch
and aggregate. A single broken provider (or one with no API key configured)
returns a `ProviderOutcome` with an error or `unavailable=True` flag — it
never raises, so a research run survives individual source failures.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from . import deep_research, providers
from .providers import REGISTRY, FetchResult, ProviderOutcome, Result


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
    return providers.new_http_client()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


async def _search_by_name(
    name: str,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
    **kwargs,
) -> list[Result]:
    """Run a single registered search provider; return its results or [].

    A missing/unavailable/erroring provider contributes zero results without
    raising — that's the whole point of the registry. Per-provider stderr
    logging already happened inside ``provider.search``.
    """
    try:
        provider = REGISTRY.get(name)
    except KeyError:
        print(f"[web-research] no provider registered as {name!r}", file=sys.stderr, flush=True)
        return []
    outcome: ProviderOutcome = await provider.search(query, max_results, client, **kwargs)
    return list(outcome.results)


async def _fetch_by_name(
    name: str,
    url: str,
    client: httpx.AsyncClient,
) -> FetchResult:
    """Run a single registered fetch provider; return its FetchResult."""
    try:
        provider = REGISTRY.get(name)
    except KeyError as e:
        return FetchResult(url=url, error=f"no provider registered as {name!r}: {e}")
    return await provider.fetch(url, client)


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
            _search_by_name("brave", query, max_results, client),
            _search_by_name("tavily", query, max_results, client),
        )

    merged = providers.merge_results(brave_res, tavily_res, max_total=max_results * 2)

    if pro_mode and merged:
        async with await _new_client() as client:
            top = merged[:3]
            fetched = await asyncio.gather(
                *(_fetch_by_name("jina", r.url, client) for r in top),
                return_exceptions=True,
            )
        for r, fetched_result in zip(top, fetched):
            if isinstance(fetched_result, BaseException) or fetched_result.error:
                continue
            content = fetched_result.content
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
        result = await _fetch_by_name("jina", url, client)
    if result.error:
        return f"Error: {result.error}"

    title = result.title or url
    content = result.content
    truncated = result.truncated
    length = result.length

    parts = [
        f"# {title}",
        f"**Source:** {result.url}",
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
        res = await _search_by_name("wikipedia", query, max_results, client)
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
        res = await _search_by_name("arxiv", query, max_results, client)
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
        res = await _search_by_name("hackernews", query, max_results, client)
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
        res = await _search_by_name("stackexchange", query, max_results, client, site=site)
    return _format_results(query, res, f"stackexchange:{site}") if res else f"No Stack Exchange/{site} results for: {query}"


@app.tool(
    name="search_scholar_meta",
    description=(
        "Search Crossref for scholarly metadata across publishers (Elsevier, Springer, Wiley, "
        "IEEE, ACM, etc.). Returns DOI, citation count, publication date, and abstract. "
        "Covers peer-reviewed papers that arXiv may not. No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search_scholar_meta(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await _search_by_name("crossref", query, max_results, client)
    return _format_results(query, res, "crossref") if res else f"No Crossref results for: {query}"


# --------------------------------------------------------------------------------------
# Deep research tools
# --------------------------------------------------------------------------------------


@app.tool(
    name="plan_research",
    description=(
        "Build a structured research plan for a complex question without executing it. "
        "Returns the planned sub-questions, the sources recommended for each, and "
        "estimated cost (searches + fetches). Use this when you want to review or "
        "edit the plan before committing to the full deep-research pipeline. "
        "Returns JSON; pass it back to the `research` tool to execute."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
async def plan_research(
    question: Annotated[str, Field(description="The research question")],
    depth: Annotated[
        str,
        Field(description="Plan breadth: 'quick' (2-3 sub-questions), 'standard' (4-6), or 'deep' (6-8)."),
    ] = "standard",
) -> str:
    try:
        plan = deep_research.build_plan(question, depth)
    except ValueError as e:
        return f"Error: {e}"
    return json.dumps(plan.to_dict(), indent=2)


@app.tool(
    name="extract_evidence",
    description=(
        "Fetch a URL and extract the passages most relevant to a specific question. "
        "Returns each passage with a short context window before/after, a relevance "
        "score, and a character offset into the original page so citations are "
        "verifiable. Use this when you want to drill into a specific source for "
        "evidence on a narrow claim."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def extract_evidence(
    url: Annotated[str, Field(description="HTTP(S) URL to read")],
    question: Annotated[str, Field(description="What you're looking for on this page")],
    max_passages: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        fetched = await _fetch_by_name("jina", url, client)
        if fetched.error:
            return f"Error fetching {url}: {fetched.error}"
        evidence = deep_research.extract_evidence(
            content=fetched.content,
            question=question,
            max_passages=max_passages,
        )
        canonical = providers._canonical_url(url)
        payload = {
            "url": canonical,
            "title": fetched.title or canonical,
            "length": fetched.length,
            "truncated": fetched.truncated,
            "passages": [
                {
                    "relevance": round(e.relevance, 3),
                    "offset": e.char_offset,
                    "before": e.context_before,
                    "quote": e.quote,
                    "after": e.context_after,
                }
                for e in evidence
            ],
        }
    return json.dumps(payload, indent=2)


@app.tool(
    name="research",
    description=(
        "Run a full deep-research pipeline on a complex question. Decomposes the question "
        "into sub-questions, fans out across multiple sources (Wikipedia, arXiv, Hacker News, "
        "Stack Exchange, Crossref, plus Brave/Tavily if keys are configured), fetches the top "
        "URLs, extracts the most relevant passages, and returns a structured ResearchReport "
        "containing: the plan, a numbered citation manifest, per-sub-question evidence "
        "with quotes + character offsets, and a Markdown synthesis template for you to fill in. "
        "You (the model) should write the narrative synthesis citing the [n] markers; the "
        "server does the gathering, not the writing. Use `depth='quick'` for fast overviews, "
        "'standard' for normal research, 'deep' for thorough multi-source investigations."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def research(
    question: Annotated[str, Field(description="The research question to investigate")],
    depth: Annotated[
        str,
        Field(description="Research depth: 'quick' (2-3 sub-questions, ~10 fetches), 'standard' (4-6, ~20 fetches), or 'deep' (6-8, ~32 fetches)."),
    ] = "standard",
) -> str:
    try:
        report = await deep_research.run_research(question, depth)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Research pipeline failed: {type(e).__name__}: {e}"
    d = report.to_dict()
    # Return a two-part response: a short status header for the LLM to parse
    # quickly, then the full structured data as JSON for machine inspection.
    header = (
        f"# Deep Research Complete: {question}\n\n"
        f"**Depth:** {depth}\n"
        f"**Sub-questions answered:** {len(d['plan']['sub_questions'])}\n"
        f"**Sources gathered:** {len(d['citations'])}\n"
        f"**Evidence passages:** {sum(len(v) for v in d['evidence'].values())}\n\n"
        f"Below this line is the structured report (JSON). Use the `synthesis_template` "
        f"to draft a cited answer; use the `evidence` field to find the exact quotes "
        f"supporting each sub-question; use the `citations` field to build the Sources "
        f"table — each `[n]` citation_id in your answer must reference a row there.\n\n"
        f"---\n\n"
    )
    return header + json.dumps(d, indent=2)


# --------------------------------------------------------------------------------------
# Formatting helper
# --------------------------------------------------------------------------------------

def _format_results(query: str, results: list[Result], source_label: str) -> str:
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
