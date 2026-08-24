"""MCP server entrypoint — registers all research tools and handles stdio transport.

Uses the modern mcp.server.mcpserver.MCPServer API (MCP SDK 1.x+).
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import sys
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from . import deep_research, observability, providers, synthesis


app = MCPServer(
    "web-research",
    instructions=(
        "High-quality web research tools. Use search_web for general queries, fetch_url "
        "to read any page, and the *_meta tools for vertical-specific sources "
        "(Wikipedia, arXiv, Hacker News, Stack Exchange, Crossref)."
    ),
)


def _with_request_context(function):
    """Ensure all provider events emitted by one MCP tool share one id."""
    @functools.wraps(function)
    async def wrapped(*args, **kwargs):
        with observability.request_context():
            return await function(*args, **kwargs)
    return wrapped


# Shared HTTP client — module-level so it can be reused across tool calls if the
# SDK keeps the event loop alive. For stdio transport each call still pays setup.
async def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": providers.USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


async def _search_provider(
    name: str, query: str, max_results: int, client: httpx.AsyncClient, **kwargs: object
) -> list[providers.Result]:
    provider = providers.provider_registry.get(name, providers.Capability.SEARCH)
    return await _search_registered_provider(provider, query, max_results, client, **kwargs)


async def _search_registered_provider(
    provider: providers.SearchProvider,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
    failure_sink: list[str] | None = None,
    **kwargs: object,
) -> list[providers.Result]:
    started = observability.provider_started(provider.name, "search", query)
    try:
        results = await providers.cached_search(provider, query, max_results, client, **kwargs)
    except Exception as exc:
        observability.provider_failed(provider.name, "search", exc)
        observability.provider_finished(provider.name, "search", started, 0, partial=False, status="error")
        if failure_sink is not None:
            failure_sink.append(provider.name)
        raise
    status = observability.provider_finished(
        provider.name,
        "search",
        started,
        len(results),
        partial=0 < len(results) < max_results,
    )
    if failure_sink is not None and status in {"error", "rate_limited", "partial"}:
        failure_sink.append(provider.name)
    return results


async def _search_capability(
    capability: providers.Capability,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
    failure_sink: list[str] | None = None,
    **kwargs: object,
) -> list[providers.Result]:
    """Search every configured provider advertising a capability."""
    selected = providers.provider_registry.providers_for(capability, available_only=True)
    outcomes = await asyncio.gather(
        *(
            _search_registered_provider(
                provider, query, max_results, client, failure_sink=failure_sink, **kwargs
            )
            for provider in selected
        ),
        return_exceptions=True,
    )
    if failure_sink is not None:
        for provider, outcome in zip(selected, outcomes):
            if isinstance(outcome, Exception) and provider.name not in failure_sink:
                failure_sink.append(provider.name)
    return [result for outcome in outcomes if isinstance(outcome, list) for result in outcome]


async def _fetch_provider(url: str, client: httpx.AsyncClient) -> dict[str, object]:
    provider = providers.provider_registry.get("jina", providers.Capability.FETCH)
    started = observability.provider_started(provider.name, "fetch", url)
    try:
        result = await providers.cached_fetch(provider, url, client)
    except Exception as exc:
        observability.provider_failed(provider.name, "fetch", exc)
        observability.provider_finished(provider.name, "fetch", started, 0, partial=False, status="error")
        raise
    failed = bool(result.get("error"))
    observability.provider_finished(
        provider.name,
        "fetch",
        started,
        0 if failed else 1,
        partial=False,
        status="error" if failed else "ok",
    )
    return result


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
@_with_request_context
async def search_web(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=30, default=10)] = 10,
    pro_mode: Annotated[
        bool,
        Field(description="If true, also fetch top 3 results via Jina Reader and append their content snippets."),
    ] = False,
) -> str:
    failed_sources: list[str] = []
    async with await _new_client() as client:
        results = await _search_capability(
            providers.Capability.GENERAL_SEARCH,
            query,
            max_results,
            client,
            failure_sink=failed_sources,
        )

    merged = providers.merge_results(results, max_total=max_results * 2)

    if pro_mode and merged:
        async with await _new_client() as client:
            top = merged[:3]
            fetched = await asyncio.gather(
                *(_fetch_provider(r.url, client) for r in top),
                return_exceptions=True,
            )
        for r, fetched_result in zip(top, fetched):
            if isinstance(fetched_result, Exception) or fetched_result.get("error"):
                continue
            content = fetched_result.get("content", "")
            if content:
                r.snippet = content[:1500].strip()

    if not merged:
        if failed_sources:
            unavailable = ", ".join(dict.fromkeys(failed_sources))
            return f"No web results. Partial failure: unavailable providers: {unavailable}."
        return (
            "No web results. This is likely because no API key is configured — set "
            "BRAVE_API_KEY or TAVILY_API_KEY in your web-research.env. "
            "(Wikipedia, arXiv, HN, Stack Exchange, Crossref, and Jina fetching still work keyless.)"
        )

    formatted = _format_results(query, merged, "search_web")
    if failed_sources:
        unavailable = ", ".join(dict.fromkeys(failed_sources))
        formatted += f"\n**Provider status:** Partial results; unavailable providers: {unavailable}."
    return formatted


@app.tool(
    name="fetch_url",
    description=(
        "Fetch any URL and return clean markdown content. Uses Jina Reader which handles "
        "JS rendering and bot detection on your behalf, returning readable text. "
        "Ideal for reading articles, papers, docs, or blog posts."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
@_with_request_context
async def fetch_url(
    url: Annotated[str, Field(description="HTTP(S) URL to fetch")],
) -> str:
    async with await _new_client() as client:
        result = await _fetch_provider(url, client)
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
@_with_request_context
async def search_wikipedia(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await _search_provider("wikipedia", query, max_results, client)
    return _format_results(query, res, "wikipedia") if res else f"No Wikipedia results for: {query}"


@app.tool(
    name="search_academic",
    description=(
        "Search arXiv for academic preprints across all fields (CS, physics, math, bio, etc.). "
        "Returns title, authors, abstract snippet, and PDF URL. No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
@_with_request_context
async def search_academic(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await _search_provider("arxiv", query, max_results, client)
    return _format_results(query, res, "arxiv") if res else f"No arXiv results for: {query}"


@app.tool(
    name="search_news",
    description=(
        "Search Hacker News for tech news, discussions, and trending links. "
        "Returns title, URL, points, comments, and dates. No API key required."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
@_with_request_context
async def search_news(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=20, default=10)] = 10,
) -> str:
    async with await _new_client() as client:
        res = await _search_provider("hackernews", query, max_results, client)
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
@_with_request_context
async def search_stackexchange(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
    site: Annotated[str, Field(description="Stack Exchange site slug (default: stackoverflow)")] = "stackoverflow",
) -> str:
    async with await _new_client() as client:
        res = await _search_provider("stackexchange", query, max_results, client, site=site)
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
@_with_request_context
async def search_scholar_meta(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        res = await _search_provider("crossref", query, max_results, client)
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
@_with_request_context
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
@_with_request_context
async def extract_evidence(
    url: Annotated[str, Field(description="HTTP(S) URL to read")],
    question: Annotated[str, Field(description="What you're looking for on this page")],
    max_passages: Annotated[int, Field(ge=1, le=10, default=5)] = 5,
) -> str:
    async with await _new_client() as client:
        fetched = await _fetch_provider(url, client)
        if fetched.get("error"):
            return f"Error fetching {url}: {fetched['error']}"
        evidence = deep_research.extract_evidence(
            content=fetched.get("content", ""),
            question=question,
            max_passages=max_passages,
        )
        canonical = providers._canonical_url(url)
        # Synthesize a single synthetic citation so callers can reference passages inline
        fake_citation = {
            "id": 1,
            "url": canonical,
            "title": fetched.get("title") or canonical,
            "source": "fetch",
        }
        payload = {
            "url": canonical,
            "title": fetched.get("title") or canonical,
            "length": fetched.get("length", 0),
            "truncated": fetched.get("truncated", False),
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
        "into sub-questions, runs iterative multi-query search per sub-question (a broad framing "
        "query plus targeted follow-ups on recent developments, evidence, and criticism) across "
        "multiple sources (Wikipedia, arXiv, Hacker News, Stack Exchange, Crossref, plus Brave/Tavily "
        "if keys are configured), suppresses near-duplicate sources across queries, fetches the top "
        "URLs for full-page extraction, and returns a structured ResearchReport containing: the plan, "
        "a numbered citation manifest with complete metadata (title, authors, publisher/venue, "
        "publication date, URL, access date), per-sub-question evidence passages each stamped with "
        "its source's citation_id + character offset, and a Markdown synthesis template for you to "
        "fill in. You (the model) should write the narrative synthesis citing the [n] markers; the "
        "server does the gathering, not the writing. Use `depth='quick'` for fast overviews, "
        "'standard' for normal research, 'deep' for thorough multi-source investigations."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
@_with_request_context
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
# Long-form synthesis tools
# --------------------------------------------------------------------------------------

@app.tool(
    name="synthesize_report",
    description=(
        "Turn a structured research payload (the JSON returned by the `research` tool) into the "
        "six-section long-form report scaffold defined by the research-report.v1 agent contract: "
        "Executive Summary; Concept Deep-Dive (foundational→frontier explanation ladder); Survey of "
        "Perspectives (steelmannable opposing positions with attribution and an explicit "
        "disagreement summary, CONFLICTING SOURCES flag where detected); Cutting-Edge Developments "
        "(last ~24 months, preprint status flagged explicitly); Annotated Bibliography (relevance / "
        "credibility / date on 100% of references); Source Register (number → entry). Returns JSON "
        "plus a ready-to-fill Markdown skeleton with writing guardrails. You (the model) write the "
        "narrative prose citing [n] markers; this tool enforces the structure."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
@_with_request_context
async def synthesize_report(
    research_payload: Annotated[str, Field(description="The JSON output of the `research` tool")],
) -> str:
    try:
        payload = json.loads(research_payload)
    except json.JSONDecodeError as e:
        return f"Error: research_payload is not valid JSON: {e}"
    if not isinstance(payload, dict) or "citations" not in payload:
        return "Error: research_payload must be the JSON object returned by the `research` tool."
    scaffold = synthesis.build_longform_scaffold(payload)
    markdown = synthesis.render_markdown(scaffold)
    header = (
        f"# Synthesis scaffold ready ({len(scaffold['sections'])} sections, "
        f"{len(scaffold['annotated_bibliography'])} annotated references)\n\n"
        "Write the narrative per the guardrails; every substantive claim needs an [n] marker.\n"
        "Validate your draft with the `audit_citations` tool before finalizing.\n\n---\n\n"
    )
    return header + markdown + "\n\n---\n\n```json\n" + json.dumps(scaffold, indent=2) + "\n```"


@app.tool(
    name="audit_citations",
    description=(
        "Audit a drafted report for citation discipline before delivery. Flags every substantive "
        "sentence lacking an inline [n] reference (with line numbers), rejects references to "
        "bibliography entries that don't exist when a known-reference list is supplied, and reports "
        "coverage stats (total inline markers, distinct references used). Use it as the final "
        "quality gate for the zero-uncited-claims acceptance rule."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
@_with_request_context
async def audit_citations_tool(
    draft: Annotated[str, Field(description="The drafted report text (Markdown)")],
    known_refs: Annotated[
        str,
        Field(description="Optional comma-separated list of valid bibliography numbers, e.g. '1,2,3'"),
    ] = "",
) -> str:
    refs: set[int] | None = None
    if known_refs.strip():
        try:
            refs = {int(part) for part in re.split(r"[,\s]+", known_refs.strip()) if part}
        except ValueError:
            return "Error: known_refs must be a comma-separated list of integers."
    result = synthesis.audit_citations(draft, known_refs=refs)
    n_uncited = len(result["uncited_claims"])
    verdict = "PASS" if n_uncited == 0 and not result["unknown_refs"] else "FAIL"
    lines = [
        f"# Citation audit: {verdict}",
        f"- Substantive sentences without [n]: {n_uncited}",
        f"- Unknown reference numbers: {result['unknown_refs'] or 'none'}",
        f"- Inline markers total: {result['total_inline_markers']}",
        f"- Distinct references used: {sorted(result['distinct_refs_used'])}",
        "",
    ]
    for line_no, snippet in result["uncited_claims"]:
        lines.append(f"  - line {line_no}: {snippet}")
    return "\n".join(lines)


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
