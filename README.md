# Web Research MCP

> A high-quality, multi-source web research MCP server for AI agents. Plug it into Claude Desktop, Hermes, Cursor, or any MCP-compatible client and get production-grade search + page-fetching across Wikipedia, arXiv, Hacker News, Stack Exchange, Crossref, Brave, Tavily, and any URL on the web.

[![MCP](https://img.shields.io/badge/MCP-1.x-blue)](https://modelcontextprotocol.io) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![GitHub stars](https://img.shields.io/github/stars/infinit3labs/web-research-mcp?style=social)](https://github.com/infinit3labs/web-research-mcp/stargazers) [![CI](https://github.com/infinit3labs/web-research-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/infinit3labs/web-research-mcp/actions/workflows/ci.yml)

```bash
# One-line install (anywhere on disk)
git clone https://github.com/infinit3labs/web-research-mcp.git
hermes mcp add web-research --command "$(pwd)/web-research-mcp/bin/web-research-mcp"
# 9 of 10 tools work with zero API keys. Add Brave or Tavily to unlock general web search.
```

---

## Why this exists

Most "web search" MCP servers try to scrape Google through a headless browser with randomized fingerprints. That approach is a losing arms race — search engines detect and ban scrapers within days, and even when it works, you get DOM soup that your LLM has to clean up.

**This server takes a different approach** — it talks to APIs that are *built* for agents:

| What it does | How |
|---|---|
| Real web search | Brave Search API, Tavily API (whitelisted, ranked, structured JSON) |
| Reads any URL | Jina Reader (handles JS rendering + anti-bot, returns clean markdown) |
| Encyclopedic lookup | Wikipedia MediaWiki API |
| Academic preprints | arXiv API |
| Peer-reviewed papers | Crossref API |
| Tech signal | Hacker News Algolia API |
| Code Q&A | Stack Exchange API (any site) |

The six vertical providers work **without any API keys**; Brave and Tavily keys unlock real-time general web search. That's the highest-quality approach — you get *better* results than scraping because real web-index APIs use signals (click models, freshness, link analysis) that no scraper can replicate.

---

## Quick start

### Option A — `pip install` (when published)

```bash
pip install deep-web-research-mcp
hermes mcp add web-research --command "$(which web-research-mcp)"
```

> **Note on naming.** The PyPI distribution name is `deep-web-research-mcp` (so `pip install deep-web-research-mcp`), but the binary on your `PATH` after install is `web-research-mcp` (defined by `[project.scripts]` in `pyproject.toml`). That's intentional — the binary matches the local launcher `bin/web-research-mcp` and the MCP registration name `web-research`. Same package, two names.

### Option B — Clone from source

```bash
git clone https://github.com/infinit3labs/web-research-mcp.git
hermes mcp add web-research \
  --command "$(pwd)/bin/web-research-mcp"
```

When prompted, accept all 10 tools. Done.

### Option C — Install with Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "web-research": {
      "command": "/Users/code/mcp-servers/web-research/bin/web-research-mcp"
    }
  }
}
```

### Option D — Install with Cursor / any stdio MCP client

```json
{
  "mcpServers": {
    "web-research": {
      "command": "/absolute/path/to/web-research-mcp/bin/web-research-mcp"
    }
  }
}
```

The launcher script auto-creates a venv on first run, installs dependencies from `pyproject.toml`, and sources `web-research.env` for any API keys you've configured.

### 2. (Optional) Add API keys for real web search

```bash
cp web-research.env.example web-research.env
$EDITOR web-research.env
```

| Key | What it unlocks | Free tier |
|---|---|---|
| `BRAVE_API_KEY` | `search_web` real general-web index | 2,000 queries/month |
| `TAVILY_API_KEY` | `search_web` + research-optimized snippets, Tavily News in `search_news`, Tavily Extract fallback for `fetch_url` | 1,000 queries/month |
| `JINA_API_KEY` | Higher fetch rate for `fetch_url` | 1M tokens/month |

The launcher picks up keys from `web-research.env` on every invocation — no restart of your MCP client needed.

### 3. Use it

Ask your agent things like:

> "Search Hacker News and Stack Overflow for the best MCP servers released in 2026"

> "Use pro_mode to research the current state of small language models"

> "Fetch https://arxiv.org/abs/2506.06962 and summarize the methodology"

> "Cross-reference this claim against Wikipedia and arXiv"

---

## Upgrading & rollback

**pip install:**
```bash
pip install --upgrade deep-web-research-mcp   # upgrade to latest
pip install deep-web-research-mcp==0.1.0      # roll back to a specific version
```
Restart your MCP client after changing versions so it re-spawns the server process.

**Source checkout (`bin/web-research-mcp` launcher):**
```bash
git pull                              # upgrade to latest main
git checkout v0.1.0                   # roll back to a tagged release
```
The launcher re-syncs `.venv` from `pyproject.toml` on every invocation, so no manual venv rebuild is needed either way. See [CHANGELOG.md](CHANGELOG.md) for what changed between versions.

---

## Tools

All 10 tools registered in `tools/list`. Tools fall into two layers:

- **Search & fetch** (7 tools) — single-shot lookups. One tool, one API, one result.
- **Deep research** (3 tools) — multi-step pipelines that plan, gather, and structure evidence. Use these when a single search isn't enough.

### Search & fetch

### `search_web` — multi-source general web search
```python
search_web(
    query: str,                  # search query
    max_results: int = 10,       # per source, before dedup (1–30)
    pro_mode: bool = False,      # also fetch top 3 URLs and append excerpts
) -> str
```
Backed by **Brave + Tavily** with URL-canonicalization dedup and cross-source score boosting. Requires `BRAVE_API_KEY` and/or `TAVILY_API_KEY`. With no keys, returns a clear message telling you how to enable it. Tavily runs at **advanced search depth** with `chunks_per_source=3` (multiple relevant excerpts per source) and supports up to 20 results per query.

`pro_mode: true` is the research-killer feature — it runs a normal search, fetches the top 3 results via Jina, and appends the content as the snippet. One call does what would otherwise be `search_web` + 3 × `fetch_url`.

### `fetch_url` — clean markdown of any page
```python
fetch_url(url: str) -> str
```
Goes through Jina Reader, which:
- renders JS-heavy pages (SPAs, React apps)
- bypasses most bot-detection (Jina is whitelisted)
- returns clean markdown with metadata block (`Title:`, `URL Source:`, `Published Time:`)
- truncates to ~20k chars to protect your context window

**Tavily Extract fallback:** with `TAVILY_API_KEY` configured, pages that fail via Jina (hard bot walls, paywalls) are automatically retried through Tavily Extract at advanced depth with markdown formatting. If both paths fail, you get a combined error naming both providers.

### `search_wikipedia` — encyclopedic grounding
```python
search_wikipedia(query: str, max_results: int = 5) -> str
```
Wikipedia MediaWiki API. Keyless. Fast. Best for definitions and historical context.

### `search_academic` — arXiv preprints
```python
search_academic(query: str, max_results: int = 5) -> str
```
Returns title, authors, abstract snippet, published date, PDF URL. Keyless. Best for CS, physics, math, bio.

### `search_news` — Hacker News + mainstream news signal
```python
search_news(query: str, max_results: int = 10) -> str
```
Searches Hacker News (keyless) and — when `TAVILY_API_KEY` is configured — blends in Tavily's news-topic results from the last week, merged and deduplicated with HN. Best for what's trending in tech right now plus its mainstream coverage.

### `search_stackexchange` — Q&A from 180+ sites
```python
search_stackexchange(query: str, max_results: int = 5, site: str = "stackoverflow") -> str
```
Set `site` to any SE community: `serverfault`, `superuser`, `askubuntu`, `math`, `tex`, `datascience`, `ai`, etc. Keyless.

### `search_scholar_meta` — peer-reviewed papers via Crossref
```python
search_scholar_meta(query: str, max_results: int = 5) -> str
```
Returns title, DOI, citation count, publisher, publication date, abstract. Covers papers arXiv doesn't (Elsevier, Springer, Wiley, IEEE, ACM). Keyless.

### Deep research

These three tools compose the search/fetch primitives above into multi-step research workflows. They never call an LLM themselves — the calling model stays in charge of writing the final narrative; the server's job is to plan, gather, and structure evidence with verifiable citations.

### `plan_research` — structured plan only (no fetches)
```python
plan_research(question: str, depth: str = "standard") -> str  # JSON
```
Returns a JSON research plan: sub-questions, recommended sources per sub-question, rationale, queries to run, and estimated searches + fetches. Use this when you want to inspect or modify the plan before committing to the full pipeline.

- `depth`: `"quick"` (2-3 sub-questions), `"standard"` (4-6), `"deep"` (6-8)

### `extract_evidence` — targeted quotes from one URL
```python
extract_evidence(
    url: str,
    question: str,
    max_passages: int = 5,
) -> str  # JSON
```
Fetches the URL via Jina, splits into paragraphs, scores each for relevance to your question, and returns the top passages. Each passage includes `before` / `quote` / `after` context, a `relevance` score (0-1), and a `offset` (character position in the source) so citations are independently verifiable.

Use this when you already have a specific source and want to drill into it for evidence on a narrow claim.

### `research` — full deep-research pipeline
```python
research(question: str, depth: str = "standard") -> str  # markdown + JSON
```
End-to-end research workflow:

1. **Plan** — builds the sub-question plan
2. **Fan out** — searches across the recommended sources for each sub-question in parallel
3. **Rank** — deduplicates URLs across the whole plan, ranks them with source-aware composite scoring (Wikipedia/arXiv/Crossref 2.0×, Stack Exchange 1.7×, web search 1.5×, Hacker News 1.0×)
4. **Fetch** — pulls the top URLs via Jina Reader
5. **Extract** — scores paragraphs for relevance with a quality floor (filters out nav menus, link-only paragraphs, footer cruft)
6. **Return** — emits a structured `ResearchReport`:

```json
{
  "question": "What is retrieval augmented generation?",
  "depth": "quick",
  "plan": { "sub_questions": [...], "estimated_searches": 4, ... },
  "citations": [
    { "id": 1, "url": "...", "title": "...", "source": "wikipedia", "quotes": 2 }
  ],
  "evidence": {
    "sq_def": [
      { "citation_id": 1, "relevance": 0.78, "offset": 1234,
        "before": "...", "quote": "...", "after": "..." }
    ]
  },
  "synthesis_template": "# Research Report: ..."
}
```

The `synthesis_template` is a Markdown skeleton with one section per sub-question plus a Sources table. **You (the model) fill in the narrative**, citing each `[n]` marker against the corresponding entry in `citations`. Every quoted passage carries a character `offset` so a reader can verify the citation against the original page.

`depth` controls breadth:
- `"quick"` — 2-3 sub-questions, ~6 fetches, ~2 minutes
- `"standard"` — 4-6 sub-questions, ~20 fetches, ~3 minutes
- `"deep"` — 6-8 sub-questions, ~32 fetches, ~5 minutes

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client                            │
│  (Claude Desktop, Hermes, Cursor, custom agent)          │
└────────────────────┬────────────────────────────────────┘
                     │ JSON-RPC over stdio
                     ▼
┌─────────────────────────────────────────────────────────┐
│              bin/web-research-mcp                         │
│  • Boots venv (or reuses cached one)                     │
│  • Sources web-research.env for API keys                 │
│  • Execs python -m web_research.server                   │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│           web_research.server (MCPServer)                 │
│  10 tool functions registered via @app.tool() decorator   │
│  • Pydantic-driven JSON schemas from type hints           │
│  • Single shared httpx.AsyncClient per call              │
│  • Graceful degradation: one bad source ≠ failed call    │
└────────────────────┬────────────────────────────────────┘
                     │ asyncio.gather for parallel fan-out
                     ▼
┌─────────────────────────────────────────────────────────┐
│          web_research.providers (7 backends)              │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐               │
│  │ brave    │ │ tavily   │ │ jina_fetch  │  ← general web│
│  └──────────┘ └──────────┘ └─────────────┘               │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐               │
│  │ wikipedia│ │ arxiv    │ │ crossref    │  ← academic   │
│  └──────────┘ └──────────┘ └─────────────┘               │
│  ┌──────────┐ ┌──────────┐                                │
│  │ hn_algolia│ │stackex   │  ← tech signal               │
│  └──────────┘ └──────────┘                                │
│  + merge_results() with URL-canonical dedup               │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

**API-first, not scrape-first.** This is the core thesis. Every source is an official API designed for programmatic access. You get clean structured data, no IP bans, no maintenance burden when sites redesign.

**Per-source error isolation.** Each provider wraps its HTTP call in try/except. A 429 from one source never sinks the whole search — you get partial results plus a clear message about which source failed.

**URL canonicalization.** `merge_results()` strips tracking params (`utm_*`, `fbclid`, `gclid`, `ref`) before dedup, normalizes case on host, drops fragments. When Brave and Tavily return the same article, you see it once with `also_found_in: [brave, tavily]` and a boosted score.

**Shared HTTP client per call.** `httpx.AsyncClient` with connection pooling (`max_connections=20`), sane timeouts (`30s` default, `45s` for fetch_url), and automatic redirect following. New client per call because stdio MCP servers process one request at a time and we want clean state.

**Bounded fan-out + result caching.** Every search/fetch call — plain tool calls and the deep-research pipeline alike — goes through `providers.cached_search`/`cached_fetch`. A per-provider semaphore caps concurrent in-flight requests (`WEB_RESEARCH_MAX_CONCURRENCY`, default 4), so a `depth="deep"` research run can't fan out into dozens of simultaneous requests against one upstream. A short-TTL in-memory cache (`WEB_RESEARCH_CACHE_TTL_SECONDS`, default 300s) avoids repeating identical search/fetch calls within one run; failures and empty results are never cached, so a rate-limited provider still gets retried on the next call instead of being "stuck empty" for the TTL window.

**No headless browsers.** Zero Playwright, Selenium, Puppeteer, or proxy rotation. Smaller attack surface, smaller dependencies, no JVM/Chrome footprint. Jina does the heavy lifting on the few sites that need JS rendering.

---

## Comparison with alternatives

| Feature | This server | SerpAPI MCP | Google scraping MCPs | Local search MCPs |
|---|---|---|---|---|
| General web index | ✅ Brave/Tavily | ✅ Google | ⚠️ Fragile | ❌ |
| API-only (no scraping) | ✅ | ✅ | ❌ | ✅ |
| JS rendering handled | ✅ via Jina | ✅ | ⚠️ Varies | ❌ |
| Academic sources | ✅ arXiv + Crossref | ❌ | ❌ | ⚠️ |
| Tech/Q&A sources | ✅ HN + StackExchange | ❌ | ❌ | ❌ |
| Encyclopedic | ✅ Wikipedia | ❌ | ❌ | ⚠️ |
| Works without API keys | ✅ (9/10 tools) | ❌ | ✅ | ✅ |
| Citation-friendly output | ✅ | ⚠️ | ❌ | ⚠️ |
| MIT-licensed | ✅ | ⚠️ | ⚠️ | ⚠️ |

---

## Testing

```bash
.venv/bin/python tests/e2e_protocol.py
```

This launches the actual server, performs a real MCP `initialize` + `tools/list` handshake, then makes live JSON-RPC calls against every tool and verifies that:

- Real APIs return real data (not stubs)
- Each tool's response has the expected shape
- Error states are handled gracefully
- `search_web` without keys returns a clear "set API keys" message

The live subprocess check exercises all 10 registered tools. The deterministic matrix runs separately and covers provider fixtures, failure degradation, generated schemas, and MCP tool-call content contracts without network access:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

---

## Troubleshooting

### Server starts but tools don't show in my MCP client

Check `hermes mcp list` (or equivalent). The server is registered with `--command`, which means Hermes will exec the launcher directly. Make sure the launcher is executable:

```bash
chmod +x bin/web-research-mcp
```

### `fetch_url` returns truncated content

By design — 20k char cap protects your context window. For longer reads, fetch the page yourself and pass excerpts to `search_web` for follow-up questions, or split into sections via multiple calls.

### `search_web` returns "No web results. This is likely because no API key is configured"

You need at least one of `BRAVE_API_KEY` or `TAVILY_API_KEY` set in `web-research.env` for general web search. The other 9 tools (including the six vertical providers, `fetch_url`, and the three deep-research tools) work without those keys; `JINA_API_KEY` is optional for higher fetch limits.

### Stack Exchange returns 400 Bad Request

If you've configured a custom `filter` parameter, the API rejects unknown filter IDs. Use the default filter (omit the param) — it returns more fields than you need but everything works. This server uses the default.

### Server crashes on first launch

Check stderr for the actual traceback. Common cause: Python <3.10. Check with `python3 --version`.

### Rate limits

Provider calls use bounded retries for transient network errors, HTTP 5xx responses,
and HTTP 429 responses. A provider that remains unavailable is isolated and returns
no results, so other sources and deep-research phases can still complete. Configure
the behavior with `WEB_RESEARCH_TIMEOUT_SECONDS`, provider-specific overrides such as
`WEB_RESEARCH_TIMEOUT_WIKIPEDIA_SECONDS` or `WEB_RESEARCH_TIMEOUT_JINA_SECONDS`,
`WEB_RESEARCH_MAX_RETRIES`, `WEB_RESEARCH_RETRY_BACKOFF_SECONDS`, and
`WEB_RESEARCH_MAX_BACKOFF_SECONDS`. Exhausted 429s are reported as rate-limited;
multi-source responses identify unavailable providers when partial results remain.

Concurrent fan-out per provider is capped by `WEB_RESEARCH_MAX_CONCURRENCY` (default 4),
with per-provider overrides such as `WEB_RESEARCH_MAX_CONCURRENCY_JINA`. Search/fetch
results are cached in-memory for `WEB_RESEARCH_CACHE_TTL_SECONDS` (default 300, set to
`0` to disable) up to `WEB_RESEARCH_CACHE_MAX_ENTRIES` (default 256) entries; set either
to `0` to turn caching off entirely.

Each keyless API has its own limits. If you hit them:
- Wikipedia: ~200 req/min, identify yourself with a real `User-Agent` (this server sends one)
- arXiv: ~1 req/3s for unauthenticated, please back off
- Hacker News Algolia: 10k req/hour with API key, 5k without
- Stack Exchange: 300 req/day without key (plenty for research sessions)
- Crossref: please add mailto in User-Agent (this server does), then it's polite-pool unlimited

---

## Development

### Project layout
```
web-research-mcp/
├── bin/
│   └── web-research-mcp          # Launcher: venv bootstrap + exec
├── src/web_research/
│   ├── __init__.py
│   ├── server.py                  # MCPServer + 10 @app.tool functions
│   └── providers.py               # 7 search backends + Result dataclass
├── tests/
│   └── e2e_protocol.py            # Real subprocess JSON-RPC test
├── web-research.env.example       # API key template
├── pyproject.toml                 # PEP 621, uv-installable
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

### Adding a new tool

1. Add an async function to `providers.py`:
   ```python
   async def search_my_source(query: str, max_results: int, client: httpx.AsyncClient) -> list[Result]:
       try:
           # ... your HTTP call ...
       except Exception as e:
           print(f"[my_source] error: {e}", flush=True)
           return []
       return [Result(title=..., url=..., snippet=..., source="my_source")]
   ```

2. Register it in `server.py`:
   ```python
   @app.tool(name="search_my_source", description="...", annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
   async def search_my_source(query: Annotated[str, Field(description="Search query")], max_results: Annotated[int, Field(ge=1, le=10, default=5)] = 5) -> str:
       async with await _new_client() as client:
           res = await providers.search_my_source(query, max_results, client)
       return _format_results(query, res, "my_source") if res else f"No my_source results for: {query}"
   ```

3. Add a live test case in `tests/e2e_protocol.py`.

4. Update the README's Tools section.

### Coding style

- Python 3.10+, async-first
- Type hints everywhere; let Pydantic derive the MCP JSON schema
- Every provider wraps its network call in try/except and degrades to `[]`
- Per-call HTTP client (`_new_client()`) — don't share across calls in stdio mode

---

## Contributing

PRs welcome. Before opening one:

1. Run the e2e test against a live install: `.venv/bin/python tests/e2e_protocol.py`
2. Add a test case for any new tool
3. Keep `providers.py` independent of MCP-specific types — it should be reusable as a plain Python module
4. Don't add dependencies on headless browsers or proxy rotation — that violates the project's thesis

For major changes, open an issue first.

---

## License

MIT — see [LICENSE](LICENSE).

## Credits

- Built on the [Model Context Protocol](https://modelcontextprotocol.io) by Anthropic
- Uses [Jina Reader](https://jina.ai/reader/) for clean page fetching
- Search APIs: [Brave](https://brave.com/search/api/), [Tavily](https://tavily.com), [Wikipedia](https://www.mediawiki.org/wiki/API:Main_page), [arXiv](https://info.arxiv.org/help/api/index.html), [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/), [Hacker News Algolia](https://hn.algolia.com/api), [Stack Exchange](https://api.stackexchange.com/docs)
