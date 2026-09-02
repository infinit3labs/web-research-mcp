# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/), version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-09-02

### Added
- Longform synthesis scaffolding with citation auditing and structured report rendering.
- Tavily advanced search controls, news enrichment, and Extract fallback for Jina failures.

### Added
- CI now builds the sdist/wheel, runs `twine check`, installs the wheel into a clean venv, and smoke-tests the installed `web-research-mcp` console script over real MCP stdio (`.github/workflows/ci.yml` `package` job).
- `tests/test_packaging.py` — deterministic checks that `pyproject.toml` and `__init__.py` versions stay in sync and that `CHANGELOG.md` has a dated heading for the current version.
- README "Upgrading & rollback" section covering both `pip install` and source-checkout upgrade/rollback paths.
- In-memory TTL cache (`src/web_research/cache.py`) and per-provider concurrency caps for search/fetch calls, applied uniformly to plain tool calls and the deep-research pipeline via `providers.cached_search`/`cached_fetch`. Configurable via `WEB_RESEARCH_CACHE_TTL_SECONDS`, `WEB_RESEARCH_CACHE_MAX_ENTRIES`, and `WEB_RESEARCH_MAX_CONCURRENCY[_<PROVIDER>]`. Failures and empty results are never cached, and cache hits report zero estimated cost in observability events.

### Fixed
- Execute every planned query for each selected provider in the deep-research pipeline.
- Synchronize runtime version metadata with the `0.2.0` package release.
- Refresh tool counts and keyless-tool guidance for the 10-tool server.

## [0.2.0] — 2026-08-23

### Added
- **`plan_research`** tool — builds a structured research plan (sub-questions, recommended sources, estimated cost) without executing it, so the model can review before committing to the full pipeline. Returns JSON.
- **`extract_evidence`** tool — fetches a URL and extracts the passages most relevant to a specific question. Returns each passage with context-before, the quote, context-after, a relevance score, and a character offset into the source page for verifiable citations.
- **`research`** tool — full deep-research pipeline. Decomposes a question into sub-questions, fans out across the existing sources (Wikipedia, arXiv, Hacker News, Stack Exchange, Crossref, Brave, Tavily), fetches the top URLs via Jina, extracts evidence, and returns a structured `ResearchReport` containing the plan, a numbered citation manifest, per-sub-question evidence with quotes + offsets, and a Markdown synthesis template for the calling LLM to fill in.
- **`deep_research.py` module** — heuristic question decomposition, source-aware composite ranking (rebalances Hacker News vs. authoritative sources), paragraph-level relevance scoring with quality floor (filters out nav/footer cruft), and synthesis template generation.
- E2E protocol tests extended to cover the 3 new tools (`plan_research`, `extract_evidence`, `research`); all 10 tools pass against live APIs.

### Design notes
- The server never invokes an LLM itself. The calling model is the LLM; the server's job is to plan, gather, extract evidence, and hand the structure back. This keeps the server deterministic, keyless, and fast.
- Every quoted evidence passage carries a character offset into the original page so citations are independently verifiable — no fabricated references.
- Sub-question routing uses source-aware composite scoring: Wikipedia / arXiv / Crossref get a 2.0× weight, Stack Exchange 1.7×, web search 1.5×, Hacker News 1.0× (HN's high raw vote scores would otherwise crowd out authoritative sources).

## [0.1.0] — 2026-08-22

### Added
- 7 MCP tools: `search_web`, `fetch_url`, `search_wikipedia`, `search_academic`, `search_news`, `search_stackexchange`, `search_scholar_meta`
- Multi-source aggregation in `search_web` (Brave + Tavily) with URL-canonicalization dedup and cross-source score boosting
- `pro_mode` flag on `search_web` that auto-fetches top 3 results via Jina Reader and appends excerpts
- 6 of 7 tools work with zero API keys; Brave/Tavily/Jina keys optional
- Stdout-only stdio MCP transport, compatible with Hermes, Claude Desktop, Cursor, and any MCP client
- Self-bootstrapping launcher (`bin/web-research-mcp`) that creates a venv on first run and sources `web-research.env` automatically
- End-to-end JSON-RPC test suite (`tests/e2e_protocol.py`) that spawns the real server subprocess and verifies all 7 tools against live APIs
- Per-source error isolation: one broken source never sinks the whole call
- Shared `httpx.AsyncClient` per call with connection pooling, sensible timeouts, and redirect following
