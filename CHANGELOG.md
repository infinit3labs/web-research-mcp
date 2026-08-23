# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/), version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-08-23

### Added
- **Typed provider interface + capability-based registry** (`providers.BaseSearchProvider`, `providers.BaseFetchProvider`, `providers.ProviderRegistry`). Each provider advertises a `Capability` (web search / reference / academic / news / community Q&A / scholarly metadata / fetch); tool handlers look providers up by name or capability instead of importing them directly. New providers slot in by registering themselves.
- **Shared HTTP hardening module** (`net.py`) — bounded retry with full-jitter exponential backoff on 429 / 5xx / transport errors, `Retry-After` header honored, and a 2 MB response-byte cap with binary content-type guard. Centralized so retry and timeout behavior is consistent across every provider.
- **SSRF-safe URL validation** (`net.validate_public_url`). Blocks non-`http(s)` schemes and any hostname whose `A`/`AAAA` records resolve to a private, loopback, link-local (incl. AWS / GCP / Azure cloud metadata `169.254.169.254`), multicast, reserved, or unspecified address. Applied to the Jina fetch path before any outbound call.
- **Graceful degradation** at every layer. A provider missing its API key returns `ProviderOutcome(unavailable=True)`; a provider that fails returns `ProviderOutcome(error=..., rate_limited=...)` instead of raising. One broken source never sinks a research run.
- **64 unit tests** covering the new surfaces (`tests/test_net.py`, `tests/test_providers.py`, `tests/test_deep_research.py`) plus `pytest` + `pytest-asyncio` declared as the `[dev]` extra.

### Fixed
- **Crossref year-only dates no longer crash.** Records with `date-parts: [[2023]]` (no month/day) used to raise `IndexError`; now formatted as `2024-03-15` when full, `2024` when year-only, `None` when missing.
- **`_best_sq_for_evidence` is now deterministic on ties.** Tied sub-questions are broken by sub-question id, so identical inputs always route evidence to the same target across runs.

### Migration
- `server.py` and `deep_research.py` now go through the registry — `providers.search_brave(...)`, `providers.fetch_jina(...)`, etc. are gone. Call sites use `_search_by_name` / `_fetch_by_name` and consume `ProviderOutcome` / `FetchResult` (dataclass, not dict).

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
