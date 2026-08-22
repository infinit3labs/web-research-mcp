# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/), version numbers follow [Semantic Versioning](https://semver.org/).

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
