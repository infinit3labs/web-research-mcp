# Contributing

Thanks for considering a contribution. This project follows a few simple rules to keep the codebase coherent.

## Code of conduct

Be kind. Assume good faith. We're all here to build good tools.

## Before opening a PR

1. **Search existing issues** — someone may already be working on it
2. **For major changes, open an issue first** — describe the problem before proposing the solution
3. **Run the unit test suite locally** — `.venv/bin/python -m pytest tests/ --ignore=tests/e2e_protocol.py` should be all green
4. **Run the e2e test locally** — `.venv/bin/python tests/e2e_protocol.py` should show 7/7 passing

## Coding standards

- Python 3.10+, async-first
- Type hints everywhere; let Pydantic derive the MCP JSON schema
- Every provider in `providers.py` is a class implementing `BaseSearchProvider`/`BaseFetchProvider`, registered by capability in `REGISTRY` (see `providers.py`'s module docstring) — never call a provider's HTTP logic directly from a tool handler
- HTTP calls go through `net.request_with_retry` (or the Jina streaming equivalent), which applies bounded retries/backoff on timeouts, connection errors, 429s, and 5xxs — don't hand-roll a new `try/except` around `client.get`
- Any code that fetches a user-supplied URL must validate it with `net.validate_public_url` first (SSRF guard against private/loopback/link-local/metadata targets)
- A provider never raises out of `.search()`/`.fetch()` — failures become `ProviderOutcome.error` / `FetchResult.error`, so a single broken source never sinks a whole research run
- Don't add dependencies on headless browsers (Playwright, Selenium) or proxy rotation — violates the project's "API-first" thesis
- Don't share `httpx.AsyncClient` instances across tool calls in stdio mode
- No silent failures — log to stderr (`print(..., file=sys.stderr, flush=True)`) so users can diagnose

## Adding a new tool

1. Implement the provider as a `BaseSearchProvider`/`BaseFetchProvider` subclass in `providers.py` and register it in `REGISTRY`
2. Register the tool in `server.py` via `@app.tool(...)`, looking the provider up from `REGISTRY` by name or capability — don't import provider classes directly into tool handlers
3. Add unit tests in `tests/test_providers.py` (mock the HTTP layer with `httpx.MockTransport`, no live network)
4. Add a live test case in `tests/e2e_protocol.py`
5. Update the README's Tools section and (if applicable) the architecture diagram
6. Add an entry to CHANGELOG.md under "Unreleased"

## Setting up your dev environment

```bash
git clone https://github.com/infinit3labs/web-research-mcp.git
cd web-research-mcp
./bin/web-research-mcp --help   # Just to trigger venv bootstrap, or:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ --ignore=tests/e2e_protocol.py   # unit tests, offline
.venv/bin/python tests/e2e_protocol.py                             # e2e, hits live APIs
```

## Commit messages

Conventional Commits preferred:

- `feat: add DuckDuckGo fallback to search_web`
- `fix: handle 503 from arXiv API gracefully`
- `docs: clarify fetch_url truncation behavior`
- `chore: bump httpx to 0.28`

## Release process

1. Bump version in `pyproject.toml` and `__init__.py`
2. Move CHANGELOG entry from "Unreleased" to dated version heading
3. Tag: `git tag -a v0.2.0 -m "Release 0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions (TBD) publishes to PyPI

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
