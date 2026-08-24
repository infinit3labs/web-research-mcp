# Contributing

Thanks for considering a contribution. This project follows a few simple rules to keep the codebase coherent.

## Code of conduct

Be kind. Assume good faith. We're all here to build good tools.

## Before opening a PR

1. **Search existing issues** — someone may already be working on it
2. **For major changes, open an issue first** — describe the problem before proposing the solution
3. **Run the deterministic matrix locally** — `.venv/bin/python -m coverage run --branch -m unittest discover -s tests -p 'test_*.py' && .venv/bin/python -m coverage report --show-missing`
4. **Run the opt-in live protocol smoke suite when network access is available** — `WEB_RESEARCH_RUN_LIVE_INTEGRATION=1 .venv/bin/python -m unittest tests.test_live_protocol`
5. **Run the full end-to-end protocol check when API/network access is available** — `.venv/bin/python tests/e2e_protocol.py`

## Coding standards

- Python 3.10+, async-first
- Type hints everywhere; let Pydantic derive the MCP JSON schema
- Every provider in `providers.py` wraps its network call in `try/except Exception` and degrades to `[]`
- Don't add dependencies on headless browsers (Playwright, Selenium) or proxy rotation — violates the project's "API-first" thesis
- Don't share `httpx.AsyncClient` instances across tool calls in stdio mode
- No silent failures — log to stderr (`print(..., file=sys.stderr, flush=True)`) so users can diagnose

## Adding a new tool

1. Implement the provider in `providers.py` returning `list[Result]`
2. Register it in `server.py` via `@app.tool(...)`
3. Add a live test case in `tests/e2e_protocol.py`
4. Update the README's Tools section and (if applicable) the architecture diagram
5. Add an entry to CHANGELOG.md under "Unreleased"

## Setting up your dev environment

```bash
git clone https://github.com/infinit3labs/web-research-mcp.git
cd web-research-mcp
./bin/web-research-mcp --help   # Just to trigger venv bootstrap, or:
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
# Deterministic provider and MCP contract matrix (no network or API keys required)
.venv/bin/python -m coverage run --branch -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m coverage report --show-missing
# Optional live smoke suite (requires network access)
WEB_RESEARCH_RUN_LIVE_INTEGRATION=1 .venv/bin/python -m unittest tests.test_live_protocol
```

## Commit messages

Conventional Commits preferred:

- `feat: add DuckDuckGo fallback to search_web`
- `fix: handle 503 from arXiv API gracefully`
- `docs: clarify fetch_url truncation behavior`
- `chore: bump httpx to 0.28`

## Release process

1. Bump version in `pyproject.toml` and `__init__.py` — `tests/test_packaging.py` fails CI if these two drift apart
2. Move CHANGELOG entry from "Unreleased" to a dated version heading matching the new version — `tests/test_packaging.py` checks the heading exists
3. Tag: `git tag -a v0.2.0 -m "Release 0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. Cut a GitHub Release from the tag — the `publish` workflow (`.github/workflows/publish.yml`) builds and publishes to PyPI via Trusted Publishing on `release: published`
6. The `package` job in `.github/workflows/ci.yml` already validates on every push that the build is reproducible (`python -m build` + `twine check`) and that the installed console script speaks the MCP protocol correctly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
