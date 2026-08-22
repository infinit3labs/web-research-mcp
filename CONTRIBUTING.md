# Contributing

Thanks for considering a contribution. This project follows a few simple rules to keep the codebase coherent.

## Code of conduct

Be kind. Assume good faith. We're all here to build good tools.

## Before opening a PR

1. **Search existing issues** — someone may already be working on it
2. **For major changes, open an issue first** — describe the problem before proposing the solution
3. **Run the test suite locally** — `.venv/bin/python tests/e2e_protocol.py` should show 7/7 passing

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
.venv/bin/pip install -e .
.venv/bin/python tests/e2e_protocol.py
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
