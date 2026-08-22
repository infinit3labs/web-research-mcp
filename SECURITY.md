# Security

## Reporting a vulnerability

If you discover a security issue in this project, please email security@infinit3labs.com rather than opening a public GitHub issue. We aim to respond within 48 hours.

## API keys

This project reads API keys from `web-research.env` at runtime. **Never commit this file** — it's in `.gitignore` by default and contains:

- `BRAVE_API_KEY` — Brave Search API key
- `TAVILY_API_KEY` — Tavily research API key
- `JINA_API_KEY` — Jina Reader API key

### If you accidentally commit a key

1. **Revoke the key immediately** at the provider's dashboard:
   - Brave: https://brave.com/search/api/
   - Tavily: https://tavily.com
   - Jina: https://jina.ai/reader/
2. Generate a new key
3. Update your local `web-research.env`
4. The leaked key may still be in git history — use `git filter-repo` or BFG Repo-Cleaner to purge it, then force-push

The project maintainers are not responsible for costs incurred by leaked keys.

## Network egress

This server makes outbound HTTPS requests to the following hosts when the relevant tool is called:

- `api.search.brave.com` (Brave Search)
- `api.tavily.com` (Tavily)
- `r.jina.ai` (Jina Reader proxy)
- `en.wikipedia.org` (Wikipedia MediaWiki)
- `export.arxiv.org` (arXiv)
- `api.crossref.org` (Crossref)
- `hn.algolia.com` (HN Algolia)
- `api.stackexchange.com` (Stack Exchange)

All requests are made with a descriptive `User-Agent` header (`research-agent/0.1`). No tracking pixels, no telemetry, no analytics.
