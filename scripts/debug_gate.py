"""Probe why an off-topic arXiv result passes the topicality gate."""
import asyncio
import sys

sys.path.insert(0, "src")

from web_research import deep_research, providers

question = (
    "What are the mechanisms and evidence behind intermittent fasting metabolic health claims?"
)
topic_terms = [
    t[:-1] if t.endswith("s") and len(t) > 4 else t
    for t in deep_research._extract_keywords(question, max_keywords=6)
]
print("topic terms:", topic_terms)


async def main() -> None:
    client = providers.httpx.AsyncClient(
        headers={"User-Agent": providers.USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    try:
        results = await providers.search_arxiv(
            "mechanisms evidence intermittent fasting study report analysis", 10, client
        )
        for r in results:
            text = f"{r.title} {r.snippet}".lower()
            hits = [t for t in topic_terms if t in text]
            print(f"hits={len(hits)} {hits[:4]} :: {r.title[:70]}")
    finally:
        await client.aclose()


asyncio.run(main())
