"""One-off diagnostic: inspect raw provider behavior for the live-run gaps."""
import asyncio
import os
import sys

sys.path.insert(0, "src")

from web_research import providers


async def main() -> None:
    client = providers.httpx.AsyncClient(
        headers={"User-Agent": providers.USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    try:
        # Tavily raw status
        key = os.environ.get("TAVILY_API_KEY")
        print("tavily key present:", bool(key))
        if key:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": "transformer context windows", "max_results": 3},
            )
            print("tavily status:", r.status_code)
            print("tavily body head:", r.text[:300].replace("\n", " "))
            if r.status_code == 200:
                data = r.json()
                first = (data.get("results") or [{}])[0]
                print("tavily result keys:", sorted(first.keys()))

        # Wikipedia score field
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": "large language models context window",
                "srlimit": 3, "format": "json", "utf8": 1, "origin": "*",
            },
        )
        items = r.json().get("query", {}).get("search", [])
        print("\nwikipedia items:", len(items))
        if items:
            print("wikipedia item keys:", sorted(items[0].keys()))
            print("score field:", items[0].get("score"))

        # HN author field name
        r = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": "context window", "hitsPerPage": 2, "tags": "story"},
        )
        hits = r.json().get("hits", [])
        if hits:
            print("\nhn hit keys sample:", sorted(hits[0].keys())[:20])
            print("hn author:", hits[0].get("author"), "| hn by:", hits[0].get("by"))
    finally:
        await client.aclose()


asyncio.run(main())
