"""Probe Crossref recency-sort behavior for the recency-probe query."""
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
        query = "immune system cancer cells evidence criticism limitations latest developments"
        for sort in ("issued", "relevance"):
            r = await client.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": 5, "sort": sort},
                headers={"User-Agent": providers.USER_AGENT},
            )
            items = r.json().get("message", {}).get("items", [])
            print(f"--- sort={sort}")
            for item in items:
                parts = item.get("issued", {}).get("date-parts", [[None]])[0]
                title = (item.get("title") or ["?"])[0][:60]
                print(f"  {parts} :: {title}")
    finally:
        await client.aclose()


asyncio.run(main())
