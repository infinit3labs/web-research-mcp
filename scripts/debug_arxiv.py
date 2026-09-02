"""Probe arXiv AND syntax variants to find one that works through httpx."""
import asyncio
import sys

import httpx
import xml.etree.ElementTree as ET

sys.path.insert(0, "src")

from web_research import providers


async def main() -> None:
    async with httpx.AsyncClient(headers={"User-Agent": providers.USER_AGENT}, timeout=30.0) as client:
        for label, sq, sort in [
            ("and-space", "all:immune AND all:system", "relevance"),
            ("and-plus", "all:immune+AND+all:system", "relevance"),
            ("and-encoded-params", "all:immune AND all:system", "submittedDate"),
            ("quoted-phrase", 'all:"immune system"', "submittedDate"),
            ("single-term-date-sort", "all:immunity", "submittedDate"),
        ]:
            r = await client.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": sq, "max_results": 3, "sortBy": sort,
                        "sortOrder": "descending"},
            )
            try:
                root = ET.fromstring(r.text)
                entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                titles = [(e.find("{http://www.w3.org/2005/Atom}title").text or "")[:45] for e in entries]
                dates = []
                for e in entries:
                    pub = e.find("{http://www.w3.org/2005/Atom}published")
                    dates.append((pub.text or "")[:10] if pub is not None else "?")
            except Exception as exc:  # noqa: BLE001
                titles, dates = [f"parse error: {exc}"], []
            print(f"{label}: {len(titles)} :: {list(zip(titles, dates))[:3]}")
            await asyncio.sleep(3)


asyncio.run(main())
