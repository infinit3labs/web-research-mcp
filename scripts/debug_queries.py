"""Probe the exact planner queries against Crossref/arXiv to find where items vanish."""
import asyncio
import sys

sys.path.insert(0, "src")

from web_research import deep_research, providers

question = "How does the immune system recognize and attack cancer cells?"


async def main() -> None:
    plan = deep_research.build_plan(question, "quick")
    client = providers.httpx.AsyncClient(
        headers={"User-Agent": providers.USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    try:
        for sq in plan.sub_questions:
            print(f"\n=== {sq.id} sources={sq.sources}")
            for q in sq.queries:
                for pname in sq.sources:
                    fn = {
                        "arxiv": providers.search_arxiv,
                        "crossref": providers.search_crossref,
                        "wikipedia": providers.search_wikipedia,
                    }.get(pname)
                    if fn is None:
                        continue
                    try:
                        res = await fn(q, 8, client)
                        head = "; ".join(r.title[:40] for r in res[:3])
                        print(f"  [{pname}] {q[:50]!r} -> {len(res)} :: {head}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{pname}] {q[:50]!r} -> EXC {exc}")
    finally:
        await client.aclose()


asyncio.run(main())
