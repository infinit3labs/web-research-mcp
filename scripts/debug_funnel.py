"""Diagnose candidate loss through the run_research phases for one topic."""
import asyncio
import os
import sys

sys.path.insert(0, "src")

from web_research import deep_research

question = (
    "What are the mechanisms and evidence behind intermittent fasting metabolic health claims?"
)


async def main() -> None:
    plan = deep_research.build_plan(question, "quick")
    print("plan:", [(sq.id, sq.sources, sq.queries) for sq in plan.sub_questions])

    client = deep_research.httpx.AsyncClient(
        headers={"User-Agent": deep_research.providers.USER_AGENT},
        timeout=30.0, follow_redirects=True,
    )
    try:
        per_sq = {}
        gathered = await asyncio.gather(
            *[deep_research._gather_search(sq, 5, client) for sq in plan.sub_questions],
            return_exceptions=True,
        )
        for sq, res in zip(plan.sub_questions, gathered):
            per_sq[sq.id] = res if isinstance(res, list) else []
            print(f"{sq.id}: {len(per_sq[sq.id])} raw results")

        merged = deep_research.providers.merge_results(*per_sq.values(), max_total=30)
        print("merged unique canonical urls:", len(merged))

        seen = {deep_research.providers._canonical_url(r.url): r for r in merged}
        near_seen, uniq = set(), []
        for r in sorted(seen.values(), key=lambda x: x.score, reverse=True):
            k = deep_research.providers._near_duplicate_key(r.url)
            if k not in near_seen:
                near_seen.add(k)
                uniq.append(r)
        print("after near-dup:", len(uniq))

        terms = [t[:-1] if t.endswith("s") and len(t) > 4 else t
                 for t in deep_research._extract_keywords(question, max_keywords=6)]
        print("topic terms:", terms)

        def hits(r):
            text = f"{r.title} {r.snippet}".lower()
            return sum(1 for t in terms if t in text)

        strict = [r for r in uniq if hits(r) >= 2]
        loose = [r for r in uniq if hits(r) >= 1]
        print(f"strict(>=2): {len(strict)}  loose(>=1): {len(loose)}")
        budget = 4 * len(plan.sub_questions)
        needed = min(len(uniq), budget)
        kept = strict if len(strict) >= needed else loose if len(loose) >= needed else uniq
        print(f"needed={needed} kept={len(kept)}")

        from collections import Counter
        print("kept by source:", Counter(r.source.split(':')[0] for r in kept))
        fetched = await asyncio.gather(
            *[deep_research.providers.provider_registry.get("jina", deep_research.providers.Capability.FETCH).fetch(r.url, client) for r in kept[:needed]],
            return_exceptions=True,
        )
        ok = sum(1 for f in fetched if isinstance(f, dict) and f.get("content") and not f.get("error"))
        errs = [f.get("error", "?")[:60] for f in fetched if isinstance(f, dict) and (f.get("error") or not f.get("content"))]
        print(f"fetch ok: {ok}/{len(fetched)}; errors: {errs[:5]}")
    finally:
        await client.aclose()


asyncio.run(main())
