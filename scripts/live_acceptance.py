"""Live acceptance check for the deep citation-capture pipeline.

Runs the `research` tool path end-to-end against real providers and verifies:
  1. >= 10 distinct credible sources with full metadata
  2. passage-level extracts paired to source IDs
  3. at least some primary/recent research material (arXiv/Crossref, recent dates)
Usage: .venv/bin/python scripts/live_acceptance.py "<question>" [depth]
"""
import asyncio
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

from web_research import deep_research


async def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "How do large language model transformers process context windows?"
    )
    depth = sys.argv[2] if len(sys.argv) > 2 else "standard"

    report = await deep_research.run_research(question, depth)
    d = report.to_dict()
    citations = d["citations"]
    evidence = {sq: evs for sq, evs in d["evidence"].items() if evs}
    all_passages = [ev for evs in evidence.values() for ev in evs]

    print(f"Question: {question}")
    print(f"Depth: {depth}")
    print(f"Citations: {len(citations)}")
    print(f"Passages: {len(all_passages)}")

    checks: list[tuple[bool, str]] = []

    # 1. At least 10 distinct sources
    checks.append((len(citations) >= 10, f">=10 distinct sources (got {len(citations)})"))

    # Full metadata presence
    with_publisher = sum(1 for c in citations if c.get("publisher"))
    with_date_or_access = sum(1 for c in citations if c.get("published") or c.get("fetched_at"))
    checks.append((with_publisher >= len(citations) // 2,
                   f">=half of citations carry publisher/venue ({with_publisher}/{len(citations)})"))
    checks.append((with_date_or_access == len(citations),
                   f"all citations have publication or access date ({with_date_or_access}/{len(citations)})"))
    authors_present = sum(1 for c in citations if c.get("authors"))
    checks.append((authors_present > 0, f"at least one citation has authors ({authors_present})"))

    # 2. Passage-level extracts paired to source IDs
    valid_ids = {c["id"] for c in citations}
    paired = all(ev.get("citation_id") in valid_ids for ev in all_passages)
    checks.append((len(all_passages) > 0, f"passage extracts exist ({len(all_passages)})"))
    checks.append((paired, "every passage's citation_id resolves to a manifest entry"))

    # 3. Primary/recent research material
    primary_sources = [c for c in citations if c.get("source") in ("arxiv", "crossref")]
    this_year = str(datetime.now(timezone.utc).date().year)
    recent_years = (this_year, str(int(this_year) - 1), str(int(this_year) - 2))
    recent_primary = [
        c for c in primary_sources
        if (c.get("published") or "")[:4] in recent_years
    ]
    checks.append((bool(primary_sources), f"primary research sources present ({len(primary_sources)})"))
    checks.append((bool(recent_primary),
                   f"recent (last 3y) primary material present ({len(recent_primary)}): "
                   + "; ".join(f"[{c['id']}] {(c.get('published') or '?')[:10]} {c['title'][:60]}"
                               for c in recent_primary[:5])))

    ok = True
    for passed, label in checks:
        print(("PASS " if passed else "FAIL ") + label)
        ok = ok and passed

    if not ok or "-v" in sys.argv:
        print("\n--- Citation manifest sample ---")
        for c in citations[:15]:
            print(json.dumps({k: c[k] for k in ("id", "title", "source", "published", "authors", "publisher")}, ensure_ascii=False))
        print("\n--- Sample passages ---")
        shown = 0
        for sq, evs in evidence.items():
            for ev in evs:
                if shown >= 5:
                    break
                print(f"sq={sq} cid={ev['citation_id']} rel={ev['relevance']} :: {ev['quote'][:120]}...")
                shown += 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
