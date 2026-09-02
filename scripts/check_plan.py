"""Quick manual sanity check of plan generation (run: .venv/bin/python scripts/check_plan.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from web_research import deep_research as dr

plan = dr.build_plan(
    "How do transformer models compare to recurrent models for language modeling?",
    "standard",
)
for sq in plan.sub_questions:
    for q in sq.queries:
        print(f"{sq.id:14s} | {q}")
print("est searches:", plan.estimated_searches)
print("est fetches:", plan.estimated_fetches)

# Near-dup key sanity
from web_research import providers

pairs = [
    ("https://Example.com/a/story.html?utm_source=x#top", "https://www.example.com/a/story"),
    ("https://example.com/list?page=2", "https://example.com/list?page=3"),
    ("https://example.com/list?page=2", "https://example.com/other"),
]
for a, b in pairs:
    print(providers._near_duplicate_key(a) == providers._near_duplicate_key(b), "|", a, "vs", b)
