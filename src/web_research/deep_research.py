"""Deep research: planning, evidence extraction, and synthesis for multi-step research.

Design notes:
- We do NOT call an LLM ourselves. The MCP client (calling model) is the LLM.
  Our job is to (a) decompose the question into a plan, (b) gather and structure
  evidence with citations, (c) hand the evidence + a synthesis template back to
  the model. This keeps the server deterministic, keyless, and fast.
- Every evidence item carries a stable citation_id so the final report can
  reference [n] and the sources manifest is independently verifiable.
- Evidence extraction is heuristic (TF-style relevance scoring + paragraph
  proximity to the query terms). Good enough for the calling model to pick
  from and re-rank — better than dumping full pages.
"""

from __future__ import annotations

import asyncio
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import httpx

from . import providers


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------

@dataclass
class SubQuestion:
    """One decomposed sub-question with its planned queries and sources."""

    id: str
    question: str
    rationale: str
    queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # provider names
    priority: int = 5  # 1-10, higher = more important


@dataclass
class ResearchPlan:
    """Structured output of plan_research."""

    question: str
    depth: str  # quick | standard | deep
    sub_questions: list[SubQuestion] = field(default_factory=list)
    estimated_searches: int = 0
    estimated_fetches: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "depth": self.depth,
            "sub_questions": [
                {
                    "id": sq.id,
                    "question": sq.question,
                    "rationale": sq.rationale,
                    "queries": sq.queries,
                    "sources": sq.sources,
                    "priority": sq.priority,
                }
                for sq in self.sub_questions
            ],
            "estimated_searches": self.estimated_searches,
            "estimated_fetches": self.estimated_fetches,
            "notes": self.notes,
        }


@dataclass
class Citation:
    """A single source used as evidence."""

    id: int
    url: str
    title: str
    source: str  # provider name (e.g. "arxiv", "jina")
    published: str | None = None
    fetched_at: str | None = None  # ISO date
    quote_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "published": self.published,
            "fetched_at": self.fetched_at,
            "quotes": self.quote_count,
        }


@dataclass
class Evidence:
    """A quoted passage from a fetched page, tied to its source."""

    citation_id: int
    quote: str
    context_before: str = ""
    context_after: str = ""
    relevance: float = 0.0  # 0..1, heuristic
    char_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "relevance": round(self.relevance, 3),
            "offset": self.char_offset,
            "before": self.context_before,
            "quote": self.quote,
            "after": self.context_after,
        }


# --------------------------------------------------------------------------------------
# Plan generation
# --------------------------------------------------------------------------------------

# Question-type detection — drives which sources we recommend
_QUESTION_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\b(what|who|when|where|define|history of)\b", re.I), ["wikipedia", "brave", "tavily"]),
    (re.compile(r"\b(paper|study|research|arxiv|scholar|cite|doi|publication)\b", re.I), ["arxiv", "crossref", "scholar"]),
    (re.compile(r"\b(recent|news|today|latest|2024|2025|2026)\b", re.I), ["hackernews", "brave", "tavily"]),
    (re.compile(r"\b(code|error|how to|implement|tutorial|example|stack|overflow)\b", re.I), ["stackexchange", "brave", "tavily"]),
    (re.compile(r"\b(compare|vs\.?|versus|difference|benchmark)\b", re.I), ["brave", "tavily", "stackexchange"]),
]

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "we", "they", "it",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "why", "how", "when", "where", "tell", "me", "about", "explain", "describe",
    "should", "would", "could", "can", "will", "shall", "may", "might", "must",
    "any", "all", "some", "most", "more", "less", "much", "many", "few",
    "best", "worst", "good", "bad", "really", "very", "just",
}

_DEPTH_CONFIG = {
    "quick":    {"sub_questions": 3, "queries_per_sq": 2, "fetches_per_sq": 2, "max_results": 5},
    "standard": {"sub_questions": 5, "queries_per_sq": 2, "fetches_per_sq": 3, "max_results": 8},
    "deep":     {"sub_questions": 8, "queries_per_sq": 3, "fetches_per_sq": 4, "max_results": 10},
}


def _extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Pull content-bearing keywords from a question."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    counter: Counter[str] = Counter()
    for w in words:
        if w in _STOPWORDS:
            continue
        counter[w] += 1
    return [w for w, _ in counter.most_common(max_keywords)]


def _detect_question_types(question: str) -> list[str]:
    """Return list of source recommendations for the question."""
    sources: list[str] = []
    for pattern, srcs in _QUESTION_PATTERNS:
        if pattern.search(question):
            for s in srcs:
                if s not in sources:
                    sources.append(s)
    if not sources:
        sources = ["brave", "tavily", "wikipedia"]
    return sources


def _decompose_question(question: str, depth: str) -> list[SubQuestion]:
    """Heuristic decomposition — works without an LLM.

    Strategy:
    - Pull the question's keywords
    - Generate factual/contextual/comparative/practical sub-questions
    - Each sub-question gets targeted queries
    """
    keywords = _extract_keywords(question)
    primary = " ".join(keywords[:3]) if keywords else question
    secondary = " ".join(keywords[3:6]) if len(keywords) > 3 else ""

    subs: list[SubQuestion] = []

    # 1. Definitional — what's the thing?
    subs.append(SubQuestion(
        id="sq_def",
        question=f"What is {primary}? What are its defining characteristics?",
        rationale="Establish a shared definition and core concepts before deeper analysis.",
        queries=[f"what is {primary}", f"{primary} definition overview"],
        sources=["wikipedia", "brave", "tavily"],
        priority=9,
    ))

    # 2. Background/history — where did it come from?
    subs.append(SubQuestion(
        id="sq_background",
        question=f"What is the history and current state of {primary}?",
        rationale="Context on origins and evolution helps anchor the rest of the research.",
        queries=[f"{primary} history", f"{primary} recent developments {secondary}"],
        sources=["wikipedia", "hackernews", "brave"],
        priority=7,
    ))

    # 3. Evidence — what's the evidence/proof/data?
    if any(k in question.lower() for k in ("research", "study", "evidence", "data", "compare", "vs", "effective", "works")):
        subs.append(SubQuestion(
            id="sq_evidence",
            question=f"What does the evidence say about {primary}?",
            rationale="Survey peer-reviewed and preprint literature on the topic.",
            queries=[f"{primary} research paper", f"{primary} study evidence"],
            sources=["arxiv", "crossref"],
            priority=8,
        ))

    # 4. Practical/community — how do practitioners use it?
    if any(k in question.lower() for k in ("how", "use", "implement", "tutorial", "code", "best practice", "tips")):
        subs.append(SubQuestion(
            id="sq_practical",
            question=f"How is {primary} used in practice? What do practitioners recommend?",
            rationale="Capture real-world usage patterns, common pitfalls, and best practices.",
            queries=[f"{primary} best practices", f"{primary} implementation"],
            sources=["stackexchange", "hackernews", "brave"],
            priority=7,
        ))

    # 5. Trade-offs / comparisons
    if any(k in question.lower() for k in ("compare", "vs", "versus", "difference", "alternative", "or")):
        subs.append(SubQuestion(
            id="sq_compare",
            question=f"How does {primary} compare to alternatives?",
            rationale="Surface trade-offs and adjacent approaches.",
            queries=[f"{primary} comparison alternatives", f"{primary} vs"],
            sources=["brave", "tavily", "stackexchange"],
            priority=6,
        ))

    # 6. Limitations / criticism
    if depth in ("standard", "deep"):
        subs.append(SubQuestion(
            id="sq_limits",
            question=f"What are the known limitations, criticisms, or open problems with {primary}?",
            rationale="Balanced research requires surfacing downsides and unresolved issues.",
            queries=[f"{primary} limitations", f"{primary} criticism problems"],
            sources=["brave", "tavily", "arxiv"],
            priority=6,
        ))

    # 7. Recent developments (deep only)
    if depth == "deep":
        subs.append(SubQuestion(
            id="sq_recent",
            question=f"What are the most recent developments and breaking changes in {primary}?",
            rationale="Deep research includes the freshest signal.",
            queries=[f"{primary} 2025 2026", f"{primary} latest news"],
            sources=["hackernews", "brave", "tavily"],
            priority=7,
        ))
        # 8. Adjacent topics (deep only)
        subs.append(SubQuestion(
            id="sq_adjacent",
            question=f"What adjacent or complementary concepts should a researcher on {primary} know about?",
            rationale="Maps the surrounding conceptual territory.",
            queries=[f"{primary} related topics", f"{primary} ecosystem"],
            sources=["wikipedia", "brave"],
            priority=4,
        ))

    return subs


def build_plan(question: str, depth: str = "standard") -> ResearchPlan:
    """Build a structured research plan from a question."""
    if depth not in _DEPTH_CONFIG:
        raise ValueError(f"depth must be one of {list(_DEPTH_CONFIG)}, got {depth!r}")

    cfg = _DEPTH_CONFIG[depth]
    sub_questions = _decompose_question(question, depth)
    # Sort by priority descending, then truncate
    sub_questions.sort(key=lambda s: -s.priority)
    sub_questions = sub_questions[: cfg["sub_questions"]]

    notes: list[str] = []
    sources_recommended = _detect_question_types(question)
    notes.append(
        f"Recommended primary sources based on question type: {', '.join(sources_recommended)}"
    )
    if not (any(s in ("brave", "tavily") for s in sources_recommended)):
        notes.append(
            "Note: web-search sources (Brave/Tavily) require API keys for best results. "
            "Wikipedia, arXiv, Hacker News, Stack Exchange, and Crossref work keyless."
        )

    n_searches = len(sub_questions) * cfg["queries_per_sq"]
    n_fetches = cfg["fetches_per_sq"] * len(sub_questions)

    return ResearchPlan(
        question=question,
        depth=depth,
        sub_questions=sub_questions,
        estimated_searches=n_searches,
        estimated_fetches=n_fetches,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# Evidence extraction
# --------------------------------------------------------------------------------------

def _normalize_for_scoring(text: str) -> list[str]:
    """Lowercase + strip punctuation, keeping alphanumeric and intra-word hyphens."""
    text = text.lower()
    # Replace punctuation with spaces (keep apostrophes inside words for "don't" etc.)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def _score_paragraph(paragraph: str, query_terms: Iterable[str]) -> float:
    """Heuristic relevance: keyword density + term coverage + length sweet-spot.

    Bonus: a paragraph that starts with the query term or appears in the first
    few hundred chars gets a position boost (often a page's lede or definition).
    """
    terms = set(query_terms)
    if not terms:
        return 0.0
    words = _normalize_for_scoring(paragraph)
    if not words:
        return 0.0
    word_set = set(words)
    matched = sum(1 for t in terms if t in word_set or any(t in w for w in words))
    coverage = matched / len(terms)  # 0..1
    # Density: matches per word
    density = sum(1 for w in words if any(t in w for t in terms)) / len(words)
    # Length penalty for very short (likely fragments) and very long (likely boilerplate)
    length_score = 1.0
    n_words = len(words)
    if n_words < 15:
        length_score = n_words / 15.0
    elif n_words > 200:
        length_score = 200 / n_words
    # Coverage is the dominant signal; density and length are tiebreakers.
    return 0.65 * coverage + 0.20 * min(1.0, density * 10) + 0.15 * length_score


def extract_evidence(
    content: str,
    question: str,
    max_passages: int = 5,
    context_chars: int = 120,
) -> list[Evidence]:
    """Pick the most relevant passages from a fetched page for a given question.

    Splits on double-newlines (paragraphs), scores each, returns top-k with
    surrounding context. Pure local function — no API calls.
    """
    if not content or not content.strip():
        return []

    query_words = list(_normalize_for_scoring(question))
    if not query_words:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if len(p.strip()) > 80]

    scored: list[tuple[float, int, str]] = []
    for i, p in enumerate(paragraphs):
        score = _score_paragraph(p, query_words)
        # Quality floor: at least 40% of query terms must hit AND at least 20 words.
        # This filters out nav menus, footer crumbs, link-only paragraphs, etc.
        words = _normalize_for_scoring(p)
        hits = sum(1 for w in words if any(t in w for t in query_words))
        if hits < max(1, int(0.4 * len(set(query_words)))) or len(words) < 20:
            continue
        if score > 0:
            scored.append((score, i, p))

    scored.sort(key=lambda x: -x[0])
    top = scored[:max_passages]

    # Compute global char offsets so citations are verifiable
    out: list[Evidence] = []
    cursor = 0
    index_by_offset: dict[int, tuple[int, int]] = {}  # paragraph_index -> (start, end)
    for idx, p in enumerate(paragraphs):
        start = content.find(p, cursor)
        if start == -1:
            start = cursor
        end = start + len(p)
        index_by_offset[idx] = (start, end)
        cursor = end + 2

    for score, idx, p in top:
        start, end = index_by_offset[idx]
        before = content[max(0, start - context_chars):start].strip()
        after = content[end:end + context_chars].strip()
        # Trim very long passages
        quote = p if len(p) <= 500 else p[:500].rsplit(" ", 1)[0] + "…"
        out.append(Evidence(
            citation_id=0,  # filled by caller
            quote=quote,
            context_before=before[-context_chars:],
            context_after=after[:context_chars],
            relevance=min(1.0, score),
            char_offset=start,
        ))
    return out


# --------------------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------------------

@dataclass
class ResearchReport:
    """Structured output of the research tool — the LLM fills the narrative."""

    question: str
    depth: str
    plan: ResearchPlan
    citations: list[Citation]
    evidence_by_subquestion: dict[str, list[Evidence]]
    synthesis_template: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "depth": self.depth,
            "plan": self.plan.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "evidence": {
                sq_id: [e.to_dict() for e in evs]
                for sq_id, evs in self.evidence_by_subquestion.items()
            },
            "synthesis_template": self.synthesis_template,
        }


def _build_synthesis_template(plan: ResearchPlan, citations: list[Citation]) -> str:
    """Markdown skeleton the calling LLM fills in. Lives in the tool output so
    the model sees the exact shape we want."""
    sub_headers = "\n".join(
        f"## {i}. {sq.question}\n*Rationale: {sq.rationale}*\n"
        f"[Your answer here — cite sources using [n] markers referencing the Sources table below]\n"
        for i, sq in enumerate(plan.sub_questions, 1)
    )
    sources_table = "\n".join(
        f"| [{c.id}] | [{c.title}]({c.url}) | {c.source} | {c.published or '—'} |"
        for c in citations
    )
    return f"""# Research Report: {plan.question}

**Depth:** {plan.depth}
**Sub-questions planned:** {len(plan.sub_questions)}
**Sources gathered:** {len(citations)}

---

## Executive Summary
[2–4 sentences. The single most important takeaway, with [citation_id] for each factual claim.]

---

{sub_headers}

---

## Cross-cutting findings
[Patterns that emerge across multiple sub-questions. Cite the strongest sources.]

## Open questions / gaps
[What couldn't be answered from the gathered evidence. Suggests next research direction.]

---

## Sources

| # | Title | Source | Published |
|---|-------|--------|-----------|
{sources_table}

---

## How to use this template

- Each `[citation_id]` references a numbered entry in the **Sources** table.
- The `evidence` field in the tool response contains the exact quoted passages
  supporting each sub-question, with character offsets into the source page
  for verification.
- Aim for ≥ 1 citation per factual claim. Inline markers should match a row
  in the Sources table.
- If `confidence` is needed: cite ≥ 2 independent sources for high confidence.
"""


async def _gather_search(
    sq: SubQuestion,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[providers.Result]:
    """Fan out across the recommended sources for one sub-question."""
    src_set = set(sq.sources)
    primary_query = sq.queries[0] if sq.queries else sq.question
    task_specs: list[tuple[str, Any]] = []
    if "brave" in src_set:
        task_specs.append(("brave", providers.search_brave(primary_query, max_results, client)))
    if "tavily" in src_set:
        task_specs.append(("tavily", providers.search_tavily(primary_query, max_results, client)))
    if "wikipedia" in src_set:
        task_specs.append(("wikipedia", providers.search_wikipedia(primary_query, max_results, client)))
    if "arxiv" in src_set:
        task_specs.append(("arxiv", providers.search_arxiv(primary_query, max_results, client)))
    if "hackernews" in src_set:
        task_specs.append(("hackernews", providers.search_hn(primary_query, max_results, client)))
    if "stackexchange" in src_set:
        task_specs.append(("stackexchange", providers.search_stackexchange(primary_query, max_results, client)))
    if "crossref" in src_set:
        task_specs.append(("crossref", providers.search_crossref(primary_query, max_results, client)))

    if not task_specs:
        return []
    outcomes = await asyncio.gather(*(t[1] for t in task_specs), return_exceptions=True)
    results: list[providers.Result] = []
    for (_tag, _t), outcome in zip(task_specs, outcomes):
        # Each outcome is either a list[Result] (success) or a BaseException.
        # We only extend on a concrete list — this satisfies Pyright and is the
        # safe path; exceptions are silently dropped (logged at the call site).
        if isinstance(outcome, list):
            results.extend(outcome)
    return results


async def _fetch_and_extract(
    result: providers.Result,
    sq: SubQuestion,
    max_passages: int,
    client: httpx.AsyncClient,
) -> tuple[providers.Result | None, list[Evidence]]:
    """Fetch one URL and extract evidence relevant to the sub-question."""
    fetched = await providers.fetch_jina(result.url, client)
    if fetched.get("error") or not fetched.get("content"):
        return None, []
    evidence = extract_evidence(
        content=fetched["content"],
        question=sq.question + " " + " ".join(sq.queries),
        max_passages=max_passages,
    )
    # Update the result with the canonical title if Jina gave us a better one
    if fetched.get("title") and len(fetched["title"]) > len(result.title):
        result.title = fetched["title"]
    return result, evidence


async def run_research(
    question: str,
    depth: str = "standard",
    client_factory=None,  # callable -> httpx.AsyncClient (for tests)
) -> ResearchReport:
    """Full deep-research pipeline.

    1. Build the plan
    2. For each sub-question: fan-out search across recommended sources
    3. Dedup URLs across the whole plan
    4. For each unique URL: fetch (via Jina) and extract evidence
    5. Build the citation manifest and synthesis template
    """
    plan = build_plan(question, depth)
    cfg = _DEPTH_CONFIG[depth]
    max_per_source = cfg["max_results"]
    max_passages_per_page = max(2, cfg["fetches_per_sq"] // 2 or 2)

    if client_factory is None:
        def _default_factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers={"User-Agent": providers.USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        client_factory = _default_factory

    client = client_factory()
    try:
        # Phase 1: gather search results per sub-question
        per_sq_results: dict[str, list[providers.Result]] = {}
        gather_tasks = [_gather_search(sq, max_per_source, client) for sq in plan.sub_questions]
        gathered = await asyncio.gather(*gather_tasks, return_exceptions=True)
        for sq, res in zip(plan.sub_questions, gathered):
            if isinstance(res, BaseException):
                per_sq_results[sq.id] = []
            elif isinstance(res, list):
                per_sq_results[sq.id] = res
            else:
                per_sq_results[sq.id] = []

        # Phase 2: dedup URLs and pick the top N to actually fetch
        seen_urls: dict[str, providers.Result] = {}
        for sq in plan.sub_questions:
            # Pre-populate with search-engine-provided snippets as evidence fallback
            for r in per_sq_results.get(sq.id, []):
                canon = providers._canonical_url(r.url)
                if canon not in seen_urls:
                    seen_urls[canon] = r

        # Pick which URLs to fetch: top by source-aware composite score.
        # Source weights rebalance Hacker News (high raw scores) against
        # Wikipedia/StackExchange (lower raw scores but higher authority).
        SOURCE_WEIGHTS = {
            "wikipedia": 2.0,
            "arxiv": 2.0,
            "crossref": 2.0,
            "stackexchange": 1.7,
            "stackexchange:stackoverflow": 1.7,
            "brave": 1.5,
            "tavily": 1.5,
            "hackernews": 1.0,
        }
        def composite(r: providers.Result) -> float:
            base = float(r.score) if r.score else 1.0
            weight = SOURCE_WEIGHTS.get(r.source, 1.0)
            return base * weight
        all_ranked = sorted(seen_urls.values(), key=composite, reverse=True)
        # Limit total fetches to keep this bounded
        max_total_fetches = min(len(all_ranked), cfg["fetches_per_sq"] * len(plan.sub_questions))
        to_fetch = all_ranked[:max_total_fetches]

        # Phase 3: fetch + extract evidence (parallel)
        fetch_tasks = [
            _fetch_and_extract(r, _matching_sq(r, plan, per_sq_results), max_passages_per_page, client)
            for r in to_fetch
        ]
        fetched_pairs: list[Any] = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Phase 4: build citation manifest + evidence map
        citations: list[Citation] = []
        url_to_id: dict[str, int] = {}
        evidence_by_sq: dict[str, list[Evidence]] = {sq.id: [] for sq in plan.sub_questions}
        today_iso = _today_iso()

        for pair in fetched_pairs:
            if isinstance(pair, BaseException):
                continue
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            result, evidence = pair
            if result is None:
                continue
            canon = providers._canonical_url(result.url)
            if canon not in url_to_id:
                cid = len(citations) + 1
                url_to_id[canon] = cid
                citations.append(Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    source=result.source,
                    published=result.published,
                    fetched_at=today_iso,
                ))
            cid = url_to_id[canon]
            citations[cid - 1].quote_count += len(evidence)
            for ev in evidence:
                ev.citation_id = cid
                # Attach evidence to whichever sub-question it answers best
                best_sq = _best_sq_for_evidence(evidence_by_sq, ev)
                evidence_by_sq[best_sq].append(ev)

        # Add any search-only sources (no fetch succeeded) as low-confidence citations
        for r in to_fetch:
            canon = providers._canonical_url(r.url)
            if canon not in url_to_id and r.snippet:
                cid = len(citations) + 1
                url_to_id[canon] = cid
                citations.append(Citation(
                    id=cid,
                    url=r.url,
                    title=r.title,
                    source=r.source,
                    published=r.published,
                    fetched_at=today_iso,
                    quote_count=0,
                ))
                # Use the search snippet as fallback evidence
                if r.snippet:
                    best_sq = _best_sq_for_evidence(evidence_by_sq, None)
                    evidence_by_sq[best_sq].append(Evidence(
                        citation_id=cid,
                        quote=r.snippet[:500],
                        relevance=0.1,
                        char_offset=0,
                    ))

        template = _build_synthesis_template(plan, citations)

        return ResearchReport(
            question=question,
            depth=depth,
            plan=plan,
            citations=citations,
            evidence_by_subquestion=evidence_by_sq,
            synthesis_template=template,
        )
    finally:
        await client.aclose()


def _matching_sq(  # noqa: E501
    result: providers.Result,
    plan: ResearchPlan,
    per_sq_results: dict[str, list[providers.Result]],
) -> SubQuestion:
    """Find the sub-question this result was gathered for."""
    for sq in plan.sub_questions:
        for r in per_sq_results.get(sq.id, []):
            if providers._canonical_url(r.url) == providers._canonical_url(result.url):
                return sq
    return plan.sub_questions[0]


def _best_sq_for_evidence(
    evidence_by_sq: dict[str, list[Evidence]],
    ev: Evidence | None,
) -> str:
    """Pick the sub-question with the least evidence so far (load-balance).

    If `ev` is None (search-snippet fallback), still load-balance.
    """
    if not evidence_by_sq:
        return "sq_def"
    return min(evidence_by_sq.keys(), key=lambda k: len(evidence_by_sq[k]))


def _today_iso() -> str:
    from datetime import date
    return date.today().isoformat()