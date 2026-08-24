"""Long-form synthesis stage: structured research output -> six-section report scaffold.

Design notes:
- Like the rest of this server we do NOT call an LLM. The calling model writes the
  narrative; this module supplies the deterministic structure around it: the
  section ladder, viewpoint-contrast framing, preprint flags, annotated
  bibliography generation, and post-hoc citation-discipline auditing.
- Output contract mirrors the ``research-report.v1`` agent definition exactly:
  six sections in order, ``numbered_inline`` citation convention, bibliography +
  source register sections, and the guardrail set (unsourced claims forbidden,
  uncertainty flags on conflicting sources, fabricated sources forbidden).
- Conflict detection is deliberately conservative: opposing stances are only
  declared between DIFFERENT sources within one sub-question, and only when the
  quoted evidence carries explicit positive vs negative stance markers. When in
  doubt we say CONSISTENT SOURCES and let the writer check.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

# --------------------------------------------------------------------------------------
# Agent-spec contract constants
# --------------------------------------------------------------------------------------

SECTIONS: list[str] = [
    "Executive Summary",
    "Concept Deep-Dive",
    "Survey of Perspectives",
    "Cutting-Edge Developments",
    "Annotated Bibliography",
    "Source Register",
]

MIN_WORD_COUNT = 3000
FRONTIER_WINDOW_DAYS = 730  # ~24 months, per the agent definition

GUARDRAILS: list[str] = [
    "Every substantive claim carries an inline numbered reference [n] tied to the Source Register.",
    "Conflicting sources get an explicit CONFLICTING SOURCES flag; present each side steelmanned and attributed.",
    "Preprint / non-peer-reviewed status is flagged explicitly wherever it applies.",
    "Fabricating sources, dates, or data is forbidden; gaps are stated as gaps.",
    "Distinguish established results, majority views, contested claims, and your own synthesis.",
]

# Sources whose material is by definition not peer reviewed at capture time.
PREPRINT_SOURCES = {"arxiv"}


def _utc_today() -> date:
    """Timezone-aware 'today' in UTC (DTZ-safe)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date()

_STANCE_POSITIVE = re.compile(
    r"\b(improv\w+|effectiv\w+|outperform\w*|significant\w*|benefit\w*|superior|works|succeed\w*|gain\w*"
    r"|reduc\w+|decreas\w+|lower\w*|eliminat\w+|mitigat\w+)\b",
    re.IGNORECASE,
)
_STANCE_NEGATIVE = re.compile(
    r"\b(no measurable|not significant|no effect|no benefit|no advantage|ineffective|fail\w*|inferior|unclear|harm\w*)\b",
    re.IGNORECASE,
)


def _stance_of(quote: str) -> str:
    """Classify quoted evidence: 'positive', 'negative', or 'neutral'.

    Negation wins: “no measurable reduction” is negative even though it
    contains a positive-stemmed word (“reduction”).
    """
    if _STANCE_NEGATIVE.search(quote):
        return "negative"
    if _STANCE_POSITIVE.search(quote):
        return "positive"
    return "neutral"


# Comparative claims: "A outperforms B", "B remains superior", "A beats B".
_COMPARATIVE_WINNER_PATTERNS = [
    re.compile(r"\b(\w[\w\- ]*?)\s+outperforms?\s+\w+", re.IGNORECASE),
    re.compile(r"\b(\w[\w\- ]*?)\s+(?:remains?|is)\s+superior\b", re.IGNORECASE),
    re.compile(r"\b(\w[\w\- ]*?)\s+beats?\s+\w+", re.IGNORECASE),
]

_FOUNDATIONAL_HINTS = ("sq_def", "sq_background", "definition", "history", "what is")
_FRONTIER_HINTS = ("sq_recent", "recent development", "latest", "breaking")

_CREDIBILITY_BY_SOURCE = {
    "wikipedia": "Tertiary aggregator — good for orientation; verify contested claims against primary sources.",
    "arxiv": "Preprint server — author-submitted, NOT peer reviewed; cite with an explicit preprint flag.",
    "crossref": "Registered publisher metadata — signals peer-reviewed venue publication.",
    "stackexchange": "Community Q&A — practitioner insight, not authoritative.",
    "hackernews": "Community discussion — signal of attention, not of authority.",
    "brave": "Web search index — provenance varies; assess the underlying page directly.",
    "tavily": "Web search index — provenance varies; assess the underlying page directly.",
}
_CREDIBILITY_DEFAULT = "Provenance unverified — confirm publisher and venue before relying on it."


# --------------------------------------------------------------------------------------
# Scaffold construction
# --------------------------------------------------------------------------------------

def _sub_question_tier(sq: dict[str, Any]) -> int:
    """0 = foundational, 1 = body, 2 = frontier — drives the explanation ladder."""
    blob = f"{sq.get('id', '')} {sq.get('question', '')}".lower()
    if any(h in blob for h in _FOUNDATIONAL_HINTS):
        return 0
    if any(h in blob for h in _FRONTIER_HINTS):
        return 2
    return 1


def _ordered_ladder(sub_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order sub-questions foundational -> frontier, priority as tiebreaker."""
    return sorted(
        sub_questions,
        key=lambda sq: (_sub_question_tier(sq), -int(sq.get("priority", 5))),
    )


def _comparative_winner(quote: str) -> str | None:
    """Return the normalized 'winner' named by a comparative claim, if any."""
    for pat in _COMPARATIVE_WINNER_PATTERNS:
        m = pat.search(quote)
        if m:
            return re.sub(r"[^a-z0-9]+", "", m.group(1).lower())
    return None


def _detect_pov(evidence_by_sq: dict[str, list[dict[str, Any]]], citations: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Find genuinely opposing positions and frame them for steelmanned treatment."""
    positions: list[dict[str, Any]] = []

    for sq_id, evs in evidence_by_sq.items():
        pair: tuple[dict[str, Any], dict[str, Any], str] | None = None

        # Path 1: opposing stances from different sources. Negation wins when a
        # quote matches both lexicons ("no measurable reduction" is negative).
        stances: list[tuple[dict[str, Any], str]] = [
            (e, _stance_of(e.get("quote", ""))) for e in evs if e.get("citation_id") is not None
        ]
        supportive = [e for e, s in stances if s == "positive"]
        critical = [e for e, s in stances if s == "negative"]
        if supportive and critical:
            pos, neg = supportive[0], critical[0]
            if pos.get("citation_id") != neg.get("citation_id"):
                pair = (pos, neg, sq_id)

        # Path 2: opposed comparative winners ("A outperforms B" vs "B is superior").
        if pair is None:
            winners: list[tuple[str | None, dict[str, Any]]] = [
                (_comparative_winner(e.get("quote", "")), e)
                for e in evs
                if e.get("citation_id") is not None
            ]
            for i in range(len(winners)):
                for j in range(i + 1, len(winners)):
                    w1, e1 = winners[i]
                    w2, e2 = winners[j]
                    if (
                        w1 and w2 and w1 != w2
                        and e1.get("citation_id") != e2.get("citation_id")
                    ):
                        pair = (e1, e2, sq_id)
                        break
                if pair:
                    break

        if pair is None:
            continue
        pos, neg, _sq = pair
        for stance, ev in (("supporting", pos), ("opposing", neg)):
            cid = ev["citation_id"]
            src = citations.get(cid, {})
            positions.append({
                "stance": stance,
                "claim": ev.get("quote", ""),
                "citations": [cid],
                "attribution": f"[{cid}] {src.get('title', 'unknown source')} ({src.get('url', 'no url')})",
                "steelman_prompt": (
                    f"State the strongest version of the {stance} position as its proponents would "
                    f"present it, anchored on the quoted evidence, attributed to [{cid}]."
                ),
            })
        a, b = positions[0]["citations"][0], positions[1]["citations"][0]
        disagreement = (
            f"Sources [{a}] and [{b}] take opposing positions on “{sq_id}”. Present each "
            f"steelmanned with attribution, then summarize the state of disagreement explicitly: "
            f"it remains unresolved within the current source set."
        )
        return {
            "status": "CONFLICTING SOURCES",
            "positions": positions,
            "disagreement_summary": disagreement,
        }

    # No conflicts found anywhere: report the consensus framing.
    consensus_quotes = [
        (sq_id, e) for sq_id, evs in evidence_by_sq.items() for e in evs
    ]
    seen: set[int] = set()
    support: list[int] = []
    for _sq_id, e in consensus_quotes:
        cid = e.get("citation_id")
        if cid and cid not in seen:
            seen.add(cid)
            support.append(cid)
    return {
        "status": "CONSISTENT SOURCES",
        "positions": [
            {
                "stance": "consensus",
                "claim": "Gathered sources do not openly contradict one another on this question.",
                "citations": support,
                "steelman_prompt": (
                    "Summarize the shared position and note the strongest supporting passage "
                    + (" ".join(f"[{n}]" for n in support[:3]) if support else "(no passages extracted)") + "."
                ),
            }
        ],
        "disagreement_summary": (
            "No direct disagreements detected among gathered sources. State the majority view "
            "with attribution, and flag any tension you notice while writing even if automated "
            "detection did not."
        ),
    }

    # Unreachable in practice; keeps mypy honest.
    return {"status": "CONSISTENT SOURCES", "positions": [], "disagreement_summary": ""}


def _frontier_items(citations: dict[int, dict[str, Any]], today: date) -> list[dict[str, Any]]:
    """Cutting-edge items: preprints plus anything published in the last ~24 months."""
    items: list[dict[str, Any]] = []
    for cid in sorted(citations):
        src = citations[cid]
        is_preprint = src.get("source", "") in PREPRINT_SOURCES or "arxiv.org" in str(src.get("url", ""))
        published_raw = src.get("published")
        published_date: date | None = None
        if published_raw:
            try:
                published_date = date.fromisoformat(str(published_raw)[:10])
            except ValueError:
                published_date = None
        recent = bool(published_date and (today - published_date).days <= FRONTIER_WINDOW_DAYS)
        if not (is_preprint or recent):
            continue
        if is_preprint:
            status = "preprint — not peer reviewed"
        elif src.get("source") == "crossref":
            status = "peer reviewed (venue publication)"
        else:
            status = "venue status unverified — confirm before treating as peer reviewed"
        item = {
            "id": cid,
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            "published": published_raw,
            "preprint": is_preprint,
            "peer_review_status": status,
        }
        if published_date:
            item["age_days"] = (today - published_date).days
        items.append(item)
    items.sort(key=lambda i: i.get("age_days", 10**9))
    return items


def _annotation_for(src: dict[str, Any]) -> dict[str, str]:
    """Annotation block covering relevance, credibility, and date — required on 100% of references."""
    q = int(src.get("quotes", 0) or 0)
    if q >= 5:
        relevance = f"Central source — {q} extracted passages anchor multiple sections."
    elif q >= 1:
        relevance = f"Supporting source — {q} extracted passage(s) back specific claims."
    else:
        relevance = "Peripheral — surfaced in search but contributed no extractable passages; corroborative only."
    credibility = _CREDIBILITY_BY_SOURCE.get(src.get("source", ""), _CREDIBILITY_DEFAULT)
    fetched = src.get("fetched_at")
    published = src.get("published")
    if published:
        date_note = f"Published {published}; accessed {fetched or 'date unrecorded'}."
    else:
        date_note = f"No publication date captured; accessed {fetched or 'date unrecorded'} — recency cannot be verified."
    return {"relevance": relevance, "credibility": credibility, "date": date_note}


def build_longform_scaffold(report: dict[str, Any]) -> dict[str, Any]:
    """Turn a ``ResearchReport.to_dict()`` payload into the six-section scaffold.

    Deterministic: every section brief carries its sub-question inputs, citation
    ids, guardrail instructions, and (where detected) framed conflicts for the
    calling model to narrate.
    """
    citations_list = report.get("citations", [])
    citations = {int(c["id"]): c for c in citations_list}
    evidence_by_sq = report.get("evidence", {})
    plan = report.get("plan", {})
    sub_questions = plan.get("sub_questions", [])

    # Evidence lookup: citation_id -> [(sq_id, evidence)]
    evidence_by_cid: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for sq_id, evs in evidence_by_sq.items():
        for e in evs:
            evidence_by_cid.setdefault(int(e.get("citation_id", 0)), []).append((sq_id, e))

    ladder = []
    for sq in _ordered_ladder(sub_questions):
        sq_id = sq.get("id", "")
        cids: list[int] = []
        for e in evidence_by_sq.get(sq_id, []):
            cid = int(e.get("citation_id", 0))
            if cid and cid not in cids:
                cids.append(cid)
        tier = _sub_question_tier(sq)
        tier_label = {0: "foundational", 1: "developing", 2: "frontier"}[tier]
        ladder.append({
            "sub_question": sq_id,
            "tier": tier_label,
            "question": sq.get("question", ""),
            "rationale": sq.get("rationale", ""),
            "queries": sq.get("queries", []),
            "citation_ids": cids,
        })

    pov = _detect_pov(evidence_by_sq, citations)
    today = _utc_today()

    scaffold: dict[str, Any] = {
        "question": report.get("question", plan.get("question", "")),
        "depth": report.get("depth", plan.get("depth", "standard")),
        "min_word_count": MIN_WORD_COUNT,
        "sections": list(SECTIONS),
        "guardrails": list(GUARDRAILS),
        "section_briefs": {
            "Executive Summary": {
                "instruction": (
                    f"In at most 400 words, state the single most important takeaway for: "
                    f"“{report.get('question', '')}”. Every factual sentence carries [n]."
                ),
                "key_citation_ids": sorted(evidence_by_cid.keys()),
            },
            "Concept Deep-Dive": {
                "instruction": (
                    "Build the concept-explanation ladder: begin each step from first principles "
                    "in plain language, keep precision, and climb toward the research frontier. "
                    "Do not skip a rung."
                ),
                "ladder": ladder,
            },
            "Survey of Perspectives": {
                "instruction": (
                    "Identify genuinely opposing positions in the source set. Present each one "
                    "steelmanned, with attribution, then summarize the state of disagreement. "
                    "No silent winner-picking."
                ),
                **pov,
            },
            "Cutting-Edge Developments": {
                "instruction": (
                    "Surface findings from roughly the last 24 months. Flag preprint / "
                    "non-peer-reviewed status explicitly next to every affected claim."
                ),
                "items": _frontier_items(citations, today),
            },
            "Annotated Bibliography": {
                "instruction": (
                    "Every reference gets an annotation block covering relevance, credibility, "
                    "and date. 100% coverage required before the report ships."
                ),
            },
            "Source Register": {
                "instruction": "Map each inline number to exactly one bibliography entry; never renumber mid-report.",
            },
        },
        "annotated_bibliography": [],
        "source_register": [],
    }

    for cid in sorted(citations):
        src = citations[cid]
        scaffold["annotated_bibliography"].append({
            "id": cid,
            "title": src.get("title", ""),
            "authors": list(src.get("authors", []) or []),
            "publisher": src.get("publisher"),
            "url": src.get("url", ""),
            "published": src.get("published"),
            "fetched_at": src.get("fetched_at"),
            "annotation": _annotation_for(src),
        })
    scaffold["source_register"] = [
        {"n": entry["id"], "bibliography_entry_id": entry["id"]}
        for entry in scaffold["annotated_bibliography"]
    ]
    return scaffold


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def render_markdown(scaffold: dict[str, Any]) -> str:
    """Render the scaffold as the Markdown skeleton the calling LLM fills in."""
    briefs = scaffold["section_briefs"]
    lines: list[str] = [
        f"# Research Report: {scaffold['question']}",
        "",
        f"**Depth:** {scaffold['depth']} · **Minimum length:** {scaffold['min_word_count']} words",
        "",
        "## Writing guardrails",
        "",
    ]
    lines.extend(f"- {g}" for g in scaffold["guardrails"])
    lines.extend(["", "---", ""])

    # 1. Executive Summary
    es = briefs["Executive Summary"]
    lines.extend([
        "## Executive Summary",
        "",
        f"*{es['instruction']}*",
        "",
        "[Your summary here.]",
        "",
    ])

    # 2. Concept Deep-Dive
    dd = briefs["Concept Deep-Dive"]
    lines.extend(["## Concept Deep-Dive", "", f"*{dd['instruction']}*", ""])
    for i, step in enumerate(dd["ladder"], 1):
        cites = " ".join(f"[{c}]" for c in step["citation_ids"]) or "(no evidence gathered yet)"
        lines.extend([
            f"### 2.{i} [{step['tier'].upper()}] {step['question']}",
            "",
            f"*Rationale:* {step['rationale']}",
            f"*Evidence available:* {cites}",
            "",
            "[Foundations first, then nuance, ending at the edge of what sources support.]",
            "",
        ])

    # 3. Survey of Perspectives
    pov = briefs["Survey of Perspectives"]
    lines.extend(["## Survey of Perspectives", "", f"*{pov['instruction']}*", ""])
    if pov["status"] == "CONFLICTING SOURCES":
        lines.append("**CONFLICTING SOURCES** — treat the disagreement as live and unresolved.")
        lines.append("")
        for pos in pov["positions"]:
            attr = pos.get("attribution", "")
            lines.extend([
                f"- **{pos['stance'].capitalize()} position** {attr}",
                f"  - Quoted anchor: “{pos['claim']}”",
                f"  - {pos['steelman_prompt']}",
            ])
        lines.extend(["", f"*Disagreement summary:* {pov['disagreement_summary']}", ""])
    else:
        lines.extend([
            "*Consistent sources detected — still verify tension manually while writing.*",
            "",
            f"{pov['disagreement_summary']}",
            "",
        ])

    # 4. Cutting-Edge Developments
    ce = briefs["Cutting-Edge Developments"]
    lines.extend(["## Cutting-Edge Developments", "", f"*{ce['instruction']}*", ""])
    if not ce["items"]:
        lines.extend(["_No sources within the ~24-month frontier window were captured._", ""])
    for item in ce["items"]:
        flag = f" **({item['peer_review_status']})**" if item["preprint"] else ""
        pub = f", published {item['published']}" if item.get("published") else ""
        lines.extend([f"- [{item['id']}] **{item['title']}**{flag}{pub}", f"  {item['url']}"])
    lines.append("")

    # 5. Annotated Bibliography
    lines.extend(["## Annotated Bibliography", "", f"*{briefs['Annotated Bibliography']['instruction']}*", ""])
    for entry in scaffold["annotated_bibliography"]:
        authors = ", ".join(entry["authors"]) if entry["authors"] else "—"
        ann = entry["annotation"]
        lines.extend([
            f"**[{entry['id']}] {entry['title']}**",
            f"- Authors: {authors} · Publisher: {entry['publisher'] or '—'} · Published: {entry['published'] or '—'}",
            f"- URL: {entry['url']}",
            f"- Relevance: {ann['relevance']}",
            f"- Credibility: {ann['credibility']}",
            f"- Date: {ann['date']}",
            "",
        ])

    # 6. Source Register
    lines.extend(["## Source Register", "", f"*{briefs['Source Register']['instruction']}*", ""])
    lines.extend(["| [n] | Bibliography entry |", "|-----|----------------------|"])
    for reg in scaffold["source_register"]:
        lines.append(f"| [{reg['n']}] | [{reg['bibliography_entry_id']}] |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Citation-discipline audit
# --------------------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_HR_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_BULLET_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_ITALIC_LABEL_RE = re.compile(r"^\*[^*]+\*")
_BOLD_FLAG_RE = re.compile(r"^\*\*[^*]+\*\*")  # e.g. **CONFLICTING SOURCES** banners
_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_CLAIM_WORDS = 4


def _is_substantive(sentence: str) -> bool:
    words = [w for w in re.split(r"\s+", sentence.strip()) if w]
    return len(words) >= _MIN_CLAIM_WORDS


def _logical_blocks(text: str) -> list[tuple[int, int, str]]:
    """Group raw lines into logical blocks: bullets (with wrapped continuations),
    paragraphs, or single exempt lines.

    Yields (first_line_no, last_line_no, joined_text). Headings, table rows,
    horizontal rules, and code fences become their own single-line blocks.
    Wrapped bullet/paragraph lines are joined so sentences can be audited whole.
    """
    lines = text.splitlines()
    blocks: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 0

    def flush(end_line: int) -> None:
        nonlocal current
        if not current:
            return
        blocks.append((current_start, end_line, " ".join(current)))
        current = []

    def is_structural(s: str, raw: str) -> bool:
        return (
            not s
            or s.startswith("```")
            or _HEADING_RE.match(raw) is not None
            or _TABLE_ROW_RE.match(raw) is not None
            or _HR_RE.match(s) is not None
            or _BULLET_RE.match(raw) is not None
        )

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        line_no = i + 1

        if not stripped or stripped.startswith("```") or _HEADING_RE.match(raw) or _TABLE_ROW_RE.match(raw) or _HR_RE.match(stripped) or _BOLD_FLAG_RE.match(stripped):
            # Structural lines (headings, tables, rules, banners) are never
            # claims — act as block boundaries only.
            flush(line_no - 1)
            i += 1
            continue

        if _BULLET_RE.match(raw):
            flush(line_no - 1)
            current_start = line_no
            current = [stripped]
            i += 1
            # Continuations: any following line until blank/structural/column-0 text.
            while i < len(lines):
                cont = lines[i]
                c = cont.strip()
                if is_structural(c, cont) or (c and len(cont) - len(cont.lstrip()) == 0):
                    break
                current.append(c)
                i += 1
            flush(i)
            continue

        # Plain paragraph: accumulate until blank/structural line.
        flush(line_no - 1)
        current_start = line_no
        current = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            n = nxt.strip()
            if is_structural(n, nxt):
                break
            current.append(n)
            i += 1
        flush(i)

    flush(len(lines))
    return blocks


def _sentence_line(sentence: str, block_text: str, line_map: list[tuple[int, int]]) -> int:
    """Map a sentence from the joined block back to its original line number."""
    pos = block_text.find(sentence.strip()[:40])
    if pos == -1:
        return line_map[0][1]
    for start, line_no in line_map:
        if pos >= start:
            current = (start, line_no)
        else:
            break
    return current[1]


def audit_citations(text: str, known_refs: set[int] | None = None) -> dict[str, Any]:
    """Check a drafted report for citation discipline.

    Sentences are evaluated across full logical blocks (hard-wrapped Markdown
    is joined before sentence splitting), so an [n] marker on a later visual
    line of the same sentence satisfies the requirement. Flagged sentences are
    reported at the exact line where they begin.

    Returns:
        uncited_claims: list of (line_number, snippet) for substantive sentences
            with no [n] marker anywhere in their block. Headings, table rows,
            rules, code blocks, and short fragments are exempt.
        unknown_refs: per-line lists of [n] values not present in
            ``known_refs`` (only checked when ``known_refs`` is provided).
        total_inline_markers / distinct_refs_used: coverage stats.
    """
    uncited: list[tuple[int, str]] = []
    unknown: list[list[int]] = []
    total_markers = 0
    distinct: set[int] = set()

    for start_line, _end_line, joined in _logical_blocks(text):
        markers = [int(m) for m in _MARKER_RE.findall(joined)]
        total_markers += len(markers)
        distinct.update(markers)

        if known_refs is not None:
            bad = sorted({m for m in markers if m not in known_refs})
            if bad:
                unknown.append(bad)

        # Offset -> line map: cumulative stripped-line lengths within the block.
        raw_lines = text.splitlines()
        line_map: list[tuple[int, int]] = [(0, start_line)]
        consumed = 0
        for ln in range(start_line, _end_line + 1):
            consumed += len(raw_lines[ln - 1].strip())
            line_map.append((consumed, ln + 1))

        for sentence in _SENTENCE_SPLIT_RE.split(joined):
            if not _is_substantive(sentence):
                continue
            if _MARKER_RE.search(sentence):
                continue
            snippet = sentence.strip()
            if _ITALIC_LABEL_RE.match(snippet):
                snippet = _ITALIC_LABEL_RE.sub("", snippet).strip()
            if snippet:
                uncited.append((_sentence_line(snippet, joined, line_map), snippet[:120]))

    return {
        "uncited_claims": uncited,
        "unknown_refs": unknown,
        "total_inline_markers": total_markers,
        "distinct_refs_used": distinct,
    }
