"""Tests for the deep-research pipeline: plan generation, evidence extraction, dedup.

These are the LLM-side of the research flow — they don't hit the network and
are the easiest things to test thoroughly. The end-to-end research() call
needs network and is covered by tests/e2e_protocol.py.
"""

from __future__ import annotations

import pytest

from src.web_research import deep_research


# --------------------------------------------------------------------------------------
# build_plan
# --------------------------------------------------------------------------------------


def test_build_plan_rejects_unknown_depth():
    with pytest.raises(ValueError, match="depth must be one of"):
        deep_research.build_plan("anything", depth="ultra")


@pytest.mark.parametrize("depth,n_sq", [("quick", 3), ("standard", 5), ("deep", 8)])
def test_build_plan_respects_depth_caps(depth, n_sq):
    plan = deep_research.build_plan("compare Python and Rust for systems programming", depth=depth)
    assert len(plan.sub_questions) <= n_sq
    assert plan.depth == depth
    # estimated_searches reflects actual sub-questions, not just depth cap
    assert plan.estimated_searches == len(plan.sub_questions) * {
        "quick": 2, "standard": 2, "deep": 3,
    }[depth]


def test_build_plan_sorts_by_priority_descending():
    plan = deep_research.build_plan("how do I implement a custom React hook?", depth="standard")
    priorities = [sq.priority for sq in plan.sub_questions]
    assert priorities == sorted(priorities, reverse=True)


def test_build_plan_adds_practical_subquestion_for_how_to_questions():
    plan = deep_research.build_plan("how do I use Kubernetes?", depth="standard")
    ids = {sq.id for sq in plan.sub_questions}
    assert "sq_practical" in ids


def test_build_plan_adds_comparison_subquestion_for_vs_questions():
    plan = deep_research.build_plan("compare Postgres vs SQLite for an embedded app", depth="standard")
    ids = {sq.id for sq in plan.sub_questions}
    assert "sq_compare" in ids


def test_build_plan_always_has_a_definitional_subquestion():
    plan = deep_research.build_plan("what is the capital of France?", depth="quick")
    assert any(sq.id == "sq_def" for sq in plan.sub_questions)


def test_build_plan_notes_recommend_keyless_sources_when_no_web_keyword():
    plan = deep_research.build_plan("what is photosynthesis?", depth="quick")
    notes_blob = " ".join(plan.notes).lower()
    # 'photosynthesis' doesn't match web/news/paper/code/compare patterns → wikipedia/brave/tavily default
    assert "wikipedia" in notes_blob


# --------------------------------------------------------------------------------------
# extract_evidence
# --------------------------------------------------------------------------------------


def test_extract_evidence_returns_empty_for_empty_content():
    assert deep_research.extract_evidence("", "anything", 5) == []
    assert deep_research.extract_evidence("   \n  \n  ", "anything", 5) == []


def test_extract_evidence_returns_empty_for_empty_query_words():
    body = "Some real text with actual content and words that are long enough."
    out = deep_research.extract_evidence(body, "a the an", 5)  # all stopwords
    assert out == []


def test_extract_evidence_picks_relevant_paragraphs():
    body = """
Navigation menu. Home About Contact Support.

Retrieval augmented generation combines a retrieval system with a generative model. The retrieval system
fetches documents from a vector database at inference time, and the generator uses those documents to
produce more accurate answers grounded in real sources rather than memorized parameters.

Cookies disclaimer. This site uses cookies. By continuing you accept cookies. Manage preferences.
Privacy policy. Terms of service. All rights reserved.

In production, retrieval augmented generation is widely used for question answering, customer support
chatbots, and document summarization. Major frameworks like LangChain and LlamaIndex provide
abstractions over the retrieval step, the embedding model, and the language model.
"""
    passages = deep_research.extract_evidence(body, "retrieval augmented generation", max_passages=3)
    assert passages
    # The two truly-relevant paragraphs should be the top hits; nav/footer should be filtered
    for p in passages:
        # None of the picked passages should be the cookies/privacy paragraph
        assert "cookies" not in p.quote.lower()
        assert "privacy" not in p.quote.lower()


def test_extract_evidence_includes_context_windows():
    body = (
        "Some long prefix text that establishes context for the next paragraph. " * 5 +
        "\n\n" +
        "Retrieval augmented generation fetches relevant documents and conditions the model on them. " * 3 +
        "\n\n" +
        "Some long suffix text that comes after the relevant passage. " * 5
    )
    passages = deep_research.extract_evidence(body, "retrieval augmented generation", max_passages=1, context_chars=80)
    assert len(passages) == 1
    assert passages[0].context_before
    assert passages[0].context_after
    # Char offset should be inside the original body so citations are verifiable
    assert 0 < passages[0].char_offset < len(body)


def test_extract_evidence_truncates_very_long_quotes():
    long_para = "Retrieval augmented generation " + ("word " * 200)
    body = long_para + "\n\n" + "unrelated filler " * 20
    passages = deep_research.extract_evidence(body, "retrieval augmented generation", max_passages=1)
    assert passages
    assert len(passages[0].quote) <= 600  # capped at ~500 + ellipsis


# --------------------------------------------------------------------------------------
# _best_sq_for_evidence (load-balancing)
# --------------------------------------------------------------------------------------


def test_best_sq_for_evidence_load_balances():
    from src.web_research.deep_research import Evidence, _best_sq_for_evidence
    e1 = Evidence(citation_id=1, quote="q1")
    e2 = Evidence(citation_id=1, quote="q2")
    state: dict[str, list[Evidence]] = {"sq_a": [e1, e2], "sq_b": [], "sq_c": [e1]}
    # sq_b has the least evidence (0) → should be picked
    assert _best_sq_for_evidence(state, None) == "sq_b"


def test_best_sq_for_evidence_breaks_ties_deterministically_by_id():
    """When multiple sub-questions have equal evidence, pick by id — not by dict iteration order."""
    from src.web_research.deep_research import Evidence, _best_sq_for_evidence
    e1 = Evidence(citation_id=1, quote="q1")
    # sq_z and sq_m are tied; min-key on the (count, id) tuple should pick sq_m
    state: dict[str, list[Evidence]] = {"sq_z": [e1], "sq_m": [e1]}
    assert _best_sq_for_evidence(state, None) == "sq_m"
    # Same call with the keys in the opposite order should still return sq_m
    state2: dict[str, list[Evidence]] = {"sq_m": [e1], "sq_z": [e1]}
    assert _best_sq_for_evidence(state2, None) == "sq_m"


def test_best_sq_for_evidence_falls_back_to_sq_def():
    from src.web_research.deep_research import _best_sq_for_evidence
    assert _best_sq_for_evidence({}, None) == "sq_def"
