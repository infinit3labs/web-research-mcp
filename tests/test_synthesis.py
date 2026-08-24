"""Tests for the long-form synthesis stage (kanban task t_5b2fdc30).

The synthesis stage turns a structured ResearchReport (from deep_research.run_research)
into the six-section long-form scaffold defined by the research-report.v1 agent
definition, with:

- concept-explanation ladder sections (foundational -> frontier)
- steelmanned viewpoint-contrast treatment with attribution
- explicit preprint/non-peer-reviewed flags on frontier sources
- citation discipline validation (every claim sentence carries [n])
- 100% annotated-bibliography coverage (relevance / credibility / date)

All functions are deterministic — the calling LLM fills the prose; this module
supplies structure, guardrails, and audit.
"""

import unittest

from web_research import synthesis
from web_research.deep_research import Citation, Evidence, ResearchPlan, SubQuestion


def _citation(idx: int, **overrides) -> Citation:
    defaults = {
        "id": idx,
        "url": f"https://example.test/src/{idx}",
        "title": f"Source {idx}",
        "source": "wikipedia",
        "published": "2025-03-01",
        "fetched_at": "2026-08-24",
        "quote_count": 3,
        "authors": ["A. Author"],
        "publisher": "Example Press",
    }
    defaults.update(overrides)
    return Citation(**defaults)


def _report(citations=None, evidence=None):
    citations = citations if citations is not None else [
        _citation(1),
        _citation(2),
        _citation(3, source="arxiv", url="https://arxiv.test/abs/1"),
        _citation(4, source="crossref", url="https://doi.test/10.1/2"),
    ]
    evidence = evidence if evidence is not None else {
        "sq_def": [
            Evidence(citation_id=1, quote="Alpha is defined as the baseline measure.", relevance=0.8),
            Evidence(citation_id=2, quote="Early work established the core formalism.", relevance=0.7),
        ],
        "sq_evidence": [
            Evidence(citation_id=3, quote="A recent study reports a 40% improvement.", relevance=0.9),
        ],
    }
    plan = ResearchPlan(
        question="What is alpha and how does it compare?",
        depth="standard",
        sub_questions=[
            SubQuestion(id="sq_def", question="What is alpha?", rationale="r"),
            SubQuestion(id="sq_evidence", question="What does the evidence say about alpha?", rationale="r"),
        ],
    )
    return {
        "question": plan.question,
        "depth": plan.depth,
        "plan": plan.to_dict(),
        "citations": [c.to_dict() for c in citations],
        "evidence": {k: [e.to_dict() for e in v] for k, v in evidence.items()},
    }


class SectionLadderTests(unittest.TestCase):
    """Concept-explanation ladder: foundational -> frontier ordering."""

    def test_build_report_emits_six_agent_spec_sections_in_order(self):
        out = synthesis.build_longform_scaffold(_report())
        expected = [
            "Executive Summary",
            "Concept Deep-Dive",
            "Survey of Perspectives",
            "Cutting-Edge Developments",
            "Annotated Bibliography",
            "Source Register",
        ]
        self.assertEqual(out["sections"], expected)

    def test_ladder_orders_foundational_before_frontier_subquestions(self):
        out = synthesis.build_longform_scaffold(_report())
        ladder = out["section_briefs"]["Concept Deep-Dive"]["ladder"]
        positions = [step["sub_question"] for step in ladder]
        self.assertIn("sq_def", positions)
        self.assertIn("sq_evidence", positions)
        self.assertLess(positions.index("sq_def"), positions.index("sq_evidence"))

    def test_each_ladder_step_carries_citation_ids(self):
        out = synthesis.build_longform_scaffold(_report())
        for step in out["section_briefs"]["Concept Deep-Dive"]["ladder"]:
            self.assertTrue(step["citation_ids"], f"ladder step {step} has no citations")


class PovContrastTests(unittest.TestCase):
    """Steelmanned viewpoint contrast with attribution."""

    def test_conflicting_sources_produce_steelmanned_positions(self):
        report = _report(evidence={
            "sq_def": [
                Evidence(citation_id=1, quote="Alpha scaling clearly improves outcomes.", relevance=0.9),
                Evidence(citation_id=2, quote="Alpha scaling shows no measurable benefit.", relevance=0.9),
            ],
        })
        out = synthesis.build_longform_scaffold(report)
        pov = out["section_briefs"]["Survey of Perspectives"]
        self.assertEqual(pov["status"], "CONFLICTING SOURCES")
        self.assertEqual(len(pov["positions"]), 2)
        for pos in pov["positions"]:
            self.assertTrue(pos["claim"])
            self.assertTrue(pos["citations"])
            self.assertTrue(pos["steelman_prompt"])

    def test_agreeing_sources_marked_consensus_not_conflict(self):
        out = synthesis.build_longform_scaffold(_report())
        pov = out["section_briefs"]["Survey of Perspectives"]
        self.assertNotEqual(pov["status"], "CONFLICTING SOURCES")
        self.assertEqual(pov["status"], "CONSISTENT SOURCES")

    def test_disagreement_summary_names_both_sides_and_open_state(self):
        report = _report(evidence={
            "sq_def": [
                Evidence(citation_id=1, quote="Method A outperforms method B.", relevance=0.9),
                Evidence(citation_id=2, quote="Method B remains superior in practice.", relevance=0.9),
            ],
        })
        out = synthesis.build_longform_scaffold(report)
        summary = out["section_briefs"]["Survey of Perspectives"]["disagreement_summary"]
        self.assertIn("[1]", summary)
        self.assertIn("[2]", summary)


class FrontierFlagTests(unittest.TestCase):
    """Preprint status flagged explicitly on cutting-edge sources."""

    def test_arxiv_sources_flagged_as_preprint_in_frontier_section(self):
        out = synthesis.build_longform_scaffold(_report())
        items = out["section_briefs"]["Cutting-Edge Developments"]["items"]
        arxiv_items = [i for i in items if i.get("preprint")]
        self.assertTrue(arxiv_items, "expected at least one preprint-flagged item")
        for item in arxiv_items:
            self.assertEqual(item["peer_review_status"], "preprint — not peer reviewed")

    def test_recent_non_preprint_source_not_flagged_preprint(self):
        out = synthesis.build_longform_scaffold(_report())
        crossref_item = next(
            i for i in out["section_briefs"]["Cutting-Edge Developments"]["items"] if i["id"] == 4
        )
        self.assertFalse(crossref_item["preprint"])
        self.assertIn("peer reviewed", crossref_item["peer_review_status"])


class AnnotationTests(unittest.TestCase):
    """Every bibliography entry gets relevance/credibility/date annotation."""

    def test_every_reference_has_full_annotation_block(self):
        out = synthesis.build_longform_scaffold(_report())
        bib = out["annotated_bibliography"]
        self.assertEqual(len(bib), 4)
        for entry in bib:
            for key in ("relevance", "credibility", "date"):
                self.assertTrue(entry["annotation"][key], f"entry {entry['id']} missing {key}")

    def test_annotation_relevance_uses_quote_counts(self):
        many = _citation(5, quote_count=9)
        few = _citation(6, quote_count=0)
        out = synthesis.build_longform_scaffold(_report(
            citations=[many, few],
            evidence={"sq_def": [Evidence(citation_id=5, quote="x", relevance=0.9)]},
        ))
        by_id = {e["id"]: e for e in out["annotated_bibliography"]}
        self.assertIn("Central", by_id[5]["annotation"]["relevance"])
        self.assertIn("Peripheral", by_id[6]["annotation"]["relevance"])


class CitationDisciplineTests(unittest.TestCase):
    """Audit: every substantive claim resolves to a numbered reference."""

    def test_clean_paragraph_passes_audit(self):
        text = "Scaling improves outcomes [1]. Later work confirmed this [2]."
        violations = synthesis.audit_citations(text)
        self.assertEqual(violations["uncited_claims"], [])

    def test_unsupported_claim_sentence_flagged_with_line_number(self):
        text = (
            "# Report\n\n"
            "Scaling improves outcomes [1].\n"
            "This is an unsupported claim.\n"
            "Confirmed again [2].\n"
        )
        violations = synthesis.audit_citations(text)
        self.assertEqual(len(violations["uncited_claims"]), 1)
        line_no, snippet = violations["uncited_claims"][0]
        self.assertEqual(line_no, 4)
        self.assertIn("unsupported claim", snippet)

    def test_unknown_reference_numbers_rejected(self):
        text = "Claim one way [1]. Claim another way [99]."
        violations = synthesis.audit_citations(text, known_refs={1})
        self.assertEqual(violations["unknown_refs"], [[99]])

    def test_audit_reports_coverage_stats(self):
        text = "One claim [1]. Another claim [2]. A third [1]."
        stats = synthesis.audit_citations(text, known_refs={1, 2})
        self.assertEqual(stats["total_inline_markers"], 3)
        self.assertEqual(stats["distinct_refs_used"], {1, 2})

    def test_headings_and_short_lines_not_counted_as_claims(self):
        text = "# Title\n\n## Sub\n\n- point [1]\n"
        violations = synthesis.audit_citations(text, known_refs={1})
        self.assertEqual(violations["uncited_claims"], [])


class SourceRegisterTests(unittest.TestCase):
    def test_register_maps_number_to_bibliography_entry(self):
        out = synthesis.build_longform_scaffold(_report())
        reg = out["source_register"]
        self.assertEqual(len(reg), 4)
        self.assertEqual(reg[0], {"n": 1, "bibliography_entry_id": 1})
        self.assertEqual(reg[3]["n"], 4)


class ScaffoldRenderTests(unittest.TestCase):
    def test_render_produces_markdown_with_all_sections_and_guardrails(self):
        out = synthesis.build_longform_scaffold(_report())
        md = synthesis.render_markdown(out)
        for heading in (
            "# Research Report:",
            "## Executive Summary",
            "## Concept Deep-Dive",
            "## Survey of Perspectives",
            "CONFLICTING SOURCES" if False else "## Cutting-Edge Developments",
            "## Annotated Bibliography",
            "## Source Register",
        ):
            self.assertIn(heading, md)
        self.assertIn("not peer reviewed", md)  # preprint flag surfaces in render

    def test_render_flags_conflicts_when_present(self):
        report = _report(evidence={
            "sq_def": [
                Evidence(citation_id=1, quote="X improves outcomes dramatically.", relevance=0.9),
                Evidence(citation_id=2, quote="X shows no measurable effect.", relevance=0.9),
            ],
        })
        md = synthesis.render_markdown(synthesis.build_longform_scaffold(report))
        self.assertIn("**CONFLICTING SOURCES**", md)


class ServerToolContractTests(unittest.IsolatedAsyncioTestCase):
    """The MCP tool layer must expose synthesis + audit."""

    async def test_server_module_exposes_synthesize_tool(self):
        from web_research import server

        tools = {t.name for t in await server.app.list_tools()}
        self.assertIn("synthesize_report", tools)
        self.assertIn("audit_citations", tools)


if __name__ == "__main__":
    unittest.main()
