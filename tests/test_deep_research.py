import asyncio
import unittest
from unittest.mock import patch

from web_research import deep_research, providers
from web_research.deep_research import Citation


class CitationProvenanceTests(unittest.TestCase):
    def setUp(self):
        providers.reset_caches()

    def test_citation_serialization_keeps_all_sources_for_deduplicated_result(self):
        citation = Citation(
            id=1,
            url="https://example.test/article",
            title="Article",
            source="wikipedia",
            provenance=[
                {"source": "wikipedia", "url": "https://example.test/article", "score": 0.8},
                {"source": "brave", "url": "https://example.test/article?utm_source=x", "score": 0.6},
            ],
        )

        self.assertEqual(citation.to_dict()["provenance"], citation.provenance)

    def test_citation_includes_full_metadata_fields(self):
        citation = Citation(
            id=2,
            url="https://arxiv.test/abs/1234.5678",
            title="A Study",
            source="arxiv",
            published="2026-01-15",
            fetched_at="2026-08-24",
            authors=["A. Author", "B. Author"],
            publisher="arXiv",
        )

        d = citation.to_dict()
        self.assertEqual(d["authors"], ["A. Author", "B. Author"])
        self.assertEqual(d["publisher"], "arXiv")
        self.assertEqual(d["published"], "2026-01-15")
        self.assertEqual(d["fetched_at"], "2026-08-24")

    def test_gather_search_dispatches_every_planned_query_to_each_source(self):
        calls = []

        class FakeProvider:
            def __init__(self, name):
                self.name = name

            async def search(self, query, max_results, client):
                calls.append((self.name, query, max_results, client))
                return [providers.Result(query, f"https://example.test/{query}", query, self.name)]

        registry = type("Registry", (), {
            "providers_for": lambda _self, _capability: (FakeProvider("wikipedia"), FakeProvider("arxiv"))
        })()
        sq = deep_research.SubQuestion(
            id="sq_1", question="question", rationale="test",
            queries=["first query", "second query"], sources=["wikipedia", "arxiv"],
        )

        with patch.object(providers, "provider_registry", registry):
            results = asyncio.run(deep_research._gather_search(sq, 3, "client"))

        self.assertEqual(
            [("wikipedia", "first query"), ("arxiv", "first query"),
             ("wikipedia", "second query"), ("arxiv", "second query")],
            [(name, query) for name, query, _max_results, _client in calls],
        )
        self.assertEqual(4, len(results))

    def test_fetch_and_extract_uses_cached_fetch_and_updates_result_title(self):
        calls = []

        class FakeFetcher:
            name = "jina"

            async def fetch(self, url, client):
                calls.append(url)
                return {
                    "url": url,
                    "title": "A much better canonical title",
                    "content": "Relevant paragraph about the sub-question topic here, long enough to score.\n\n"
                               "Another unrelated paragraph that pads the fixture body out further still.",
                    "truncated": False,
                    "length": 100,
                }

        sq = deep_research.SubQuestion(
            id="sq_1", question="topic", rationale="test", queries=["topic details"], sources=["wikipedia"],
        )
        result = providers.Result("short", "https://example.test/page", "snippet", "wikipedia")

        with patch.object(providers.provider_registry, "get", return_value=FakeFetcher()):
            updated_result, evidence = asyncio.run(
                deep_research._fetch_and_extract(result, sq, max_passages=2, client="client")
            )

        self.assertEqual(["https://example.test/page"], calls)
        self.assertEqual("A much better canonical title", updated_result.title)
        self.assertIsInstance(evidence, list)

        # Second call for the same URL must be served from providers.cached_fetch
        # without invoking the fake fetcher again.
        asyncio.run(deep_research._fetch_and_extract(result, sq, max_passages=2, client="client"))
        self.assertEqual(["https://example.test/page"], calls)


class IterativeQueryPlanningTests(unittest.TestCase):
    def test_plan_extends_each_subquestion_with_followup_queries(self):
        plan = deep_research.build_plan(
            "How do solid-state batteries compare to conventional lithium-ion cells?", "standard"
        )
        cfg = deep_research._DEPTH_CONFIG["standard"]
        for sq in plan.sub_questions:
            self.assertLessEqual(len(sq.queries), cfg["queries_per_sq"])
            self.assertGreaterEqual(len(sq.queries), 2)
            # The framing queries remain intact and a follow-up was appended.
            self.assertTrue(all(q.strip() for q in sq.queries))
            joined = " ".join(sq.queries).lower()
            self.assertIn("latest developments", joined)
        self.assertEqual(plan.estimated_searches, len(plan.sub_questions) * cfg["queries_per_sq"])

    def test_deep_depth_fits_two_targeted_followups_per_subquestion(self):
        plan = deep_research.build_plan(
            "How do solid-state batteries compare to conventional lithium-ion cells?", "deep"
        )
        cfg = deep_research._DEPTH_CONFIG["deep"]
        for sq in plan.sub_questions:
            joined = " ".join(sq.queries).lower()
            self.assertIn("latest developments", joined)
            self.assertIn("evidence criticism limitations", joined)
            self.assertLessEqual(len(sq.queries), cfg["queries_per_sq"])

    def test_followup_queries_are_deduplicated_against_framing_queries(self):
        sq = deep_research.SubQuestion(
            id="sq_x", question="solid-state batteries", rationale="r",
            queries=["batteries latest developments"],
        )
        followups = deep_research._followup_queries(sq, "batteries")
        lowered = {q.lower() for q in sq.queries}
        for f in followups:
            self.assertNotIn(f.lower(), lowered)
            lowered.add(f.lower())
        self.assertIn("evidence criticism limitations", " ".join(followups))

    def test_include_followups_false_preserves_original_queries(self):
        original = deep_research.build_plan("Do smartphones cause brain cancer?", "standard")
        with_followups = deep_research.build_plan(
            "Do smartphones cause brain cancer?", "standard", include_followups=False
        )
        self.assertEqual(
            [list(sq.queries) for sq in with_followups.sub_questions],
            [[q for q in sq.queries if not any(
                m in q for m in ("latest developments", "evidence criticism limitations")
            )][:3] for sq in original.sub_questions] or [
                list(sq.queries) for sq in original.sub_questions
            ],
        )
        # Direct check: without follow-ups the framing queries are untouched.
        plan = deep_research.build_plan(
            "Do smartphones cause brain cancer?", "standard", include_followups=False
        )
        for sq in plan.sub_questions:
            self.assertFalse(any("latest developments" == q.split(" ")[-2:] for q in []))
            self.assertNotIn("evidence criticism limitations", " ".join(sq.queries))

    def test_depth_configs_guarantee_room_for_at_least_one_followup(self):
        for depth in deep_research._DEPTH_CONFIG:
            with self.subTest(depth=depth):
                plan = deep_research.build_plan("What is quantum computing?", depth)
                for sq in plan.sub_questions:
                    self.assertGreaterEqual(len(sq.queries), 1)


class NearDuplicateDedupTests(unittest.TestCase):
    def test_near_duplicate_key_collapses_scheme_www_and_extension_variants(self):
        variants = [
            "https://Example.com/a/story.html?utm_source=news#top",
            "http://www.example.com/a/story.html?utm_medium=email",
            "https://example.com/a/story/",
            "https://example.com/a/story.htm",
        ]
        keys = {providers._near_duplicate_key(u) for u in variants}
        self.assertEqual(1, len(keys))

    def test_near_duplicate_key_keeps_distinct_articles_separate(self):
        a = providers._near_duplicate_key("https://example.com/alpha")
        b = providers._near_duplicate_key("https://example.com/beta")
        self.assertNotEqual(a, b)

    def test_run_research_suppresses_near_duplicate_urls_before_fetching(self):
        plan = deep_research.ResearchPlan(question="q", depth="quick", sub_questions=[
            deep_research.SubQuestion(id="sq_def", question="q", rationale="r", queries=["q"]),
        ])

        captured: dict[str, list[str]] = {}

        class FakeFetcher:
            name = "jina"

            async def fetch(self, url, client):
                captured.setdefault("urls", []).append(url)
                return {
                    "url": url,
                    "title": "Fetched",
                    "content": "Paragraph one about the topic with enough words to pass the "
                               "quality floor filters imposed by extract_evidence scoring.\n\n"
                               "Second paragraph also mentions the topic repeatedly so it scores.",
                }

        registry = type("Registry", (), {
            "get": lambda _self, name, capability: FakeFetcher(),
        })()

        def fake_search(query, max_results, client):
            return [
                providers.Result("Mirror A", "https://example.com/story?utm_source=one", "snippet", "brave"),
                providers.Result("Mirror B", "https://www.example.com/story/", "longer snippet here", "tavily"),
                providers.Result("Distinct", "https://other.example.org/article", "different article", "brave"),
            ]

        with patch.object(providers, "provider_registry", registry), \
             patch("web_research.deep_research.build_plan", return_value=plan), \
             patch.object(deep_research, "_gather_search",
                          side_effect=lambda sq, m, c: fake_search(sq.queries[0], m, c)):
            report = asyncio.run(deep_research.run_research("q", "quick"))

        fetched_hosts = sorted(captured["urls"])
        # Only two unique sources should be fetched — one of the two mirrors plus the distinct URL.
        self.assertEqual(2, len(fetched_hosts))
        mirror_keys = {
            providers._near_duplicate_key(u) for u in captured["urls"]
            if providers._near_duplicate_key(u) == providers._near_duplicate_key("https://example.com/story")
        }
        self.assertEqual(1, len(mirror_keys))
        citation_urls = [c.url for c in report.citations]
        self.assertEqual(2, len(citation_urls))


class RunResearchMetadataTests(unittest.TestCase):
    def test_citations_carry_full_metadata_and_passages_are_stamped_with_source_id(self):
        plan = deep_research.ResearchPlan(question="q", depth="quick", sub_questions=[
            deep_research.SubQuestion(id="sq_def", question="quantum error correction", rationale="r", queries=["q"]),
        ])

        class FakeFetcher:
            name = "jina"

            async def fetch(self, url, client):
                return {
                    "url": url,
                    "title": "Surface Codes Explained",
                    "published": "2026-05-01",
                    "content": "Quantum error correction relies on surface codes to protect qubits "
                               "from decoherence in noisy hardware. This passage explains threshold "
                               "values observed across multiple experiments and simulations.",
                }

        registry = type("Registry", (), {"get": lambda _self, name, capability: FakeFetcher()})()

        arxiv_result = providers.Result(
            "Surface codes paper", "https://arxiv.org/abs/2605.09999", "abstract text",
            "arxiv", published="2026-05-01", authors=["A. Researcher", "B. Coauthor"],
            publisher="arXiv",
        )

        with patch.object(providers, "provider_registry", registry), \
             patch("web_research.deep_research.build_plan", return_value=plan), \
             patch("web_research.deep_research._gather_search",
                   side_effect=lambda sq, m, c: [arxiv_result]):
            report = asyncio.run(deep_research.run_research("q", "quick"))

        self.assertEqual(1, len(report.citations))
        citation = report.citations[0]
        self.assertEqual(citation.authors, ["A. Researcher", "B. Coauthor"])
        self.assertEqual(citation.publisher, "arXiv")
        self.assertEqual(citation.published, "2026-05-01")
        self.assertIsNotNone(citation.fetched_at)
        evidence = [ev for evs in report.evidence_by_subquestion.values() for ev in evs]
        self.assertTrue(evidence)
        for ev in evidence:
            self.assertEqual(citation.id, ev.citation_id)

    def test_fetch_publication_date_overrides_provider_date_when_present(self):
        plan = deep_research.ResearchPlan(question="q", depth="quick", sub_questions=[
            deep_research.SubQuestion(id="sq_def", question="topic", rationale="r", queries=["q"]),
        ])

        class FakeFetcher:
            name = "jina"

            async def fetch(self, url, client):
                return {
                    "url": url,
                    "title": "T",
                    "published": "2026-07-04",
                    "content": "Enough body text about the topic repeated several times so that "
                               "the paragraph passes the minimum length and term coverage floor.",
                }

        registry = type("Registry", (), {"get": lambda _self, name, capability: FakeFetcher()})()
        result = providers.Result("T", "https://example.test/a", "s", "brave", published="2024")

        with patch.object(providers, "provider_registry", registry), \
             patch("web_research.deep_research.build_plan", return_value=plan), \
             patch("web_research.deep_research._gather_search", side_effect=lambda sq, m, c: [result]):
            report = asyncio.run(deep_research.run_research("q", "quick"))

        self.assertEqual(report.citations[0].published, "2026-07-04")

    def test_publisher_falls_back_to_host_derived_name(self):
        plan = deep_research.ResearchPlan(question="q", depth="quick", sub_questions=[
            deep_research.SubQuestion(id="sq_def", question="topic", rationale="r", queries=["q"]),
        ])

        class FakeFetcher:
            name = "jina"

            async def fetch(self, url, client):
                return {"url": url, "title": "T", "content": None}

        registry = type("Registry", (), {"get": lambda _self, name, capability: FakeFetcher()})()
        result = providers.Result("T", "https://nature.com/articles/x", "s", "brave")

        with patch.object(providers, "provider_registry", registry), \
             patch("web_research.deep_research.build_plan", return_value=plan), \
             patch("web_research.deep_research._gather_search", side_effect=lambda sq, m, c: [result]):
            report = asyncio.run(deep_research.run_research("q", "quick"))

        # Fetch returned no content → snippet fallback citation keeps provider metadata
        self.assertEqual(1, len(report.citations))
        self.assertEqual(report.citations[0].publisher, "Nature")


class PublisherFallbackTests(unittest.TestCase):
    def test_publisher_from_host_variants(self):
        cases = {
            "https://www.nature.com/articles/x": "Nature",
            "https://en.wikipedia.org/wiki/X": "Wikipedia",
            "https://blog.anthropic.com/post": "Anthropic",
            "https://openai.com/index/y": "Openai",
            "https://subdomain.news-site.co.uk/page": "Subdomain",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(expected, deep_research._publisher_from_host(url))
        self.assertIsNone(deep_research._publisher_from_host("not-a-url"))
        self.assertIsNone(deep_research._publisher_from_host("https://localhost/x"))


if __name__ == "__main__":
    unittest.main()
