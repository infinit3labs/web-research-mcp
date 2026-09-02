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


if __name__ == "__main__":
    unittest.main()
