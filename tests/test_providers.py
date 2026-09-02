import asyncio
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import httpx

from web_research import providers


class FakeResponse:
    def __init__(self, *, json_data=None, text="", error=None, status_code=200, headers=None):
        self._json_data = json_data
        self.text = text
        self._error = error
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.content = text.encode()
        self.is_redirect = False

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


class ProviderParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_brave_normalizes_results_and_bounds_request(self):
        client = FakeClient([FakeResponse(json_data={
            "web": {"results": [
                {"title": " First ", "url": "https://one.test", "description": " Snippet ", "age": "1d"},
                {"title": "Second", "url": "https://two.test", "description": "More"},
            ]}
        })])
        with patch.dict(os.environ, {"BRAVE_API_KEY": "secret"}, clear=False):
            results = await providers.search_brave("mcp", 30, client)

        self.assertEqual([r.to_dict() for r in results], [
            {"title": "First", "url": "https://one.test", "snippet": "Snippet", "source": "brave", "score": 1.0, "published": "1d", "extra": {"age": "1d"}},
            {"title": "Second", "url": "https://two.test", "snippet": "More", "source": "brave", "score": 2.0, "published": None, "extra": {"age": None}},
        ])
        self.assertEqual(client.calls[0][2]["params"]["count"], 20)
        self.assertEqual(client.calls[0][2]["headers"]["X-Subscription-Token"], "secret")

    async def test_tavily_uses_bearer_auth_and_default_advanced_depth(self):
        client = FakeClient([FakeResponse(json_data={"results": [{
            "title": "Paper", "url": "https://paper.test", "content": "x" * 700, "score": 0.91,
        }]})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False):
            results = await providers.search_tavily("query", 30, client)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].snippet, "x" * 600)
        method, url, kwargs = client.calls[0]
        self.assertEqual((method, url), ("POST", "https://api.tavily.com/search"))
        # Modern auth: Bearer header; the key must not leak into the request body.
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        body = kwargs["json"]
        self.assertNotIn("api_key", body)
        # max_results honours the caller up to Tavily's ceiling of 20.
        self.assertEqual(body["max_results"], 20)
        self.assertEqual(body["search_depth"], "advanced")
        self.assertEqual(body["chunks_per_source"], 3)

    async def test_tavily_passes_topic_time_range_and_domain_filters(self):
        client = FakeClient([FakeResponse(json_data={"results": []})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False):
            results = await providers.search_tavily(
                "gpu prices", 5, client,
                topic="news", time_range="week",
                include_domains=["anandtech.com", "tomshardware.com"],
                exclude_domains=["pinterest.com"],
            )

        self.assertEqual(results, [])
        method, url, kwargs = client.calls[0]
        self.assertEqual(url, "https://api.tavily.com/search")
        body = kwargs["json"]
        self.assertEqual(body["topic"], "news")
        self.assertEqual(body["time_range"], "week")
        self.assertEqual(body["include_domains"], ["anandtech.com", "tomshardware.com"])
        self.assertEqual(body["exclude_domains"], ["pinterest.com"])

    async def test_tavily_topic_defaults_to_general_when_unconfigured(self):
        client = FakeClient([FakeResponse(json_data={"results": []})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False):
            await providers.search_tavily("query", 3, client)

        body = client.calls[0][2]["json"]
        self.assertNotIn("topic", body)
        self.assertNotIn("time_range", body)
        self.assertNotIn("include_domains", body)

    async def test_fetch_tavily_extracts_content_and_normalizes_result(self):
        client = FakeClient([FakeResponse(json_data={"results": [{
            "url": "https://example.test",
            "raw_content": "Clean page content",
            "title": "Example Page",
        }]})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch("web_research.providers.validate_url"):
            result = await providers.fetch_tavily("https://example.test", client)

        method, url, kwargs = client.calls[0]
        self.assertEqual((method, url), ("POST", "https://api.tavily.com/extract"))
        # Modern Bearer auth; key never appears in the request body.
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"], {
            "urls": ["https://example.test"],
            "extract_depth": "advanced",
            "format": "markdown",
        })
        self.assertEqual(result["url"], "https://example.test")
        self.assertEqual(result["title"], "Example Page")
        self.assertEqual(result["content"], "Clean page content")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["length"], len(result["content"]))
        self.assertNotIn("error", result)

    async def test_fetch_tavily_truncates_oversized_pages(self):
        client = FakeClient([FakeResponse(json_data={"results": [{
            "url": "https://big.test", "raw_content": "y" * 30_000, "title": "Big",
        }]})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch("web_research.providers.validate_url"):
            result = await providers.fetch_tavily("https://big.test", client)

        self.assertTrue(result["truncated"])
        self.assertIn("[...truncated", result["content"])
        self.assertEqual(result["length"], len(result["content"]))

    async def test_fetch_tavily_returns_error_when_page_not_extracted(self):
        client = FakeClient([FakeResponse(json_data={"results": [], "failed_results": [
            {"url": "https://blocked.test", "error": "extraction failed"},
        ]})])
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch("web_research.providers.validate_url"):
            result = await providers.fetch_tavily("https://blocked.test", client)

        self.assertTrue(result["error"])
        self.assertEqual(result["url"], "https://blocked.test")
        self.assertNotIn("content", result)

    async def test_fetch_tavily_requires_api_key(self):
        client = FakeClient([])
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = await providers.fetch_tavily("https://example.test", client)

        self.assertIn("TAVILY_API_KEY", result["error"])
        self.assertEqual(client.calls, [])

    async def test_wikipedia_strips_markup_and_builds_canonical_url(self):
        client = FakeClient([FakeResponse(json_data={"query": {"search": [{
            "title": "Ada Lovelace", "snippet": "An <span class='searchmatch'>early</span> programmer", "score": 80,
        }]}})])
        results = await providers.search_wikipedia("Ada", 2, client)
        self.assertEqual(results[0].title, "Ada Lovelace")
        self.assertEqual(results[0].snippet, "An early programmer")
        self.assertEqual(results[0].url, "https://en.wikipedia.org/wiki/Ada_Lovelace")
        self.assertEqual(results[0].score, 0.8)

    async def test_arxiv_parses_atom_authors_and_date(self):
        xml = """<feed xmlns='http://www.w3.org/2005/Atom'>
          <entry><title> A\n useful paper </title><summary> Abstract\n text </summary>
            <id>https://arxiv.org/abs/1234.5678</id><published>2025-06-07T00:00:00Z</published>
            <author><name>A. Author</name></author>
          </entry>
        </feed>"""
        results = await providers.search_arxiv("attention", 2, FakeClient([FakeResponse(text=xml)]))
        self.assertEqual(results[0].title, "A useful paper")
        self.assertEqual(results[0].snippet, "Abstract text")
        self.assertEqual(results[0].published, "2025-06-07")
        self.assertEqual(results[0].authors, ["A. Author"])
        self.assertEqual(results[0].publisher, "arXiv")

    async def test_hacker_news_uses_story_fallback_url_and_metadata(self):
        client = FakeClient([FakeResponse(json_data={"hits": [{
            "objectID": "42", "story_text": "A story", "num_comments": 3, "points": 10,
            "created_at": "2025-01-02T03:04:05Z", "_highlightResult": {},
        }]})])
        results = await providers.search_hn("mcp", 2, client)
        self.assertEqual(results[0].url, "https://news.ycombinator.com/item?id=42")
        self.assertEqual(results[0].snippet, "A story")
        self.assertEqual(results[0].published, "2025-01-02")
        self.assertEqual(results[0].extra, {"points": 10, "comments": 3})

    async def test_stackexchange_strips_excerpt_markup_and_names_site(self):
        client = FakeClient([FakeResponse(json_data={"items": [{
            "title": "How?", "link": "https://stackoverflow.com/q/1", "excerpt": "Use <b>asyncio</b>",
            "score": 4, "is_answered": True, "answer_count": 2, "tags": ["python"],
        }]})])
        results = await providers.search_stackexchange("asyncio", 2, client, site="stackoverflow")
        self.assertEqual(results[0].snippet, "Use asyncio")
        self.assertEqual(results[0].source, "stackexchange:stackoverflow")
        self.assertEqual(client.calls[0][2]["params"]["site"], "stackoverflow")

    async def test_crossref_uses_abstract_or_container_and_formats_partial_dates(self):
        client = FakeClient([FakeResponse(json_data={"message": {"items": [
            {"title": ["One"], "DOI": "10.1/one", "abstract": "<jats:p>Abstract</jats:p>",
             "published-print": {"date-parts": [[2024, 3, 2]]}, "type": "article", "is-referenced-by-count": 7},
            {"title": ["Two"], "URL": "https://doi.test/two", "container-title": ["Journal"]},
        ]}})])
        results = await providers.search_crossref("papers", 2, client)
        self.assertEqual(results[0].url, "https://doi.org/10.1/one")
        self.assertEqual(results[0].snippet, "Abstract")
        self.assertEqual(results[0].published, "2024-03-02")
        self.assertEqual(results[1].snippet, "Published in: Journal")
        self.assertIsNone(results[1].published)

    async def test_fetch_jina_parses_metadata_and_truncates_body(self):
        body = "Title: Example\nURL Source: https://source.test/\nPublished Time: today\n\n" + ("a" * 20_010)
        client = FakeClient([FakeResponse(text=body, headers={"content-type": "text/markdown"})])
        with patch("web_research.providers.validate_url"):
            result = await providers.fetch_jina("https://example.test", client)
        self.assertEqual(result["title"], "Example")
        self.assertEqual(result["url"], "https://source.test/")
        self.assertTrue(result["truncated"])
        self.assertIn("full content was", result["content"])
        self.assertEqual(result["length"], len(result["content"]))


class ProviderFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_failures_log_only_to_stderr(self):
        functions = (
            providers.search_brave, providers.search_tavily, providers.search_wikipedia,
            providers.search_arxiv, providers.search_hn,
            providers.search_stackexchange, providers.search_crossref,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, {"BRAVE_API_KEY": "secret", "TAVILY_API_KEY": "secret"}, clear=False):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                for function in functions:
                    client = FakeClient([FakeResponse(error=httpx.HTTPError("offline"))])
                    args = ("q", 1, client)
                    if function is providers.search_stackexchange:
                        args += ("stackoverflow",)
                    self.assertEqual(await function(*args), [])

        self.assertEqual(stdout.getvalue(), "")
        events = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual(
            [(event["provider"], event["event"], event["status"]) for event in events],
            [
                ("brave", "provider.failed", "error"),
                ("tavily", "provider.failed", "error"),
                ("wikipedia", "provider.failed", "error"),
                ("arxiv", "provider.failed", "error"),
                ("hackernews", "provider.failed", "error"),
                ("stackexchange", "provider.failed", "error"),
                ("crossref", "provider.failed", "error"),
            ],
        )

    async def test_optional_keyless_providers_return_empty_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            client = FakeClient([])
            self.assertEqual(await providers.search_brave("q", 1, client), [])
            self.assertEqual(await providers.search_tavily("q", 1, client), [])
            self.assertEqual(client.calls, [])

    async def test_all_network_provider_failures_degrade_to_empty_results(self):
        functions = (
            providers.search_wikipedia, providers.search_arxiv, providers.search_hn,
            providers.search_stackexchange, providers.search_crossref,
        )
        for function in functions:
            with self.subTest(function=function.__name__):
                client = FakeClient([FakeResponse(error=httpx.HTTPError("offline"))])
                args = ("q", 1, client)
                if function is providers.search_stackexchange:
                    args += ("stackoverflow",)
                self.assertEqual(await function(*args), [])

    async def test_invalid_payloads_degrade_to_empty_results(self):
        for function in (providers.search_wikipedia, providers.search_arxiv, providers.search_hn, providers.search_crossref):
            with self.subTest(function=function.__name__):
                response = FakeResponse(json_data=ValueError("bad payload")) if function is not providers.search_arxiv else FakeResponse(text="<not xml")
                self.assertEqual(await function("q", 1, FakeClient([response])), [])

    async def test_fetch_jina_failure_returns_structured_error(self):
        result = await providers.fetch_jina("https://example.test", FakeClient([FakeResponse(error=httpx.HTTPError("offline"))]))
        self.assertEqual(result["url"], "https://example.test")
        self.assertIn("fetch failed", result["error"])


class NormalizationTests(unittest.TestCase):
    def test_provider_registry_selects_by_name_and_capability(self):
        registry = providers.ProviderRegistry()
        searcher = type("Searcher", (), {"name": "search", "capabilities": frozenset({providers.Capability.SEARCH})})()
        fetcher = type("Fetcher", (), {"name": "fetch", "capabilities": frozenset({providers.Capability.FETCH})})()
        registry.register(searcher)
        registry.register(fetcher)

        self.assertIs(registry.get("search", providers.Capability.SEARCH), searcher)
        self.assertEqual(registry.providers_for(providers.Capability.FETCH), (fetcher,))
        with self.assertRaises(ValueError):
            registry.get("search", providers.Capability.FETCH)
        with self.assertRaises(KeyError):
            registry.get("missing")
        with self.assertRaises(ValueError):
            registry.register(searcher)

    def test_canonical_url_removes_tracking_query_fragment_and_trailing_slash(self):
        self.assertEqual(
            providers._canonical_url("HTTPS://Example.TEST/path/?utm_source=x&id=7#section"),
            "https://example.test/path?id=7",
        )

    def test_canonical_url_normalizes_query_order_and_default_port(self):
        self.assertEqual(
            providers._canonical_url("https://Example.TEST:443/path?b=2&a=1&utm_medium=email"),
            providers._canonical_url("https://example.test/path?a=1&b=2"),
        )

    def test_merge_results_deduplicates_keeps_longest_snippet_and_boosts_score(self):
        first = providers.Result("First", "https://example.test/?utm_source=x", "short", "brave", score=1)
        second = providers.Result("Second", "https://example.test/", "a much longer snippet", "tavily", score=2)
        results = providers.merge_results([first], [second])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "First")
        self.assertEqual(results[0].snippet, "a much longer snippet")
        self.assertEqual(results[0].score, 3)
        self.assertEqual(results[0].extra["also_found_in"], ["tavily"])

    def test_merge_results_ranks_distinct_urls_by_composite_score(self):
        low = providers.Result("Low", "https://example.test/low", "low", "wikipedia", score=0.5)
        high = providers.Result("High", "https://example.test/high", "high", "brave", score=2.0)

        results = providers.merge_results([low, high])

        self.assertEqual([result.title for result in results], ["High", "Low"])

    def test_merge_results_preserves_conflicting_source_provenance_without_mutating_inputs(self):
        first = providers.Result(
            "Canonical title", "https://example.test/article?b=2&a=1", "short summary", "wikipedia", score=0.8
        )
        conflicting = providers.Result(
            "Conflicting title", "https://example.test/article?a=1&b=2#details", "longer independent summary", "brave", score=0.6
        )

        results = providers.merge_results([first], [conflicting])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Canonical title")
        self.assertEqual(results[0].url, "https://example.test/article?a=1&b=2")
        self.assertEqual(results[0].snippet, "longer independent summary")
        self.assertEqual(results[0].extra["provenance"], [
            {"source": "wikipedia", "url": "https://example.test/article?b=2&a=1", "score": 0.8},
            {"source": "brave", "url": "https://example.test/article?a=1&b=2#details", "score": 0.6},
        ])
        self.assertEqual(first.url, "https://example.test/article?b=2&a=1")
        self.assertEqual(first.extra, {})


if __name__ == "__main__":
    unittest.main()
