import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from web_research import providers, server
from web_research import __version__


class AsyncClientContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ProtocolAndSchemaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Tool calls go through providers.cached_search/cached_fetch; without a
        # reset a result cached by one test could leak into another test that
        # reuses the same query against a different mock.
        providers.reset_caches()

    def test_runtime_version_matches_package_metadata(self):
        self.assertEqual(__version__, "0.2.0")

    async def test_tools_list_exposes_complete_names_and_generated_constraints(self):
        tools = {tool.name: tool for tool in await server.app.list_tools()}
        expected = {
            "search_web", "fetch_url", "search_wikipedia", "search_academic", "search_news",
            "search_stackexchange", "search_scholar_meta", "plan_research", "extract_evidence", "research",
            "synthesize_report", "audit_citations",
        }
        self.assertEqual(set(tools), expected)
        self.assertEqual(tools["search_web"].input_schema["properties"]["max_results"]["maximum"], 30)
        self.assertEqual(tools["search_web"].input_schema["properties"]["max_results"]["minimum"], 1)
        self.assertEqual(tools["extract_evidence"].input_schema["required"], ["url", "question"])
        self.assertEqual(tools["search_stackexchange"].input_schema["properties"]["site"]["default"], "stackoverflow")

    async def test_call_tool_dispatches_to_provider_and_returns_text_content(self):
        fake = providers.Result("MCP", "https://example.test", "snippet", "wikipedia", score=0.5)
        provider = AsyncMock(return_value=[fake])
        context = AsyncClientContext()
        with patch.object(server.providers, "search_wikipedia", provider), patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("search_wikipedia", {"query": "MCP", "max_results": 1})

        provider.assert_awaited_once()
        self.assertEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")
        self.assertIn("# Results for: MCP", result.content[0].text)
        self.assertIn("https://example.test", result.content[0].text)

    async def test_tool_errors_are_rendered_as_protocol_text_for_invalid_research_depth(self):
        result = await server.app.call_tool("plan_research", {"question": "q", "depth": "invalid"})
        self.assertEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")
        self.assertTrue(result.content[0].text.startswith("Error:"))

    async def test_extract_evidence_returns_machine_readable_json_and_canonical_url(self):
        fetched = {"title": "Page", "url": "https://page.test", "content": "MCP is a protocol.", "length": 17, "truncated": False}
        context = AsyncClientContext()
        with patch.object(server, "_fetch_provider", AsyncMock(return_value=fetched)), patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("extract_evidence", {
                "url": "https://page.test/?utm_source=test#section", "question": "What is MCP?", "max_passages": 1,
            })
        payload = json.loads(result.content[0].text)
        self.assertEqual(payload["url"], "https://page.test")
        self.assertEqual(payload["title"], "Page")
        self.assertIn("passages", payload)


class FetchFallbackTests(unittest.IsolatedAsyncioTestCase):

    async def test_fetch_url_falls_back_to_tavily_when_jina_errors_and_key_present(self):
        jina = AsyncMock(return_value={"url": "https://walled.test", "error": "fetch failed: bot wall"})
        tavily = AsyncMock(return_value={
            "url": "https://walled.test", "title": "Recovered",
            "content": "Extracted via Tavily.", "length": 21, "truncated": False,
        })
        context = AsyncClientContext()
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch.object(server, "_fetch_provider", jina), \
                patch.object(providers, "fetch_tavily", tavily), \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("fetch_url", {"url": "https://walled.test"})

        jina.assert_awaited_once()
        tavily.assert_awaited_once()
        self.assertIn("# Recovered", result.content[0].text)
        self.assertIn("Extracted via Tavily.", result.content[0].text)

    async def test_fetch_url_keeps_jina_error_when_no_tavily_key(self):
        jina = AsyncMock(return_value={"url": "https://walled.test", "error": "fetch failed: bot wall"})
        tavily = AsyncMock(return_value={"url": "https://walled.test", "content": "should not be reached"})
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        context = AsyncClientContext()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(server, "_fetch_provider", jina), \
                patch.object(providers, "fetch_tavily", tavily), \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("fetch_url", {"url": "https://walled.test"})

        jina.assert_awaited_once()
        tavily.assert_not_awaited()
        self.assertTrue(result.content[0].text.startswith("Error:"))

    async def test_fetch_url_reports_combined_error_when_both_fetchers_fail(self):
        jina = AsyncMock(return_value={"url": "https://walled.test", "error": "jina down"})
        tavily = AsyncMock(return_value={"url": "https://walled.test", "error": "tavily extract failed"})
        context = AsyncClientContext()
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch.object(server, "_fetch_provider", jina), \
                patch.object(providers, "fetch_tavily", tavily), \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("fetch_url", {"url": "https://walled.test"})

        self.assertTrue(result.content[0].text.startswith("Error:"))
        self.assertIn("jina", result.content[0].text)
        self.assertIn("tavily", result.content[0].text)


class SearchNewsTavilyEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    """search_news enriches Hacker News with Tavily's news topic when available."""

    async def test_search_news_blends_hackernews_with_tavily_news_results(self):
        hn = [
            providers.Result("HN Story A", "https://a.test/story-a", "snippet a", "hackernews", score=10.0),
            providers.Result("HN Story B", "https://b.test/story-b", "snippet b", "hackernews", score=5.0),
        ]
        tav = [
            providers.Result("Tavily Story C", "https://c.test/story-c", "snippet c", "tavily", score=0.9,
                             published="2026-08-20"),
            providers.Result("Shared Story B", "https://b.test/story-b", "longer snippet for story b from Tavily",
                             "tavily", score=0.8),
        ]
        context = AsyncClientContext()
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch.object(server.providers, "search_hn", AsyncMock(return_value=hn)), \
                patch.object(server.providers, "search_tavily", AsyncMock(return_value=tav)) as mock_tavily, \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("search_news", {"query": "mcp protocol", "max_results": 5})

        # Tavily is queried with the news topic and recency window.
        kwargs = mock_tavily.call_args.kwargs
        self.assertEqual(kwargs.get("topic"), "news")
        self.assertEqual(kwargs.get("time_range"), "week")

        text = result.content[0].text
        self.assertIn("# Results for: mcp protocol", text)
        self.assertIn("https://c.test/story-c", text)   # Tavily-only result present
        self.assertIn("https://a.test/story-a", text)   # HN result preserved

    async def test_search_news_still_works_without_tavily_key(self):
        hn = [providers.Result("Only HN", "https://hn.test/only", "snippet", "hackernews", score=3.0)]
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        tavily = AsyncMock(return_value=[providers.Result("X", "https://x.test", "y", "tavily")])
        context = AsyncClientContext()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(server.providers, "search_hn", AsyncMock(return_value=hn)), \
                patch.object(server.providers, "search_tavily", tavily), \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("search_news", {"query": "mcp", "max_results": 5})

        tavily.assert_not_awaited()
        self.assertIn("https://hn.test/only", result.content[0].text)

    async def test_search_news_survives_tavily_failure(self):
        hn = [providers.Result("Only HN", "https://hn.test/only", "snippet", "hackernews", score=3.0)]
        context = AsyncClientContext()
        with patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False), \
                patch.object(server.providers, "search_hn", AsyncMock(return_value=hn)), \
                patch.object(server.providers, "search_tavily", AsyncMock(side_effect=RuntimeError("boom"))), \
                patch.object(server, "_new_client", AsyncMock(return_value=context)):
            result = await server.app.call_tool("search_news", {"query": "mcp", "max_results": 5})

        self.assertIn("https://hn.test/only", result.content[0].text)


if __name__ == "__main__":
    unittest.main()
