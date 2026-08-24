import asyncio
import json
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
    def test_runtime_version_matches_package_metadata(self):
        self.assertEqual(__version__, "0.2.0")

    async def test_tools_list_exposes_complete_names_and_generated_constraints(self):
        tools = {tool.name: tool for tool in await server.app.list_tools()}
        expected = {
            "search_web", "fetch_url", "search_wikipedia", "search_academic", "search_news",
            "search_stackexchange", "search_scholar_meta", "plan_research", "extract_evidence", "research",
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


if __name__ == "__main__":
    unittest.main()
