"""Deterministic MCP tool-list, schema, and call-result contract matrix."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from web_research import providers, server


class AsyncClientContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class McpContractMatrix(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        providers.reset_caches()

    async def test_tools_list_has_stable_names_and_machine_readable_schemas(self):
        tools = {tool.name: tool for tool in await server.app.list_tools()}
        expected = {
            "search_web", "fetch_url", "search_wikipedia", "search_academic", "search_news",
            "search_stackexchange", "search_scholar_meta", "plan_research", "extract_evidence", "research",
        }
        self.assertEqual(set(tools), expected)
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertEqual(tool.input_schema.get("type"), "object")
                self.assertIsInstance(tool.input_schema.get("properties"), dict)
                self.assertTrue(tool.description)

        self.assertEqual(tools["search_web"].input_schema["properties"]["max_results"]["minimum"], 1)
        self.assertEqual(tools["search_web"].input_schema["properties"]["max_results"]["maximum"], 30)
        self.assertEqual(tools["extract_evidence"].input_schema["required"], ["url", "question"])

    async def test_registered_provider_call_returns_one_text_content_block(self):
        result = providers.Result("MCP", "https://example.test/mcp", "A snippet", "wikipedia")
        context = AsyncClientContext()
        with patch.object(server, "_new_client", AsyncMock(return_value=context)), patch.object(
            server, "_search_provider", AsyncMock(return_value=[result])
        ):
            response = await server.app.call_tool("search_wikipedia", {"query": "MCP", "max_results": 1})

        self.assertFalse(response.is_error)
        self.assertEqual(len(response.content), 1)
        self.assertEqual(response.content[0].type, "text")
        self.assertIn("https://example.test/mcp", response.content[0].text)

    async def test_json_tools_return_parseable_payloads(self):
        context = AsyncClientContext()
        fetched = {"title": "Page", "url": "https://example.test/page", "content": "MCP is a protocol.", "length": 18, "truncated": False}
        with patch.object(server, "_new_client", AsyncMock(return_value=context)), patch.object(
            server, "_fetch_provider", AsyncMock(return_value=fetched)
        ):
            evidence_response = await server.app.call_tool(
                "extract_evidence", {"url": "https://example.test/page", "question": "What is MCP?", "max_passages": 1}
            )
        payload = json.loads(evidence_response.content[0].text)
        self.assertEqual(payload["url"], "https://example.test/page")
        self.assertIn("passages", payload)

    async def test_invalid_tool_arguments_are_rendered_as_text_errors(self):
        response = await server.app.call_tool("plan_research", {"question": "q", "depth": "not-a-depth"})
        self.assertEqual(len(response.content), 1)
        self.assertEqual(response.content[0].type, "text")
        self.assertTrue(response.content[0].text.startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
