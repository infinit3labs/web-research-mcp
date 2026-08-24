"""Small opt-in protocol checks that exercise real provider/network boundaries."""

from __future__ import annotations

import os
import unittest

from web_research import server


LIVE_INTEGRATION_ENABLED = os.environ.get("WEB_RESEARCH_RUN_LIVE_INTEGRATION") == "1"


@unittest.skipUnless(
    LIVE_INTEGRATION_ENABLED,
    "set WEB_RESEARCH_RUN_LIVE_INTEGRATION=1 to enable network-backed checks",
)
class LiveProtocolSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_tools_list_contains_registered_tools(self):
        tools = {tool.name for tool in await server.app.list_tools()}

        self.assertIn("search_wikipedia", tools)
        self.assertIn("fetch_url", tools)
        self.assertIn("research", tools)

    async def test_live_wikipedia_call_returns_protocol_text(self):
        response = await server.app.call_tool(
            "search_wikipedia", {"query": "Model Context Protocol", "max_results": 1}
        )

        self.assertFalse(response.is_error)
        self.assertEqual(len(response.content), 1)
        self.assertEqual(response.content[0].type, "text")
        self.assertIn("Results for", response.content[0].text)


if __name__ == "__main__":
    unittest.main()
