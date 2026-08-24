"""Deterministic provider contract matrix backed by checked-in response fixtures."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import httpx

from web_research import providers


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


class FixtureResponse:
    def __init__(self, *, text: str = "", payload: object = None, error: Exception | None = None):
        self.text = text
        self.content = text.encode()
        self._payload = payload
        self._error = error
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "text/markdown"})
        self.is_redirect = False

    def json(self):
        if self._error:
            raise self._error
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self._error:
            raise self._error


class FixtureClient:
    def __init__(self, response: FixtureResponse):
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    async def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.response

    async def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.response


def fixture_json(name: str) -> object:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class ProviderParsingFixtureMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_required_search_providers_parse_fixture_responses(self):
        cases = (
            ("brave", providers.search_brave, fixture_json("brave"), "brave", "Model Context Protocol"),
            ("tavily", providers.search_tavily, fixture_json("tavily"), "tavily", "Retrieval augmented generation"),
            ("wikipedia", providers.search_wikipedia, fixture_json("wikipedia"), "wikipedia", "Model Context Protocol"),
            ("arxiv", providers.search_arxiv, (FIXTURES / "arxiv.xml").read_text(), "arxiv", "Attention Is All You Need"),
            ("hackernews", providers.search_hn, fixture_json("hackernews"), "hackernews", "MCP discussion"),
            ("stackexchange", providers.search_stackexchange, fixture_json("stackexchange"), "stackexchange:stackoverflow", "How do I use asyncio?"),
            ("crossref", providers.search_crossref, fixture_json("crossref"), "crossref", "A research paper"),
        )
        for name, function, payload, source, title in cases:
            with self.subTest(provider=name):
                response = FixtureResponse(
                    text=payload if isinstance(payload, str) else "",
                    payload=None if isinstance(payload, str) else payload,
                )
                kwargs = {"site": "stackoverflow"} if name == "stackexchange" else {}
                if name in {"brave", "tavily"}:
                    env = {"BRAVE_API_KEY": "test", "TAVILY_API_KEY": "test"}
                else:
                    env = {}
                with patch.dict(os.environ, env, clear=False):
                    result = await function(name, 1, FixtureClient(response), **kwargs)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].source, source)
                self.assertEqual(result[0].title, title)
                self.assertTrue(result[0].url.startswith("https://"))

    async def test_fetch_fixture_parses_jina_metadata(self):
        response = FixtureResponse(text=(FIXTURES / "jina.md").read_text())
        with patch("web_research.providers.validate_url"):
            result = await providers.fetch_jina("https://example.test/article", FixtureClient(response))
        self.assertEqual(result["title"], "Example article")
        self.assertEqual(result["url"], "https://example.test/article")
        self.assertIn("MCP connects", result["content"])


class ProviderFailureFixtureMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_search_providers_degrade_on_transport_failures(self):
        cases = (
            ("brave", providers.search_brave, {"BRAVE_API_KEY": "test"}, {}),
            ("tavily", providers.search_tavily, {"TAVILY_API_KEY": "test"}, {}),
            ("wikipedia", providers.search_wikipedia, {}, {}),
            ("arxiv", providers.search_arxiv, {}, {}),
            ("hackernews", providers.search_hn, {}, {}),
            ("stackexchange", providers.search_stackexchange, {}, {"site": "stackoverflow"}),
            ("crossref", providers.search_crossref, {}, {}),
        )
        for name, function, env, kwargs in cases:
            with self.subTest(provider=name):
                client = FixtureClient(FixtureResponse(error=httpx.HTTPError("fixture transport failure")))
                with patch.dict(os.environ, env, clear=False):
                    self.assertEqual(await function("query", 1, client, **kwargs), [])

    async def test_json_and_xml_parse_failures_degrade(self):
        cases = (
            ("wikipedia", providers.search_wikipedia, FixtureResponse(payload=ValueError("invalid json"))),
            ("arxiv", providers.search_arxiv, FixtureResponse(text="<not xml")),
            ("hackernews", providers.search_hn, FixtureResponse(payload=ValueError("invalid json"))),
            ("crossref", providers.search_crossref, FixtureResponse(payload=ValueError("invalid json"))),
        )
        for name, function, response in cases:
            with self.subTest(provider=name):
                self.assertEqual(await function("query", 1, FixtureClient(response)), [])


if __name__ == "__main__":
    unittest.main()
