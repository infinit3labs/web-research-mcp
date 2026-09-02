import asyncio
import contextlib
import io
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from web_research import deep_research, providers, server


class FakeResponse:
    def __init__(self, status_code=200, *, headers=None, payload=None):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(self.status_code, headers=self.headers),
            )

    def json(self):
        return self._payload


class ScriptedClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.timeouts = []

    async def get(self, url, **kwargs):
        self.timeouts.append(kwargs["timeout"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class AsyncClientContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ProviderResilienceTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            key: os.environ.get(key)
            for key in (
                "WEB_RESEARCH_MAX_RETRIES",
                "WEB_RESEARCH_RETRY_BACKOFF_SECONDS",
                "WEB_RESEARCH_MAX_BACKOFF_SECONDS",
                "WEB_RESEARCH_TIMEOUT_SECONDS",
                "WEB_RESEARCH_TIMEOUT_WIKIPEDIA_SECONDS",
            )
        }
        os.environ.update({
            "WEB_RESEARCH_MAX_RETRIES": "2",
            "WEB_RESEARCH_RETRY_BACKOFF_SECONDS": "0",
            "WEB_RESEARCH_MAX_BACKOFF_SECONDS": "0",
        })
        providers.reset_caches()

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_retries_transient_timeout_then_returns_results(self):
        response = FakeResponse(payload={"query": {"search": [{"title": "MCP"}]}})
        client = ScriptedClient([httpx.ReadTimeout("temporary"), response])

        results = asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual(["MCP"], [result.title for result in results])
        self.assertEqual([15.0, 15.0], client.timeouts)

    def test_retries_rate_limit_using_retry_after(self):
        limited = FakeResponse(429, headers={"Retry-After": "0"})
        response = FakeResponse(payload={"query": {"search": [{"title": "MCP"}]}})
        client = ScriptedClient([limited, response])

        results = asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual(1, len(results))
        self.assertEqual(2, len(client.timeouts))

    def test_timeout_override_is_applied_to_provider_request(self):
        os.environ["WEB_RESEARCH_TIMEOUT_SECONDS"] = "3.5"
        response = FakeResponse(payload={"query": {"search": []}})
        client = ScriptedClient([response])

        asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual([3.5], client.timeouts)

    def test_provider_timeout_override_takes_precedence_over_global_timeout(self):
        os.environ["WEB_RESEARCH_TIMEOUT_SECONDS"] = "3.5"
        os.environ["WEB_RESEARCH_TIMEOUT_WIKIPEDIA_SECONDS"] = "1.25"
        response = FakeResponse(payload={"query": {"search": []}})
        client = ScriptedClient([response])

        asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual([1.25], client.timeouts)

    def test_exhausted_rate_limit_is_logged_with_retry_status(self):
        os.environ["WEB_RESEARCH_MAX_RETRIES"] = "0"
        client = ScriptedClient([FakeResponse(429, headers={"Retry-After": "7"})])
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            results = asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual([], results)
        self.assertIn('"status":"rate_limited"', stderr.getvalue())
        self.assertIn('"retry_after_seconds":7.0', stderr.getvalue())

    def test_exhausted_retries_degrade_to_empty_results(self):
        client = ScriptedClient([
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(503),
        ])

        results = asyncio.run(providers.search_wikipedia("MCP", 1, client))

        self.assertEqual([], results)
        self.assertEqual(3, len(client.timeouts))

    def test_deep_research_keeps_results_when_one_provider_raises(self):
        expected = providers.Result("working", "https://example.test", "ok", "wikipedia")
        sq = deep_research.SubQuestion(
            id="sq_1", question="question", rationale="test", queries=["question"], sources=["wikipedia", "arxiv"]
        )

        async def working(*_args):
            return [expected]

        async def failing(*_args):
            raise RuntimeError("provider unavailable")

        class FakeProvider:
            def __init__(self, name, search):
                self.name = name
                self.search_fn = search

            async def search(self, *args):
                return await self.search_fn(*args)

        fake_registry = type("Registry", (), {
            "providers_for": lambda _self, _capability: (
                FakeProvider("wikipedia", working),
                FakeProvider("arxiv", failing),
            )
        })()
        with patch.object(providers, "provider_registry", fake_registry):
            results = asyncio.run(deep_research._gather_search(sq, 1, object()))

        self.assertEqual([expected], results)

    def test_search_web_returns_available_results_with_partial_failure_status(self):
        expected = providers.Result("working", "https://example.test", "ok", "tavily")

        async def capability(_capability, _query, _max_results, _client, *, failure_sink=None, **_kwargs):
            if failure_sink is not None:
                failure_sink.append("brave")
            return [expected]

        with patch.object(server, "_new_client", AsyncMock(return_value=AsyncClientContext())), patch.object(
            server, "_search_capability", side_effect=capability
        ):
            result = asyncio.run(server.search_web("question", 1))

        self.assertIn("https://example.test", result)
        self.assertIn("Partial results", result)
        self.assertIn("brave", result)


if __name__ == "__main__":
    unittest.main()
