from __future__ import annotations

import unittest
from unittest.mock import ANY, AsyncMock, patch

from web_research import providers


class FakeSearchProvider:
    name = "fake"
    capabilities = frozenset({providers.Capability.SEARCH})

    async def search(self, query: str, max_results: int, client: object, **kwargs: object) -> list[providers.Result]:
        return [providers.Result("Fake", "https://example.com", query, self.name)]


class FakeFetchProvider:
    name = "fetcher"
    capabilities = frozenset({providers.Capability.FETCH})

    async def fetch(self, url: str, client: object) -> dict[str, object]:
        return {"url": url, "content": "fetched"}


class ProviderRegistryTests(unittest.TestCase):
    def test_registry_registers_and_discovers_providers_by_capability(self) -> None:
        registry = providers.ProviderRegistry()
        provider = FakeSearchProvider()

        registry.register(provider)

        self.assertIs(registry.get("fake"), provider)
        self.assertIs(registry.get("fake", providers.Capability.SEARCH), provider)
        self.assertEqual(registry.providers_for(providers.Capability.SEARCH), (provider,))
        self.assertEqual(registry.providers_for(providers.Capability.FETCH), ())

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = providers.ProviderRegistry()
        registry.register(FakeSearchProvider())

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(FakeSearchProvider())

    def test_registry_accepts_capability_specific_provider_contracts(self) -> None:
        registry = providers.ProviderRegistry()
        searcher = FakeSearchProvider()
        fetcher = FakeFetchProvider()

        registry.register(searcher)
        registry.register(fetcher)

        self.assertIs(registry.get("fake", providers.Capability.SEARCH), searcher)
        self.assertIs(registry.get("fetcher", providers.Capability.FETCH), fetcher)

    def test_registry_rejects_capability_mismatch_before_dispatch(self) -> None:
        registry = providers.ProviderRegistry()
        registry.register(FakeSearchProvider())

        with self.assertRaisesRegex(ValueError, "does not support fetch"):
            registry.get("fake", providers.Capability.FETCH)

    def test_builtin_provider_selection_uses_capabilities(self) -> None:
        searchers = providers.provider_registry.providers_for(providers.Capability.SEARCH)
        fetchers = providers.provider_registry.providers_for(providers.Capability.FETCH)

        self.assertEqual(
            {provider.name for provider in searchers},
            {"brave", "tavily", "wikipedia", "arxiv", "hackernews", "stackexchange", "crossref"},
        )
        self.assertEqual([provider.name for provider in fetchers], ["jina"])


class ProviderAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_search_adapter_preserves_legacy_function_entrypoint(self) -> None:
        expected = [providers.Result("Patched", "https://example.com", "snippet", "wikipedia")]
        legacy_search = AsyncMock(return_value=expected)

        with patch.object(providers, "search_wikipedia", legacy_search):
            result = await providers.provider_registry.get(
                "wikipedia", providers.Capability.SEARCH
            ).search("query", 3, object())

        self.assertEqual(result, expected)
        legacy_search.assert_awaited_once_with("query", 3, ANY)


if __name__ == "__main__":
    unittest.main()
