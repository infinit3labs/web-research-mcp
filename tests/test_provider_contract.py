from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from web_research import providers, server


class ProviderContractTests(unittest.TestCase):
    def test_builtin_capability_discovery_exposes_provider_roles(self) -> None:
        registry = providers.provider_registry

        self.assertEqual(
            {config.name for config in registry.discover(providers.Capability.ACADEMIC)},
            {"arxiv", "crossref"},
        )
        self.assertEqual(
            {config.name for config in registry.discover(providers.Capability.NEWS)},
            {"hackernews"},
        )
        self.assertEqual(
            {config.name for config in registry.discover(providers.Capability.COMMUNITY)},
            {"stackexchange"},
        )

    def test_api_key_requirements_and_availability_are_discoverable(self) -> None:
        config = providers.provider_registry.configuration("brave")

        self.assertEqual(config.api_key_env, "BRAVE_API_KEY")
        self.assertTrue(config.requires_api_key)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(config.available)
            self.assertFalse(config.to_dict()["available"])

        with patch.dict(os.environ, {"BRAVE_API_KEY": "test-key"}):
            self.assertTrue(config.configured)
            self.assertTrue(config.available)
            self.assertTrue(
                providers.provider_registry.discover(
                    providers.Capability.GENERAL_SEARCH, available_only=True
                )
            )

    def test_keyless_fetch_provider_is_available_without_optional_key(self) -> None:
        config = providers.provider_registry.configuration("jina")

        self.assertEqual(config.api_key_env, "JINA_API_KEY")
        self.assertFalse(config.requires_api_key)
        self.assertTrue(config.available)

    def test_registry_addition_requires_no_handler_registration(self) -> None:
        class CustomGeneralSearch:
            name = "custom"
            capabilities = frozenset(
                {providers.Capability.SEARCH, providers.Capability.GENERAL_SEARCH}
            )

            async def search(self, query: str, max_results: int, client: object, **kwargs: object):
                return [providers.Result("Custom", "https://example.test", query, self.name)]

        registry = providers.ProviderRegistry()
        provider = CustomGeneralSearch()
        registry.register(provider)

        self.assertEqual(registry.providers_for(providers.Capability.GENERAL_SEARCH), (provider,))
        self.assertEqual(registry.discover()[0].name, "custom")


class ProviderDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_general_search_dispatches_registered_capability_without_handler_changes(self) -> None:
        class CustomGeneralSearch:
            name = "custom"
            capabilities = frozenset(
                {providers.Capability.SEARCH, providers.Capability.GENERAL_SEARCH}
            )

            async def search(self, query: str, max_results: int, client: object, **kwargs: object):
                return [providers.Result("Custom", "https://example.test", query, self.name)]

        registry = providers.ProviderRegistry()
        registry.register(CustomGeneralSearch())

        with patch.object(providers, "provider_registry", registry):
            results = await server._search_capability(
                providers.Capability.GENERAL_SEARCH, "query", 1, object()
            )

        self.assertEqual([result.source for result in results], ["custom"])


if __name__ == "__main__":
    unittest.main()
