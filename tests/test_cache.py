"""Cache TTL/size behavior and bounded-concurrency checks for INF-235.

Uses fixture providers with controlled artificial latency instead of live
network calls — deterministic in CI, and lets the cold-vs-warm benchmark
assert a real speed-up without flaking on network variance.
"""
from __future__ import annotations

import asyncio
import os
import time
import unittest

from web_research import cache as cache_module
from web_research import providers


class TTLCacheTests(unittest.TestCase):
    def test_set_and_get_round_trips_value(self):
        c = cache_module.TTLCache(max_entries=10, ttl_seconds=60)
        c.set("k", ["v"])
        self.assertEqual(c.get("k"), ["v"])

    def test_missing_key_returns_none(self):
        c = cache_module.TTLCache(max_entries=10, ttl_seconds=60)
        self.assertIsNone(c.get("missing"))

    def test_expired_entry_is_evicted_on_read(self):
        now = [0.0]
        c = cache_module.TTLCache(max_entries=10, ttl_seconds=5, clock=lambda: now[0])
        c.set("k", "v")
        now[0] = 4.9
        self.assertEqual(c.get("k"), "v")
        now[0] = 5.1
        self.assertIsNone(c.get("k"))
        self.assertEqual(0, len(c))

    def test_zero_ttl_disables_caching(self):
        c = cache_module.TTLCache(max_entries=10, ttl_seconds=0)
        c.set("k", "v")
        self.assertIsNone(c.get("k"))
        self.assertEqual(0, len(c))

    def test_zero_max_entries_disables_caching(self):
        c = cache_module.TTLCache(max_entries=0, ttl_seconds=60)
        c.set("k", "v")
        self.assertIsNone(c.get("k"))

    def test_max_entries_evicts_least_recently_used(self):
        c = cache_module.TTLCache(max_entries=2, ttl_seconds=60)
        c.set("a", 1)
        c.set("b", 2)
        c.get("a")  # touch a — b becomes the least-recently-used entry
        c.set("c", 3)  # evicts b
        self.assertIsNone(c.get("b"))
        self.assertEqual(1, c.get("a"))
        self.assertEqual(3, c.get("c"))


class FakeSearchProvider:
    def __init__(self, name="fake-search", *, results_factory=None, latency=0.0):
        self.name = name
        self.calls: list[str] = []
        self._latency = latency
        self._results_factory = results_factory or (
            lambda query: [providers.Result(f"T-{query}", f"https://example.test/{query}", "snippet", self.name)]
        )

    async def search(self, query, max_results, client, **kwargs):
        self.calls.append(query)
        if self._latency:
            await asyncio.sleep(self._latency)
        return self._results_factory(query)


class FakeFetchProvider:
    def __init__(self, name="fake-fetch", *, error=None):
        self.name = name
        self.calls: list[str] = []
        self._error = error

    async def fetch(self, url, client):
        self.calls.append(url)
        if self._error:
            return {"url": url, "error": self._error}
        return {"url": url, "title": "Title", "content": "body", "truncated": False, "length": 4}


class CachedSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        providers.reset_caches()

    async def test_second_identical_call_is_served_from_cache(self):
        provider = FakeSearchProvider("cache-hit-provider")
        first = await providers.cached_search(provider, "python asyncio", 5, client=None)
        second = await providers.cached_search(provider, "python asyncio", 5, client=None)
        self.assertEqual(1, len(provider.calls))
        self.assertEqual([r.title for r in first], [r.title for r in second])

    async def test_cache_hit_returns_a_copy_callers_cannot_corrupt(self):
        provider = FakeSearchProvider("mutation-provider")
        first = await providers.cached_search(provider, "q", 5, client=None)
        first[0].title = "mutated-by-caller"
        second = await providers.cached_search(provider, "q", 5, client=None)
        self.assertEqual(1, len(provider.calls))
        self.assertNotEqual("mutated-by-caller", second[0].title)

    async def test_empty_results_are_never_cached(self):
        # An empty list can mean "genuinely no matches" or "provider degraded
        # after a failure" — providers.py can't tell the difference, so it
        # must not cache [] and silently suppress retries for the TTL window.
        provider = FakeSearchProvider("empty-provider", results_factory=lambda q: [])
        await providers.cached_search(provider, "q", 5, client=None)
        await providers.cached_search(provider, "q", 5, client=None)
        self.assertEqual(2, len(provider.calls))

    async def test_different_kwargs_use_different_cache_entries(self):
        provider = FakeSearchProvider("kwargs-provider")
        await providers.cached_search(provider, "q", 5, client=None, site="stackoverflow")
        await providers.cached_search(provider, "q", 5, client=None, site="serverfault")
        self.assertEqual(2, len(provider.calls))

    async def test_different_providers_do_not_share_cache_entries(self):
        a = FakeSearchProvider("provider-a")
        b = FakeSearchProvider("provider-b")
        await providers.cached_search(a, "same query", 5, client=None)
        await providers.cached_search(b, "same query", 5, client=None)
        self.assertEqual(1, len(a.calls))
        self.assertEqual(1, len(b.calls))


class CachedFetchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        providers.reset_caches()

    async def test_second_identical_fetch_is_served_from_cache(self):
        provider = FakeFetchProvider("fetch-hit-provider")
        await providers.cached_fetch(provider, "https://example.test/a?utm_source=x", client=None)
        await providers.cached_fetch(provider, "https://example.test/a", client=None)
        self.assertEqual(1, len(provider.calls), "canonicalization should map both URLs to one cache entry")

    async def test_error_results_are_never_cached(self):
        provider = FakeFetchProvider("erroring-fetch", error="fetch failed: boom")
        await providers.cached_fetch(provider, "https://example.test/a", client=None)
        await providers.cached_fetch(provider, "https://example.test/a", client=None)
        self.assertEqual(2, len(provider.calls))


class ConcurrencyBoundTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        providers.reset_caches()
        providers._provider_semaphores.pop("bound-test", None)
        os.environ["WEB_RESEARCH_MAX_CONCURRENCY_BOUND_TEST"] = "2"

    def tearDown(self):
        os.environ.pop("WEB_RESEARCH_MAX_CONCURRENCY_BOUND_TEST", None)
        providers._provider_semaphores.pop("bound-test", None)

    async def test_concurrent_fan_out_is_bounded_per_provider(self):
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        class SlowProvider:
            name = "bound-test"

            async def search(self, query, max_results, client, **kwargs):
                nonlocal in_flight, max_in_flight
                async with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.05)
                async with lock:
                    in_flight -= 1
                return [providers.Result(query, f"https://example.test/{query}", "s", self.name)]

        provider = SlowProvider()
        await asyncio.gather(*(
            providers.cached_search(provider, f"q{i}", 5, client=None) for i in range(6)
        ))

        self.assertLessEqual(max_in_flight, 2, "per-provider concurrency cap was not enforced")
        self.assertGreater(max_in_flight, 1, "fan-out should still run concurrently up to the cap")


class CacheBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    """Cold-vs-warm benchmark against a representative (fixture) workload."""

    def setUp(self):
        providers.reset_caches()

    async def test_warm_cache_run_is_faster_and_makes_no_extra_provider_calls(self):
        provider = FakeSearchProvider("latency-provider", latency=0.05)

        cold_start = time.perf_counter()
        await providers.cached_search(provider, "benchmark query", 5, client=None)
        cold_duration = time.perf_counter() - cold_start

        warm_start = time.perf_counter()
        await providers.cached_search(provider, "benchmark query", 5, client=None)
        warm_duration = time.perf_counter() - warm_start

        self.assertEqual(1, len(provider.calls), "warm run must not repeat the upstream call")
        self.assertLess(warm_duration, cold_duration / 5)


if __name__ == "__main__":
    unittest.main()
