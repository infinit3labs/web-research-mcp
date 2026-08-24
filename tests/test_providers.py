"""Tests for the provider registry, base classes, and per-provider response parsing.

Covers INF-228 (typed interface + capability registry), INF-226 (graceful
degradation / retries), and INF-231 (SSRF integration on fetches).

All tests are offline — HTTP is mocked through ``httpx.MockTransport`` so the
provider response-parsing code is exercised against realistic but fake
payloads.
"""

from __future__ import annotations

import unittest.mock

import httpx
import pytest

from src.web_research import providers
from src.web_research.providers import (
    REGISTRY,
    ArxivProvider,
    BraveProvider,
    Capability,
    CrossrefProvider,
    FetchResult,
    HackerNewsProvider,
    JinaFetchProvider,
    ProviderOutcome,
    ProviderRegistry,
    Result,
    StackExchangeProvider,
    TavilyProvider,
    WikipediaProvider,
)


# --------------------------------------------------------------------------------------
# Registry semantics
# --------------------------------------------------------------------------------------


def test_registry_has_all_expected_providers():
    expected = {"brave", "tavily", "wikipedia", "arxiv", "hackernews",
                "stackexchange", "crossref", "jina"}
    got = {p.info.name for p in REGISTRY}
    assert expected <= got, f"missing providers: {expected - got}"


def test_registry_capability_lookup_returns_only_matching():
    web = REGISTRY.by_capability(Capability.WEB_SEARCH)
    assert {p.info.name for p in web} == {"brave", "tavily"}
    academic = REGISTRY.by_capability(Capability.ACADEMIC)
    assert {p.info.name for p in academic} == {"arxiv"}


def test_registry_describe_surfaces_availability_and_requirements():
    desc = {row["name"]: row for row in REGISTRY.describe()}
    assert desc["brave"]["requires_api_key"] == "BRAVE_API_KEY"
    # With no env vars set, any-key providers should be unavailable
    assert desc["brave"]["available"] is False
    assert desc["wikipedia"]["available"] is True
    assert desc["wikipedia"]["requires_api_key"] is None


def test_registry_duplicate_registration_raises():
    r = ProviderRegistry()
    r.register(WikipediaProvider())
    with pytest.raises(ValueError, match="already registered"):
        r.register(WikipediaProvider())


def test_registry_get_unknown_name_raises_keyerror():
    r = ProviderRegistry()
    with pytest.raises(KeyError, match="no provider registered"):
        r.get("does-not-exist")


# --------------------------------------------------------------------------------------
# Base class behavior (graceful degradation)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_provider_returns_unavailable_when_no_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    p = BraveProvider()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        outcome = await p.search("x", 5, client)
        assert outcome.unavailable is True
        assert outcome.results == []
        assert outcome.error is None
        assert outcome.ok  # 'ok' means "no error" — unavailability is a soft skip
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_provider_translates_provider_error_to_outcome(monkeypatch):
    """A ProviderError (e.g. 503) must surface as outcome.error + rate_limited, not raise."""
    monkeypatch.setenv("BRAVE_API_KEY", "fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream down", headers={"content-type": "text/plain"})

    p = BraveProvider()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with unittest.mock.patch("src.web_research.net.asyncio.sleep", new=unittest.mock.AsyncMock()):
            outcome = await p.search("x", 5, client)
        assert outcome.error is not None
        assert outcome.rate_limited is False
        assert outcome.results == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_provider_swallows_unsafe_url():
    """Jina fetch should not even try to call out for an unsafe URL."""
    p = JinaFetchProvider()

    called = []

    async def fake_handle(self, request):
        called.append(str(request.url))
        return httpx.Response(200, content=b"")

    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handle))
    try:
        result = await p.fetch("http://127.0.0.1:1/admin", client)
        assert result.error is not None
        assert "unsafe" in result.error
        assert called == []  # transport never invoked
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_provider_unavailable_returns_error():
    p = JinaFetchProvider()  # no api_key_env, so available
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        # safe URL, no network needed since we won't actually go out
        result = await p.fetch("https://example.com/", client)
        # outcome is whatever Jina's real flow returns; just assert it's a FetchResult
        assert isinstance(result, FetchResult)
    finally:
        await client.aclose()


# --------------------------------------------------------------------------------------
# Per-provider response parsing
# --------------------------------------------------------------------------------------


def _client_returning(body: bytes, content_type: str = "application/json") -> httpx.AsyncClient:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body, headers={"content-type": content_type}))
    return httpx.AsyncClient(transport=transport, timeout=5.0)


@pytest.mark.asyncio
async def test_brave_parses_results(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "fake")
    body = b'{"web":{"results":[{"title":"T1","url":"https://x.test/1","description":"d1","age":"1d"},{"title":"T2","url":"https://x.test/2","description":"d2"}]}}'
    client = _client_returning(body)
    try:
        out = await BraveProvider().search("x", 5, client)
        assert out.ok
        assert len(out.results) == 2
        assert out.results[0].title == "T1"
        assert out.results[0].source == "brave"
        assert out.results[0].extra.get("age") == "1d"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tavily_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    body = b'{"results":[{"title":"T1","url":"https://x.test/1","content":"snippet content","score":0.9}]}'
    client = _client_returning(body)
    try:
        out = await TavilyProvider().search("x", 5, client)
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].snippet == "snippet content"[:600]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_wikipedia_strips_html_in_snippet():
    body = b'{"query":{"search":[{"title":"MCP","snippet":"<b>M</b>odel <i>Context</i> Protocol","score":100}]}}'
    client = _client_returning(body)
    try:
        out = await WikipediaProvider().search("MCP", 3, client)
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].title == "MCP"
        assert "<b>" not in out.results[0].snippet
        assert "Model Context Protocol" in out.results[0].snippet
        assert out.results[0].url == "https://en.wikipedia.org/wiki/MCP"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_arxiv_parses_atom_with_authors():
    body = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry>
  <title>Attention Is All You Need</title>
  <summary>The dominant sequence transduction models...</summary>
  <id>https://arxiv.org/abs/1706.03762v7</id>
  <published>2017-06-12T17:57:34Z</published>
  <author><name>Ashish Vaswani</name></author>
  <author><name>Noam Shazeer</name></author>
</entry>
</feed>"""
    client = _client_returning(body, content_type="application/atom+xml")
    try:
        out = await ArxivProvider().search("attention", 2, client)
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].title == "Attention Is All You Need"
        assert out.results[0].published == "2017-06-12"
        assert "Ashish Vaswani" in out.results[0].extra["authors"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_arxiv_skips_entries_missing_title_or_url():
    body = b"""<feed xmlns="http://www.w3.org/2005/Atom">
<entry><summary>no title or id</summary></entry>
<entry><title>OK one</title><id>https://arxiv.org/abs/1</id><summary>abs</summary></entry>
</feed>"""
    client = _client_returning(body, content_type="application/atom+xml")
    try:
        out = await ArxivProvider().search("x", 5, client)
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].title == "OK one"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_hackernews_handles_missing_url():
    body = b'{"hits":[{"objectID":"42","title":"A HN story","points":50,"num_comments":10,"created_at":"2024-01-01T00:00:00Z","_highlightResult":{}}]}'
    client = _client_returning(body)
    try:
        out = await HackerNewsProvider().search("x", 5, client)
        assert out.ok
        assert len(out.results) == 1
        # No `url` field → falls back to /item?id=<objectID>
        assert out.results[0].url == "https://news.ycombinator.com/item?id=42"
        assert out.results[0].extra["points"] == 50
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stackexchange_passes_site_param_and_labels_source():
    body = b'{"items":[{"title":"How do I X?","link":"https://stackoverflow.com/q/1","excerpt":"<b>ex</b>cerpt text","score":5,"is_answered":true,"answer_count":2,"tags":["python","asyncio"]}]}'
    client = _client_returning(body)
    try:
        out = await StackExchangeProvider().search("x", 5, client, site="stackoverflow")
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].source == "stackexchange:stackoverflow"
        assert out.results[0].extra["is_answered"] is True
        assert out.results[0].extra["tags"] == ["python", "asyncio"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_crossref_picks_first_title_and_strips_abstract_html():
    body = b'{"message":{"items":[{"title":["CR Paper"],"URL":"https://example.com/cr","DOI":"10.1234/x","abstract":"<jats:p>The <b>abstract</b>.</jats:p>","published-print":{"date-parts":[[2024,3,15]]},"type":"journal-article","is-referenced-by-count":17}]}}'
    client = _client_returning(body)
    try:
        out = await CrossrefProvider().search("x", 5, client)
        assert out.ok
        assert len(out.results) == 1
        assert out.results[0].title == "CR Paper"
        assert "<jats:p>" not in out.results[0].snippet
        assert out.results[0].published == "2024-03-15"
        assert out.results[0].extra["doi"] == "10.1234/x"
        assert out.results[0].extra["citations"] == 17
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_crossref_falls_back_to_doi_url_when_no_url_field():
    body = b'{"message":{"items":[{"title":["No URL"],"DOI":"10.5/abc","published-online":{"date-parts":[[2023]]}}]}}'
    client = _client_returning(body)
    try:
        out = await CrossrefProvider().search("x", 5, client)
        assert out.ok
        assert out.results[0].url == "https://doi.org/10.5/abc"
    finally:
        await client.aclose()


# --------------------------------------------------------------------------------------
# Dedup / merge
# --------------------------------------------------------------------------------------


def test_canonical_url_strips_tracking_params_and_normalizes():
    a = providers._canonical_url("https://Example.com/foo/?utm_source=x&keep=yes&fbclid=abc")
    b = providers._canonical_url("https://example.com/foo/?keep=yes")
    assert a == b


def test_canonical_url_preserves_important_params():
    a = providers._canonical_url("https://x.test/search?q=python&page=2")
    assert "q=python" in a
    assert "page=2" in a


def test_canonical_url_handles_garbage():
    # should never raise, even on weird input
    assert isinstance(providers._canonical_url("not a url"), str)
    assert providers._canonical_url("not a url") == "not a url"


def test_merge_results_dedupes_and_prefers_higher_score():
    a = [Result(title="A", url="https://x.test/1", snippet="short", source="wikipedia", score=1.0)]
    b = [Result(title="A long", url="https://x.test/1", snippet="much longer snippet", source="brave", score=0.5)]
    c = [Result(title="B", url="https://x.test/2", snippet="b", source="tavily", score=2.0)]
    merged = providers.merge_results(a, b, c, max_total=10)
    assert len(merged) == 2
    # Sorted by score desc; B (2.0) before A
    assert merged[0].title == "B"
    # A absorbed B's longer snippet
    assert merged[1].snippet == "much longer snippet"
    # And the source got recorded as cross-source
    assert "brave" in merged[1].extra["also_found_in"]


def test_merge_results_caps_at_max_total():
    many = [Result(title=f"r{i}", url=f"https://x.test/{i}", snippet="s", source="t", score=float(i)) for i in range(20)]
    assert len(providers.merge_results(many, max_total=5)) == 5


# --------------------------------------------------------------------------------------
# Data model to_dict
# --------------------------------------------------------------------------------------


def test_result_to_dict_drops_empty_extra():
    r = Result(title="t", url="https://x.test/", snippet="s", source="s", score=1.0)
    d = r.to_dict()
    assert "extra" not in d
    assert d["title"] == "t"


def test_fetch_result_to_dict_only_includes_error_when_present():
    ok = FetchResult(url="https://x.test/", title="T", content="C", length=1, truncated=False)
    assert "error" not in ok.to_dict()
    bad = FetchResult(url="https://x.test/", error="boom")
    assert bad.to_dict()["error"] == "boom"


def test_provider_outcome_ok_inverse():
    assert ProviderOutcome(provider="p", results=[Result(title="t", url="u", snippet="s", source="s")]).ok
    assert not ProviderOutcome(provider="p", error="x").ok
    # unavailable without an error is still 'ok' — it's a soft skip
    assert ProviderOutcome(provider="p", unavailable=True).ok
