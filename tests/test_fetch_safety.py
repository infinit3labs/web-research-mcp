import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from web_research import providers
from web_research.url_safety import UrlSafetyError, validate_url


class UrlPolicyTests(unittest.TestCase):
    def test_blocks_private_loopback_link_local_and_metadata_addresses(self):
        blocked = (
            "http://10.0.0.1/",
            "http://127.0.0.1/",
            "http://[::1]/",
            "http://169.254.169.254/latest/meta-data/",
        )

        for url in blocked:
            with self.subTest(url=url):
                with self.assertRaises(UrlSafetyError):
                    validate_url(url, resolve=False)

    def test_revalidates_redirect_targets(self):
        with self.assertRaises(UrlSafetyError):
            validate_url("http://127.0.0.1/admin", resolve=False)

    def test_resolved_hostname_cannot_point_at_private_address(self):
        with patch(
            "web_research.url_safety.resolve_host_ips",
            return_value=("93.184.216.34", "192.168.1.12"),
        ):
            with self.assertRaises(UrlSafetyError):
                validate_url("https://example.test/", resolve=True)

    def test_rejects_malformed_url_instead_of_leaking_parser_errors(self):
        with self.assertRaises(UrlSafetyError):
            validate_url("http://[not-an-ip]/", resolve=False)

    def test_blocks_cloud_metadata_hostname(self):
        with self.assertRaises(UrlSafetyError):
            validate_url("http://metadata.google.internal/", resolve=False)

    def test_allows_public_http_url(self):
        with patch(
            "web_research.url_safety.resolve_host_ips",
            return_value=("93.184.216.34",),
        ):
            validate_url("https://example.test/article", resolve=True)


class FetchJinaSafetyTests(unittest.TestCase):
    def test_rejects_private_url_before_network_request(self):
        client = _RecordingClient()

        result = asyncio.run(providers.fetch_jina("http://127.0.0.1/", client))

        self.assertIn("blocked", result["error"])
        self.assertEqual(client.calls, [])

    def test_rejects_private_redirect_reported_by_reader(self):
        response = _Response(
            "Title: Example\nURL Source: http://169.254.169.254/latest/meta-data/\n\nbody",
            headers={"content-type": "text/markdown"},
        )
        client = _RecordingClient(response)

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("blocked", result["error"])

    def test_revalidates_http_redirect_before_following_it(self):
        response = _Response(
            "",
            headers={"location": "http://127.0.0.1/admin"},
            status_code=302,
        )
        client = _RecordingClient(response)

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("blocked", result["error"])
        self.assertEqual(len(client.calls), 1)

    def test_revalidates_each_redirect_hop(self):
        client = _SequenceClient(
            [
                _Response("", headers={"location": "https://example.test/next"}, status_code=302),
                _Response("", headers={"location": "http://127.0.0.1/admin"}, status_code=302),
            ]
        )

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("blocked", result["error"])
        self.assertEqual(len(client.calls), 2)

    def test_rejects_oversized_reader_response(self):
        response = _Response("x" * (5 * 1024 * 1024 + 1), headers={"content-type": "text/markdown"})
        client = _RecordingClient(response)

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("too large", result["error"])

    def test_rejects_oversized_stream_without_content_length(self):
        client = _StreamingClient(
            _StreamingResponse(
                [b"x" * providers.FETCH_MAX_BYTES, b"y"],
                headers={"content-type": "text/markdown"},
            )
        )

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("too large", result["error"])

    def test_rejects_non_text_reader_response(self):
        response = _Response("binary", headers={"content-type": "application/octet-stream"})
        client = _RecordingClient(response)

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("content type", result["error"])

    def test_rejects_text_event_stream_response(self):
        response = _Response("event: hostile", headers={"content-type": "text/event-stream"})
        client = _RecordingClient(response)

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertIn("content type", result["error"])

    def test_caps_timeout_even_when_global_override_is_larger(self):
        client = _RecordingClient(
            _Response(
                "Title: Example\nURL Source: https://example.test/\n\nbody",
                headers={"content-type": "text/markdown"},
            )
        )

        with patch.dict(os.environ, {"WEB_RESEARCH_TIMEOUT_SECONDS": "9999"}, clear=False):
            with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
                result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertNotIn("error", result)
        self.assertLessEqual(client.calls[0]["timeout"], providers.FETCH_TIMEOUT_SECONDS)

    def test_uses_bounded_timeout_and_no_implicit_redirects(self):
        client = _RecordingClient(
            _Response("Title: Example\nURL Source: https://example.test/\n\nbody", headers={"content-type": "text/markdown"})
        )

        with patch("web_research.url_safety.resolve_host_ips", return_value=("93.184.216.34",)):
            result = asyncio.run(providers.fetch_jina("https://example.test/", client))

        self.assertNotIn("error", result)
        self.assertEqual(client.calls[0]["follow_redirects"], False)
        self.assertLessEqual(client.calls[0]["timeout"], 45.0)


class _Response:
    def __init__(self, text, headers, status_code=200):
        self.text = text
        self.content = text.encode()
        self.headers = httpx.Headers(headers)
        self.status_code = status_code

    @property
    def is_redirect(self):
        return 300 <= self.status_code < 400

    def raise_for_status(self):
        return None


class _RecordingClient:
    def __init__(self, response=None):
        self.response = response or _Response(
            "Title: Example\nURL Source: https://example.test/\n\nbody",
            headers={"content-type": "text/markdown"},
        )
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


class _SequenceClient(_RecordingClient):
    def __init__(self, responses):
        super().__init__(responses[0])
        self.responses = list(responses)

    async def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class _StreamingResponse:
    def __init__(self, chunks, headers, status_code=200):
        self.chunks = chunks
        self.headers = httpx.Headers(headers)
        self.status_code = status_code

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _StreamingContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StreamingClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return _StreamingContext(self.response)
