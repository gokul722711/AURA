"""Automated unit and integration tests for SearXNGWebSearchProvider (M14)."""

import json
from unittest.mock import MagicMock, patch
from django.test import SimpleTestCase, override_settings

import httpx

from agent.tools.builtin.web import (
    SearXNGWebSearchProvider,
    WebSearchResult,
    WebSearchTool,
    get_default_web_search_provider,
)


class SearXNGProviderTests(SimpleTestCase):
    """Unit tests for SearXNGWebSearchProvider parsing, error handling, and networking."""

    SAMPLE_SEARXNG_RESPONSE = {
        "query": "python latest stable release",
        "number_of_results": 4,
        "results": [
            {
                "title": "Welcome to Python.org",
                "url": "https://www.python.org/",
                "content": "Official home of Python programming language.",
                "engine": "google",
                "engines": ["google", "duckduckgo"],
                "score": 4.5,
                "category": "general",
            },
            {
                "title": "Download Python | Python.org",
                "url": "https://www.python.org/downloads/",
                "content": "The official home of Python downloads.",
                "engine": "duckduckgo",
                "engines": ["duckduckgo"],
                "score": 3.8,
                "category": "general",
            },
            {
                # Duplicate canonical URL with trailing slash difference
                "title": "Python.org Home",
                "url": "https://www.python.org",
                "content": "Duplicate entry.",
                "engine": "wikipedia",
                "engines": ["wikipedia"],
            },
            {
                "title": "Invalid Scheme Result",
                "url": "javascript:alert(1)",
                "content": "Malicious payload.",
            },
        ],
    }

    def test_valid_searxng_json_response_parsing(self) -> None:
        """1 & 2. Valid SearXNG JSON response is parsed into structured WebSearchResult items."""
        provider = SearXNGWebSearchProvider()
        results = provider.parse_json_results(
            self.SAMPLE_SEARXNG_RESPONSE,
            query="python latest stable release",
            top_k=5,
        )

        self.assertEqual(len(results), 2)  # Duplicate and invalid scheme omitted
        first = results[0]
        self.assertEqual(first.title, "Welcome to Python.org")
        self.assertEqual(first.url, "https://www.python.org/")
        self.assertEqual(first.snippet, "Official home of Python programming language.")
        self.assertEqual(first.domain, "www.python.org")
        self.assertEqual(first.metadata["engine"], "google")
        self.assertEqual(first.metadata["engines"], ["google", "duckduckgo"])
        self.assertEqual(first.metadata["score"], 4.5)
        self.assertEqual(first.metadata["category"], "general")
        self.assertEqual(first.metadata["query"], "python latest stable release")

    def test_result_parsing_fallback_title(self) -> None:
        """Result parsing falls back to domain if title is empty or missing."""
        payload = {
            "results": [
                {
                    "url": "https://docs.python.org/3/",
                    "content": "Documentation for Python 3.",
                }
            ]
        }
        results = SearXNGWebSearchProvider.parse_json_results(payload, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "docs.python.org")

    def test_malformed_non_json_response_raises_clear_error(self) -> None:
        """3. Malformed non-JSON response raises clear descriptive RuntimeError."""
        provider = SearXNGWebSearchProvider()
        with self.assertRaises(RuntimeError) as cm:
            provider.parse_json_results("<html><body>502 Bad Gateway</body></html>")
        self.assertIn("SearXNG returned malformed non-JSON response", str(cm.exception))

    def test_missing_results_key_raises_clear_error(self) -> None:
        """Malformed response missing 'results' key raises clear descriptive RuntimeError."""
        provider = SearXNGWebSearchProvider()
        with self.assertRaises(RuntimeError) as cm:
            provider.parse_json_results({"error": "something went wrong"})
        self.assertIn("SearXNG response missing 'results' list", str(cm.exception))

    def test_timeout_raises_runtime_error(self) -> None:
        """4. Timeout in SearXNG request raises sanitized RuntimeError."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Read timed out")

        provider = SearXNGWebSearchProvider(client=mock_client)
        with self.assertRaises(RuntimeError) as cm:
            provider.search("python release")
        self.assertIn("SearXNG search timed out", str(cm.exception))

    def test_http_failure_raises_runtime_error(self) -> None:
        """5. HTTP failure status code raises clear RuntimeError."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_client.get.return_value = mock_resp

        provider = SearXNGWebSearchProvider(client=mock_client)
        with self.assertRaises(RuntimeError) as cm:
            provider.search("python release")
        self.assertIn("SearXNG search failed with status 503", str(cm.exception))

    def test_empty_results_returns_empty_list(self) -> None:
        """6. Empty results list in SearXNG payload returns empty list without error."""
        provider = SearXNGWebSearchProvider()
        results = provider.parse_json_results({"results": []}, top_k=5)
        self.assertEqual(results, [])

    def test_duplicate_url_removal(self) -> None:
        """7. Duplicate URLs (even with minor trailing slash differences) are deduplicated."""
        payload = {
            "results": [
                {"title": "Page 1", "url": "https://example.com/docs/", "content": "Docs"},
                {"title": "Page 1 Dup", "url": "https://example.com/docs", "content": "Docs duplicate"},
                {"title": "Page 2", "url": "https://example.com/about", "content": "About"},
            ]
        }
        results = SearXNGWebSearchProvider.parse_json_results(payload, top_k=5)
        self.assertEqual(len(results), 2)
        urls = [r.url for r in results]
        self.assertIn("https://example.com/docs/", urls)
        self.assertIn("https://example.com/about", urls)

    def test_top_k_acts_as_maximum_cap(self) -> None:
        """top_k restricts the maximum number of items returned."""
        payload = {
            "results": [
                {"title": f"Page {i}", "url": f"https://example.com/{i}", "content": f"Content {i}"}
                for i in range(10)
            ]
        }
        results = SearXNGWebSearchProvider.parse_json_results(payload, top_k=3)
        self.assertEqual(len(results), 3)

    def test_empty_query_returns_empty_list_immediately(self) -> None:
        """Empty or whitespace-only query returns empty list without making HTTP requests."""
        mock_client = MagicMock()
        provider = SearXNGWebSearchProvider(client=mock_client)
        self.assertEqual(provider.search("   "), [])
        mock_client.get.assert_not_called()

    def test_get_default_web_search_provider_searxng(self) -> None:
        """get_default_web_search_provider initializes SearXNGWebSearchProvider when configured."""
        with patch.dict("os.environ", {"AI_WEB_SEARCH_PROVIDER": "searxng", "AI_SEARXNG_URL": "http://127.0.0.1:8080"}):
            provider = get_default_web_search_provider()
            self.assertIsInstance(provider, SearXNGWebSearchProvider)
            self.assertEqual(provider.base_url, "http://127.0.0.1:8080")

    def test_web_search_tool_with_searxng_provider(self) -> None:
        """WebSearchTool executes successfully with SearXNGWebSearchProvider."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(self.SAMPLE_SEARXNG_RESPONSE)
        mock_client.get.return_value = mock_resp

        searxng_provider = SearXNGWebSearchProvider(client=mock_client)
        tool = WebSearchTool(provider=searxng_provider, top_k=2)

        result = tool.execute(query="python latest stable release")
        self.assertFalse(result.is_error)
        self.assertEqual(len(result.output), 2)
        self.assertTrue(result.output[0]["chunk_id"].startswith("web-"))
        self.assertEqual(result.output[0]["document_title"], "Welcome to Python.org")
