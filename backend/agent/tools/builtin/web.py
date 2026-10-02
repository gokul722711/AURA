"""Provider-neutral WebSearchTool and web search provider abstractions for AURA (M10)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import logging
import os
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import urllib.request

from agent.security import sanitize_data, sanitize_text
from agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized result item returned by a WebSearchProvider.

    Attributes:
        title: Title of the web page or result.
        url: Full canonical URL of the web page.
        snippet: Extracted text snippet or content excerpt.
        domain: Hostname/domain of the source.
        timestamp: ISO-8601 timestamp when the result was retrieved.
        metadata: Additional provider or attribution metadata.
    """

    title: str
    url: str
    snippet: str
    domain: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            domain = parsed.netloc or parsed.path
            object.__setattr__(self, "domain", domain)
        if not self.timestamp:
            now_iso = datetime.now(timezone.utc).isoformat()
            object.__setattr__(self, "timestamp", now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


class WebSearchProvider(ABC):
    """Abstract interface for pluggable web search providers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        """Execute a web search and return structured WebSearchResults."""
        pass


class MockWebSearchProvider(WebSearchProvider):
    """Deterministic, offline-capable search provider for testing and development.

    Supports canned responses, simulated failures, or automatic deterministic
    mock result synthesis based on query keywords.
    """

    def __init__(
        self,
        canned_results: dict[str, list[WebSearchResult]] | None = None,
        default_results: list[WebSearchResult] | None = None,
        simulated_failure: Exception | None = None,
    ) -> None:
        self.canned_results = canned_results or {}
        self.default_results = default_results
        self.simulated_failure = simulated_failure

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if self.simulated_failure is not None:
            raise self.simulated_failure

        clean_query = query.strip()
        if clean_query in self.canned_results:
            return self.canned_results[clean_query][:top_k]

        if self.default_results is not None:
            return self.default_results[:top_k]

        # Generate deterministic synthetic results for the query
        q_slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean_query.lower()).strip("-") or "search"
        results = [
            WebSearchResult(
                title=f"Web Overview: {clean_query}",
                url=f"https://example.org/articles/{q_slug}-overview",
                snippet=f"Detailed reference and analysis regarding {clean_query}. Contains recent web findings and documentation.",
                domain="example.org",
                metadata={"query": clean_query, "rank": 1},
            ),
            WebSearchResult(
                title=f"Latest Updates on {clean_query}",
                url=f"https://techdocs.io/news/{q_slug}-updates",
                snippet=f"Comprehensive documentation and architectural notes covering {clean_query}.",
                domain="techdocs.io",
                metadata={"query": clean_query, "rank": 2},
            ),
            WebSearchResult(
                title=f"Community Guide: {clean_query}",
                url=f"https://open-forum.net/discussions/{q_slug}",
                snippet=f"In-depth discussion and benchmarks comparing implementations of {clean_query}.",
                domain="open-forum.net",
                metadata={"query": clean_query, "rank": 3},
            ),
        ]
        return results[:top_k]


VOID_TAGS: set[str] = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def extract_ddg_destination_url(raw_url: str) -> str:
    """Extract destination URL from a DuckDuckGo redirect link.

    DuckDuckGo HTML results often use redirect links formatted as:
    //duckduckgo.com/l/?uddg=<encoded_destination_url>&rut=...
    or /l/?kh=-1&uddg=<encoded_destination_url>
    """
    if not raw_url:
        return ""
    clean = unescape(raw_url).strip()
    if "uddg=" in clean:
        try:
            candidate = clean
            if candidate.startswith("//"):
                candidate = "https:" + candidate
            elif candidate.startswith("/"):
                candidate = "https://duckduckgo.com" + candidate
            parsed = urlparse(candidate)
            query_params = parse_qs(parsed.query)
            if "uddg" in query_params and query_params["uddg"]:
                return unquote(query_params["uddg"][0]).strip()
        except Exception:
            pass
    return clean


def validate_web_url(url: str) -> bool:
    """Validate that a URL is a well-formed http or https web URL with a valid hostname.

    Args:
        url: Candidate URL string.

    Returns:
        True if valid http/https URL with non-empty hostname, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False
    trimmed = url.strip()
    if not trimmed:
        return False
    try:
        parsed = urlparse(trimmed)
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname or not hostname.strip():
            return False
        if any(c in hostname for c in (" ", "\t", "\n", "\r", "<", ">", '"', "'")):
            return False
        return True
    except Exception:
        return False


class DuckDuckGoHTMLParser(HTMLParser):
    """HTML parser to extract web search results from DuckDuckGo HTML Lite."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.raw_results: list[dict[str, str]] = []
        self._current: dict[str, str] = {"url": "", "title": "", "snippet": ""}
        self._capture_target: str | None = None  # "title" or "snippet"
        self._capture_depth: int = 0
        self._depth: int = 0

    def _has_data(self, item: dict[str, str]) -> bool:
        return bool(item.get("url") or item.get("snippet") or item.get("title"))

    def _commit_current(self) -> None:
        raw_url = self._current.get("url", "").strip()
        raw_title = self._current.get("title", "").strip()
        raw_snippet = self._current.get("snippet", "").strip()

        actual_url = extract_ddg_destination_url(raw_url)
        if not validate_web_url(actual_url):
            return

        clean_title = unescape(re.sub(r"\s+", " ", raw_title)).strip()
        clean_snippet = unescape(re.sub(r"\s+", " ", raw_snippet)).strip()

        parsed = urlparse(actual_url)
        domain = (parsed.netloc or "").lower()

        if not clean_title:
            clean_title = f"{domain} - {clean_snippet[:50]}..." if clean_snippet else domain

        self.raw_results.append({
            "url": actual_url,
            "title": clean_title,
            "snippet": clean_snippet,
            "domain": domain,
        })

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        is_void = tag_lower in VOID_TAGS
        if not is_void:
            self._depth += 1

        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        class_attr = attr_dict.get("class", "").lower()
        class_names = set(class_attr.split())
        href = attr_dict.get("href", "").strip()

        # Check if this tag represents a new result container
        is_container = (
            tag_lower in ("div", "article", "li", "tr", "section")
            and (
                "result" in class_names
                or "web-result" in class_names
                or "result__body" in class_attr
                or "results_links" in class_names
            )
            and "results" not in class_names
            and "results--main" not in class_names
        )

        if is_container:
            if self._has_data(self._current):
                self._commit_current()
                self._current = {"url": "", "title": "", "snippet": ""}
            self._capture_target = None
            return

        # If currently capturing inside title or snippet, handle child tags
        if self._capture_target is not None:
            if href and not self._current.get("url"):
                self._current["url"] = extract_ddg_destination_url(href)
            if is_void and tag_lower == "br":
                self._current[self._capture_target] += " "
            return

        # Check if title element
        is_title = (
            "result__title" in class_attr
            or "result-title" in class_attr
            or "result_title" in class_attr
            or "result__a" in class_names
            or (not self._current.get("title") and "result__url" in class_attr)
            or (tag_lower in ("h1", "h2", "h3") and ("title" in class_attr or "result" in class_attr))
        )

        if is_title:
            if self._current.get("snippet") or (self._current.get("title") and self._current.get("url")):
                self._commit_current()
                self._current = {"url": "", "title": "", "snippet": ""}
            if href and not self._current.get("url"):
                self._current["url"] = extract_ddg_destination_url(href)
            self._capture_target = "title"
            self._capture_depth = self._depth
            return

        # Check if snippet element
        is_snippet = (
            "result__snippet" in class_attr
            or "result-snippet" in class_attr
            or "result_snippet" in class_attr
            or "snippet" in class_names
        )

        if is_snippet:
            if self._current.get("snippet"):
                self._commit_current()
                self._current = {"url": "", "title": "", "snippet": ""}
            if href and not self._current.get("url"):
                self._current["url"] = extract_ddg_destination_url(href)
            self._capture_target = "snippet"
            self._capture_depth = self._depth
            return

        # General link fallback
        if href and not self._current.get("url") and ("result" in class_attr or "link" in class_attr):
            self._current["url"] = extract_ddg_destination_url(href)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in VOID_TAGS:
            return

        if self._capture_target is not None and self._depth <= self._capture_depth:
            self._capture_target = None

        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._capture_target and data:
            self._current[self._capture_target] += data

    def close(self) -> None:
        super().close()
        if self._has_data(self._current):
            self._commit_current()
            self._current = {"url": "", "title": "", "snippet": ""}


class DuckDuckGoWebSearchProvider(WebSearchProvider):
    """Live web search provider using DuckDuckGo HTML / Instant Answers.

    Uses standard Python urllib and html.parser without requiring external third-party packages.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    @staticmethod
    def parse_html_results(html_content: str, query: str = "", top_k: int = 5) -> list[WebSearchResult]:
        """Parse raw HTML from DuckDuckGo into structured WebSearchResults using HTMLParser."""
        if not html_content or not html_content.strip():
            return []

        parser = DuckDuckGoHTMLParser()
        parser.feed(html_content)
        parser.close()

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()

        for item in parser.raw_results:
            url = item["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)

            results.append(
                WebSearchResult(
                    title=item["title"],
                    url=url,
                    snippet=item["snippet"],
                    domain=item["domain"],
                    metadata={"source": "duckduckgo", "query": query},
                )
            )
            if len(results) >= top_k:
                break

        return results

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        clean_query = query.strip()
        if not clean_query:
            return []

        # DuckDuckGo HTML lite search
        encoded_q = quote_plus(clean_query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_q}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html_content = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("DuckDuckGo web search request failed for '%s': %s", clean_query, exc)
            raise RuntimeError(f"DuckDuckGo search connection error: {sanitize_text(str(exc))}") from exc

        return self.parse_html_results(html_content, query=clean_query, top_k=top_k)


def get_default_web_search_provider() -> WebSearchProvider:
    """Instantiate the default WebSearchProvider based on application settings/environment."""
    provider_name = os.environ.get("AI_WEB_SEARCH_PROVIDER", "mock").strip().lower()
    if provider_name == "duckduckgo":
        timeout = float(os.environ.get("AI_WEB_SEARCH_TIMEOUT", "10.0"))
        return DuckDuckGoWebSearchProvider(timeout=timeout)
    return MockWebSearchProvider()


class WebSearchTool(Tool):
    """Tool that queries the live web and returns structured evidence chunks."""

    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        top_k: int = 5,
    ) -> None:
        self.provider = provider or get_default_web_search_provider()
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the live web for public information, documentation, "
            "and external sources matching a natural language query."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search the web.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (optional).",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Parameter 'query' must be a non-empty string.",
            )

        trimmed_query = query.strip()
        if not trimmed_query:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Parameter 'query' cannot be empty or whitespace-only.",
            )

        top_k = kwargs.get("top_k")
        limit = top_k if (isinstance(top_k, int) and top_k > 0) else self.top_k

        try:
            raw_results = self.provider.search(trimmed_query, top_k=limit)
        except Exception as exc:
            safe_err = sanitize_text(str(exc))
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=f"Web search failed: {safe_err}",
                metadata={"query": trimmed_query},
            )

        output_chunks: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for idx, item in enumerate(raw_results, start=1):
            if not item.url or item.url in seen_urls:
                continue
            seen_urls.add(item.url)

            # Generate stable, unique chunk_id from URL
            url_hash = hashlib.sha256(item.url.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"web-{url_hash}"
            title = item.title.strip() if item.title else (item.domain or "Web Source")

            output_chunks.append({
                "chunk_id": chunk_id,
                "document_id": item.domain or item.url,
                "document_title": title,
                "document_source": item.url,
                "rank": idx,
                "score": None,  # Explicit positional rank; web search provides ranking ('rank'), not vector similarity ('score').
                "content": item.snippet,
                "citation": f"[{title}, Chunk: {chunk_id}]",
                "metadata": {
                    "source_type": "web",
                    "domain": item.domain,
                    "url": item.url,
                    "rank": idx,
                    "retrieved_at": item.timestamp,
                    **item.metadata,
                },
            })

        return ToolResult(
            tool_name=self.name,
            output=output_chunks,
            is_error=False,
            metadata={
                "query": trimmed_query,
                "retrieval_count": len(output_chunks),
                "source_type": "web",
            },
        )
