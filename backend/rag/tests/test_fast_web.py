"""Automated unit and integration tests for Deterministic Web and Hybrid pipelines (M14)."""

from unittest.mock import MagicMock, patch
from django.test import TestCase

from agent.results import ResearchEvidence, ResearchResult
from agent.state import AgentStatus
from agent.tools.builtin.web import MockWebSearchProvider, WebSearchResult
from gateway.types import GenerationResponse, ProviderMetadata, UsageInfo
from rag.context import ContextConfig
from rag.exceptions import WebFetchError
from rag.extraction.web import WebPageExtractor
from rag.fast_web import (
    DeterministicHybridPipeline,
    DeterministicWebPipeline,
    HybridConfig,
    WebConfig,
)
from rag.rerankers.mock import MockReranker
from rag.retrieval import RetrievalResult
from rag.web.mock import MockWebFetcher


class MockGatewayHelper:
    """Helper to construct a deterministic mock ModelGateway."""

    @staticmethod
    def create(synthesis_text: str = "Synthesized grounded answer [Title, https://example.com, Chunk: c1].") -> MagicMock:
        gateway = MagicMock()
        gateway.metadata.return_value = ProviderMetadata(
            provider="mock",
            model="mock-model",
            capabilities=("generation", "streaming"),
        )
        gateway.generate.return_value = GenerationResponse(
            text=synthesis_text,
            provider="mock",
            model="mock-model",
            usage=UsageInfo(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            finish_reason="stop",
        )
        return gateway


class DeterministicWebPipelineTests(TestCase):
    """Tests for DeterministicWebPipeline (Requirements 8–15)."""

    def setUp(self) -> None:
        self.gateway = MockGatewayHelper.create()
        self.search_provider = MockWebSearchProvider(
            canned_results={
                "python release": [
                    WebSearchResult(
                        title="Python Releases",
                        url="https://python.org/downloads",
                        snippet="Latest version is Python 3.13 released in October 2024.",
                        domain="python.org",
                    ),
                    WebSearchResult(
                        title="Python DevGuide",
                        url="https://devguide.python.org/versions",
                        snippet="Status of Python versions and release schedules.",
                        domain="devguide.python.org",
                    ),
                ]
            }
        )
        self.fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": (
                    "<!DOCTYPE html><html><head><title>Python Releases</title></head>"
                    "<body><article><p>Python 3.13.0 was released on October 7, 2024.</p>"
                    "<p>It includes improved performance and interactive interpreter enhancements.</p>"
                    "</article></body></html>"
                ),
                "https://devguide.python.org/versions": (
                    "<!DOCTYPE html><html><head><title>Python DevGuide</title></head>"
                    "<body><article><p>Python 3.13 is the latest active stable branch.</p>"
                    "<p>Python 3.14 is currently in pre-release development.</p>"
                    "</article></body></html>"
                ),
            }
        )
        self.extractor = WebPageExtractor()
        self.reranker = MockReranker(
            scorer=lambda q, c: 0.9 if "3.13" in c.content else 0.5,
            min_score=0.1,
        )
        self.config = WebConfig(
            search_top_k=5,
            max_fetch_pages=3,
            rerank_top_k=3,
            reranker_min_score=0.1,
        )
        self.pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=self.config,
        )

    def test_full_web_pipeline_workflow(self) -> None:
        """8 & 9. search -> fetch -> extract -> rank -> generate with exactly ONE generate call."""
        result = self.pipeline.run("python release")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.has_evidence)
        self.assertGreaterEqual(len(result.evidence), 1)
        self.assertEqual(self.gateway.generate.call_count, 1)

        # Verify citation and provenance
        first_ev = result.evidence[0]
        self.assertTrue(first_ev.chunk_id.startswith("web-"))
        self.assertEqual(first_ev.metadata["source_type"], "web")
        self.assertIn("https://", first_ev.url)
        self.assertIn("Chunk: web-", first_ev.citation)

        # Verify sources aggregation
        self.assertGreaterEqual(len(result.sources), 1)
        self.assertTrue(bool(result.sources[0].get("url")))

    def test_individual_page_failure_does_not_fail_entire_run(self) -> None:
        """10. Failed page (e.g. 403 or network error) does not abort the run if other pages succeed."""
        failing_fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": WebFetchError("HTTP 403 Forbidden"),
                "https://devguide.python.org/versions": (
                    "<!DOCTYPE html><html><head><title>Python DevGuide</title></head>"
                    "<body><article><p>Python 3.13 is the active stable branch.</p></article></body></html>"
                ),
            }
        )
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=failing_fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=self.config,
        )

        result = pipeline.run("python release")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.has_evidence)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].document_source, "https://devguide.python.org/versions")
        self.assertEqual(self.gateway.generate.call_count, 1)

    def test_all_pages_failing_produces_no_fabricated_evidence(self) -> None:
        """11. When all fetched pages fail, produces no evidence and exactly 0 generation calls."""
        all_failing_fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": WebFetchError("Network Timeout"),
                "https://devguide.python.org/versions": WebFetchError("DNS Failure"),
            }
        )
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=all_failing_fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=self.config,
        )

        result = pipeline.run("python release")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.evidence, [])
        self.assertIn("did not produce any usable extractable content", result.final_answer)
        self.assertEqual(self.gateway.generate.call_count, 0)

    def test_relevance_threshold_removes_irrelevant_chunks(self) -> None:
        """12. Relevance threshold removes chunks scoring below min_score; 0 calls if none qualify."""
        strict_reranker = MockReranker(
            scorer=lambda q, c: 0.05,  # All chunks score below min_score of 0.8
            min_score=0.8,
        )
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=strict_reranker,
            config=WebConfig(reranker_min_score=0.8),
        )

        result = pipeline.run("python release")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.evidence, [])
        self.assertIn("did not contain sufficiently relevant evidence", result.final_answer)
        self.assertEqual(self.gateway.generate.call_count, 0)

    def test_top_k_acts_as_maximum_ceiling(self) -> None:
        """13. top_k restricts the maximum number of evidence chunks accepted."""
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=WebConfig(rerank_top_k=1, reranker_min_score=0.1),
        )
        result = pipeline.run("python release")
        self.assertEqual(len(result.evidence), 1)

    def test_filtered_chunks_never_become_citations(self) -> None:
        """14. Filtered or unaccepted chunks never appear in citations or evidence."""
        # Config with tight context budget allowing only 1 chunk
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=self.search_provider,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=WebConfig(
                rerank_top_k=5,
                context_config=ContextConfig(max_chars=200),  # Fits only first chunk
            ),
        )
        result = pipeline.run("python release")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0], result.evidence[0].citation)

    def test_search_failure_returns_clear_failure_without_fabrication(self) -> None:
        """Search failure returns clear error without fabricating results or calling LLM."""
        failing_search = MockWebSearchProvider(simulated_failure=RuntimeError("SearXNG offline"))
        pipeline = DeterministicWebPipeline(
            gateway=self.gateway,
            search_provider=failing_search,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
        )

        result = pipeline.run("python release")
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertFalse(result.has_evidence)
        self.assertIn("Web research failed", result.final_answer)
        self.assertEqual(self.gateway.generate.call_count, 0)


class DeterministicHybridPipelineTests(TestCase):
    """Tests for DeterministicHybridPipeline (Requirements 16–20)."""

    def setUp(self) -> None:
        self.gateway = MockGatewayHelper.create(
            synthesis_text="Hybrid synthesis [Python Doc, Chunk: kb-1] [Python Releases, https://python.org/downloads, Chunk: web-1]."
        )
        self.search_provider = MockWebSearchProvider(
            canned_results={
                "python info": [
                    WebSearchResult(
                        title="Python Releases",
                        url="https://python.org/downloads",
                        snippet="Python 3.13 released October 2024.",
                        domain="python.org",
                    )
                ]
            }
        )
        self.fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": (
                    "<!DOCTYPE html><html><head><title>Python Releases</title></head>"
                    "<body><article><p>Python 3.13 was released recently on the web.</p></article></body></html>"
                )
            }
        )
        self.extractor = WebPageExtractor()
        self.reranker = MockReranker(
            scorer=lambda q, c: 0.95 if "web" in c.chunk_id else 0.85,
            min_score=0.1,
        )
        self.mock_embedding_provider = MagicMock()
        self.mock_embedding_provider.embed_query.return_value = [0.1] * 384

    @patch("rag.fast_web.retrieve_chunks")
    @patch("rag.fast_web.retrieve_lexical_chunks")
    def test_web_and_kb_evidence_can_coexist(
        self,
        mock_lexical: MagicMock,
        mock_dense: MagicMock,
    ) -> None:
        """16 & 17. Web and KB evidence coexist in context with exactly ONE generate call."""
        kb_chunk = RetrievalResult(
            chunk_id="kb-101",
            document_id="doc-1",
            content="Internal KB: Python is maintained by the Python Software Foundation.",
            score=0.88,
            rank=1,
            chunk_index=0,
            document_title="Internal Python Spec",
            document_source="spec.md",
            start_offset=0,
            end_offset=50,
            chunk_metadata={"source_type": "knowledge_base"},
            chunk=None,
        )
        mock_dense.return_value = [kb_chunk]
        mock_lexical.return_value = []

        pipeline = DeterministicHybridPipeline(
            gateway=self.gateway,
            embedding_provider=self.mock_embedding_provider,
            search_provider=self.search_provider,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
            config=HybridConfig(rerank_top_k=5, reranker_min_score=0.1),
        )

        result = pipeline.run("python info")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.has_evidence)
        self.assertEqual(self.gateway.generate.call_count, 1)

        # Verify evidence includes both web and KB items
        sources_types = {e.metadata.get("source_type") for e in result.evidence}
        self.assertIn("web", sources_types)
        self.assertIn("knowledge_base", sources_types)

        # Check citations
        web_ev = [e for e in result.evidence if e.metadata.get("source_type") == "web"][0]
        kb_ev = [e for e in result.evidence if e.metadata.get("source_type") == "knowledge_base"][0]
        self.assertIn("https://python.org/downloads", web_ev.citation)
        self.assertIn("Internal Python Spec", kb_ev.citation)

    @patch("rag.fast_web.retrieve_chunks")
    @patch("rag.fast_web.retrieve_lexical_chunks")
    def test_failed_web_does_not_destroy_valid_kb_evidence(
        self,
        mock_lexical: MagicMock,
        mock_dense: MagicMock,
    ) -> None:
        """18. When web search fails completely, valid KB evidence is still returned and synthesized."""
        kb_chunk = RetrievalResult(
            chunk_id="kb-202",
            document_id="doc-2",
            content="Internal KB: Knowledge base information persists.",
            score=0.85,
            rank=1,
            chunk_index=0,
            document_title="Persistent KB Doc",
            document_source="kb.txt",
            start_offset=0,
            end_offset=40,
            chunk_metadata={"source_type": "knowledge_base"},
            chunk=None,
        )
        mock_dense.return_value = [kb_chunk]
        mock_lexical.return_value = []

        failing_search = MockWebSearchProvider(simulated_failure=RuntimeError("SearXNG unreachable"))
        pipeline = DeterministicHybridPipeline(
            gateway=self.gateway,
            embedding_provider=self.mock_embedding_provider,
            search_provider=failing_search,
            fetcher=self.fetcher,
            extractor=self.extractor,
            reranker=self.reranker,
        )

        result = pipeline.run("python info")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.has_evidence)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].chunk_id, "kb-202")
        self.assertEqual(self.gateway.generate.call_count, 1)

    @patch("rag.fast_web.retrieve_chunks")
    @patch("rag.fast_web.retrieve_lexical_chunks")
    def test_no_autonomous_planner_loop_invoked(
        self,
        mock_lexical: MagicMock,
        mock_dense: MagicMock,
    ) -> None:
        """20. The hybrid and web pipelines do not invoke ResearchPlanner or iterative agent loops."""
        mock_dense.return_value = []
        mock_lexical.return_value = []

        with patch("agent.planning.research.ResearchPlanner") as mock_planner:
            pipeline = DeterministicHybridPipeline(
                gateway=self.gateway,
                embedding_provider=self.mock_embedding_provider,
                search_provider=self.search_provider,
                fetcher=self.fetcher,
                extractor=self.extractor,
                reranker=self.reranker,
            )
            pipeline.run("python info")
            mock_planner.assert_not_called()
