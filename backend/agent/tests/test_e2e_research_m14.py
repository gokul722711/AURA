"""End-to-end integration tests for AURA M14: Real Web Research & Web+KB."""

import json
from unittest.mock import MagicMock, patch
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from agent.models import ResearchRun
from agent.state import AgentStatus
from agent.tools.builtin.web import WebSearchResult
from gateway.types import GenerationResponse, ProviderMetadata, UsageInfo
from rag.rerankers.mock import MockReranker
from rag.retrieval import RetrievalResult
from rag.web.mock import MockWebFetcher


class MockGatewayHelper:
    @staticmethod
    def create(synthesis_text: str = "Synthesized response [Title, https://example.com, Chunk: web-1].") -> MagicMock:
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
            usage=UsageInfo(prompt_tokens=25, completion_tokens=20, total_tokens=45),
            finish_reason="stop",
        )
        return gateway


class E2EResearchM14Tests(TestCase):
    """End-to-end tests for M14 web and hybrid research APIs and execution."""

    def setUp(self) -> None:
        self.client = APIClient()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_e2e_web_mode_research(self) -> None:
        """21. mode=web executes through DeterministicWebPipeline end-to-end."""
        mock_gateway = MockGatewayHelper.create(
            "Python 3.13 was released on October 7, 2024 with performance improvements [Python Releases, https://python.org/downloads, Chunk: web-c1]."
        )
        mock_search_provider = MagicMock()
        mock_search_provider.search.return_value = [
            WebSearchResult(
                title="Python Releases",
                url="https://python.org/downloads",
                snippet="Python 3.13 released October 7, 2024.",
                domain="python.org",
            )
        ]
        mock_fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": (
                    "<!DOCTYPE html><html><head><title>Python Releases</title></head>"
                    "<body><article><p>Python 3.13 was released on October 7, 2024.</p></article></body></html>"
                )
            }
        )
        mock_reranker = MockReranker(scorer=lambda q, c: 0.9, min_score=0.1)

        with patch("rag.fast_web.get_gateway", return_value=mock_gateway), patch(
            "rag.fast_web.get_default_web_search_provider", return_value=mock_search_provider
        ), patch("rag.fast_web.HTTPXWebFetcher", return_value=mock_fetcher), patch(
            "rag.fast_web._get_default_reranker", return_value=mock_reranker
        ):
            resp = self.client.post(
                "/api/research/",
                data={
                    "objective": "What is the latest stable version of Python, and when was it released?",
                    "mode": "web",
                },
                format="json",
            )
            self.assertEqual(resp.status_code, 202)
            run_id = resp.data["run_id"]
            self.assertEqual(resp.data["mode"], "web")

            run = ResearchRun.objects.get(id=run_id)
            self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)
            self.assertEqual(run.mode, "web")
            self.assertEqual(run.result["metadata"]["pipeline"], "DeterministicWebPipeline")
            self.assertTrue(run.result["has_evidence"])
            self.assertTrue(run.result["is_grounded"])
            self.assertIn("Python 3.13", run.result["final_answer"])
            self.assertGreater(len(run.result["evidence"]), 0)
            self.assertTrue(run.result["evidence"][0]["url"].startswith("https://"))

            # Test detail endpoint
            detail_resp = self.client.get(f"/api/research/{run_id}/")
            self.assertEqual(detail_resp.status_code, 200)
            self.assertEqual(detail_resp.data["status"], "completed")
            self.assertEqual(detail_resp.data["mode"], "web")
            self.assertIn("evidence", detail_resp.data)
            self.assertIn("citations", detail_resp.data)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    @patch("rag.fast_web.retrieve_chunks")
    @patch("rag.fast_web.retrieve_lexical_chunks")
    def test_e2e_hybrid_mode_research(
        self,
        mock_lexical: MagicMock,
        mock_dense: MagicMock,
    ) -> None:
        """22. mode=web_knowledge_base executes through DeterministicHybridPipeline end-to-end."""
        mock_gateway = MockGatewayHelper.create(
            "Synthesis combining internal architecture and live web data [AURA Architecture, Chunk: kb-1] [Python Releases, https://python.org/downloads, Chunk: web-1]."
        )
        mock_search_provider = MagicMock()
        mock_search_provider.search.return_value = [
            WebSearchResult(
                title="Python Releases",
                url="https://python.org/downloads",
                snippet="Python 3.13 released October 2024.",
                domain="python.org",
            )
        ]
        mock_fetcher = MockWebFetcher(
            canned_responses={
                "https://python.org/downloads": (
                    "<!DOCTYPE html><html><head><title>Python Releases</title></head>"
                    "<body><article><p>Python 3.13 released recently.</p></article></body></html>"
                )
            }
        )
        kb_chunk = RetrievalResult(
            chunk_id="kb-arch-1",
            document_id="doc-arch",
            content="AURA Architecture overview: agentic AI with RAG and LLM Gateway.",
            score=0.92,
            rank=1,
            chunk_index=0,
            document_title="AURA Architecture",
            document_source="architecture.md",
            start_offset=0,
            end_offset=60,
            chunk_metadata={"source_type": "knowledge_base"},
            chunk=None,
        )
        mock_dense.return_value = [kb_chunk]
        mock_lexical.return_value = []
        mock_reranker = MockReranker(scorer=lambda q, c: 0.9, min_score=0.1)

        with patch("rag.fast_web.get_gateway", return_value=mock_gateway), patch(
            "rag.fast_web.get_default_web_search_provider", return_value=mock_search_provider
        ), patch("rag.fast_web.HTTPXWebFetcher", return_value=mock_fetcher), patch(
            "rag.fast_web._get_default_reranker", return_value=mock_reranker
        ):
            resp = self.client.post(
                "/api/research/",
                data={
                    "objective": "Summarize AURA architecture and compare with external Python developments",
                    "mode": "web_knowledge_base",
                },
                format="json",
            )
            self.assertEqual(resp.status_code, 202)
            run_id = resp.data["run_id"]
            self.assertEqual(resp.data["mode"], "web_knowledge_base")

            run = ResearchRun.objects.get(id=run_id)
            self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)
            self.assertEqual(run.result["metadata"]["pipeline"], "DeterministicHybridPipeline")
            self.assertTrue(run.result["has_evidence"])
            self.assertTrue(run.result["is_grounded"])
            self.assertIn("AURA Architecture", run.result["final_answer"])
            self.assertGreaterEqual(len(run.result["evidence"]), 2)
