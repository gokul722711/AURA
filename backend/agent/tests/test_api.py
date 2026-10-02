"""Tests for the Research REST API (M9 Async)."""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from agent.models import ResearchRun
from agent.planning.base import ActionType, AgentStep, Plan
from agent.research import ResearchEvidence, ResearchResult, ResearchRuntime
from agent.state import AgentState, AgentStatus
from agent.tests.test_research import ScriptedLLMProvider
from agent.views import ResearchView
from gateway.base import LLMProvider
from gateway.gateway import ModelGateway
from gateway.types import (
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StructuredOutputRequest,
    StructuredOutputResponse,
    UsageInfo,
)
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document
from rag.models import Document


class ResearchAPITests(TestCase):
    """Test suite for POST /api/research/ and related endpoints."""

    def setUp(self) -> None:
        self.client = APIClient()
        self.url = "/api/research/"

    def test_missing_objective(self) -> None:
        """POST without objective field should return HTTP 400."""
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("Missing required field", data["error"])

    def test_empty_objective(self) -> None:
        """POST with empty string objective should return HTTP 400."""
        response = self.client.post(self.url, {"objective": ""}, format="json")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("cannot be empty or whitespace-only", data["error"])

    def test_whitespace_only_objective(self) -> None:
        """POST with whitespace-only objective should return HTTP 400."""
        response = self.client.post(self.url, {"objective": "   \n\t  "}, format="json")
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("cannot be empty or whitespace-only", data["error"])

    def test_non_string_objective(self) -> None:
        """POST with non-string objective should return HTTP 400."""
        invalid_values = [12345, ["invalid"], {"key": "val"}, None, True]
        for val in invalid_values:
            response = self.client.post(self.url, {"objective": val}, format="json")
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn("error", data)
            self.assertIn("must be a string", data["error"])

    def test_non_dict_body(self) -> None:
        """POST with non-object JSON body should return HTTP 400."""
        response = self.client.post(
            self.url,
            data=json.dumps("not a dictionary"),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("JSON object expected", data["error"])

    def test_valid_research_request_returns_202_accepted(self) -> None:
        """Valid research request creates a ResearchRun, enqueues task, and returns HTTP 202."""
        with patch.object(ResearchView, "dispatch_task") as mock_dispatch:
            response = self.client.post(
                self.url,
                {"objective": "How do components interact?"},
                format="json",
            )

            self.assertEqual(response.status_code, 202)
            data = response.json()
            self.assertIn("run_id", data)
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["objective"], "How do components interact?")
            mock_dispatch.assert_called_once_with(data["run_id"])

            run = ResearchRun.objects.get(id=data["run_id"])
            self.assertEqual(run.objective, "How do components interact?")
            self.assertEqual(run.status, ResearchRun.STATUS_QUEUED)

    def test_successful_research_result_serialization_via_detail_api(self) -> None:
        """Detail endpoint exposes all canonical fields from ResearchResult.to_dict()."""
        evidence_item = ResearchEvidence(
            chunk_id="chunk-test-1",
            document_title="Architecture Guide",
            document_source="docs/arch.md",
            content="Django routes to ResearchView.",
            score=0.95,
            document_id="doc-uuid-1",
        )
        sources_list = [
            {
                "document_title": "Architecture Guide",
                "document_source": "docs/arch.md",
                "document_id": "doc-uuid-1",
                "chunk_count": 1,
                "chunk_ids": ["chunk-test-1"],
            }
        ]
        mock_result = ResearchResult(
            objective="Analyze architecture",
            final_answer="Django routes to ResearchView [Architecture Guide, Chunk: chunk-test-1].",
            evidence=[evidence_item],
            sources=sources_list,
            queries=["architecture guide"],
            iteration_count=2,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=350.5,
            errors=[],
            metadata={"source": "api_test"},
        )

        run = ResearchRun.objects.create(
            objective="Analyze architecture",
            status=ResearchRun.STATUS_COMPLETED,
            result=mock_result.to_dict(),
            duration_ms=350.5,
        )

        response = self.client.get(f"/api/research/{run.id}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["run_id"], str(run.id))
        self.assertEqual(data["objective"], "Analyze architecture")
        self.assertEqual(data["final_answer"], "Django routes to ResearchView [Architecture Guide, Chunk: chunk-test-1].")
        self.assertEqual(len(data["evidence"]), 1)
        self.assertEqual(data["evidence"][0]["chunk_id"], "chunk-test-1")
        self.assertEqual(data["evidence"][0]["document_title"], "Architecture Guide")
        self.assertEqual(data["evidence"][0]["score"], 0.95)
        self.assertEqual(data["evidence"][0]["citation"], "[Architecture Guide, Chunk: chunk-test-1]")
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["document_title"], "Architecture Guide")
        self.assertEqual(data["sources"][0]["chunk_ids"], ["chunk-test-1"])
        self.assertEqual(data["citations"], ["[Architecture Guide, Chunk: chunk-test-1]"])
        self.assertEqual(data["queries"], ["architecture guide"])
        self.assertEqual(data["iteration_count"], 2)
        self.assertTrue(data["has_evidence"])
        self.assertTrue(data["is_grounded"])
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["duration_ms"], 350.5)
        self.assertEqual(data["errors"], [])
        self.assertEqual(data["metadata"], {"source": "api_test"})

    def test_research_failure_response(self) -> None:
        """API cleanly represents failure when unhandled exception occurs."""
        run = ResearchRun.objects.create(
            objective="Exceed limits",
            status=ResearchRun.STATUS_FAILED,
            error_message="Execution iteration limit (5) exceeded.",
        )
        response = self.client.get(f"/api/research/{run.id}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["error"], "Execution iteration limit (5) exceeded.")
        self.assertEqual(data["errors"], ["Execution iteration limit (5) exceeded."])

    @override_settings(AI_GATEWAY={"API_KEY": "nvapi-confidential-secret-key-12345"})
    def test_api_does_not_expose_provider_secrets(self) -> None:
        """API must not leak configured or detected API keys in responses or errors."""
        secret_key = "nvapi-confidential-secret-key-12345"
        token_pattern = "sk-live-1234567890abcdef"

        run = ResearchRun.objects.create(
            objective="Secret result test",
            status=ResearchRun.STATUS_FAILED,
            error_message=f"Connection failed for {secret_key} with header Bearer {token_pattern}",
        )

        response = self.client.get(f"/api/research/{run.id}/")
        content = response.content.decode("utf-8")
        self.assertNotIn(secret_key, content)
        self.assertNotIn(token_pattern, content)
        self.assertIn("[REDACTED]", content)

    def test_end_to_end_async_research_execution(self) -> None:
        """End-to-end execution of async research with eager Celery task."""
        Document.objects.all().delete()
        ingest_document(
            title="AURA Subsystems",
            source="docs/subsystems.md",
            content="The Model Gateway routes requests to LLM providers. RAG retrieves relevant document chunks.",
            embedding_provider=MockEmbeddingProvider(dimensions=384),
        )

        scripted_provider = ScriptedLLMProvider(
            responses=[
                json.dumps({"decision": "continue", "query": "Model Gateway and RAG"}),
                json.dumps({"decision": "finish"}),
                "The Model Gateway routes requests to providers while RAG handles retrieval [AURA Subsystems, Chunk: 1].",
            ]
        )
        gateway = ModelGateway(provider=scripted_provider)

        from agent.research import create_research_runtime
        runtime = create_research_runtime(
            gateway=gateway,
            embedding_provider=MockEmbeddingProvider(dimensions=384),
        )

        with override_settings(CELERY_TASK_ALWAYS_EAGER=True), patch(
            "agent.tasks.create_research_runtime", return_value=runtime
        ):
            post_res = self.client.post(
                self.url,
                {"objective": "How do Model Gateway and RAG interact?"},
                format="json",
            )
            self.assertEqual(post_res.status_code, 202)
            run_id = post_res.json()["run_id"]

            get_res = self.client.get(f"/api/research/{run_id}/")
            self.assertEqual(get_res.status_code, 200)
            data = get_res.json()
            self.assertEqual(data["status"], "completed")
            self.assertTrue(data["has_evidence"])
            self.assertTrue(data["is_grounded"])
            self.assertGreaterEqual(len(data["evidence"]), 1)
            self.assertEqual(data["queries"], ["Model Gateway and RAG"])
            self.assertIn("Model Gateway routes requests", data["final_answer"])

    def test_post_trailing_slash_route_requirement(self) -> None:
        """POST to /api/research/ succeeds with 202."""
        with patch.object(ResearchView, "dispatch_task"):
            res_with_slash = self.client.post(
                "/api/research/",
                {"objective": "Trailing slash test"},
                format="json",
            )
            self.assertEqual(res_with_slash.status_code, 202)

        try:
            res_without_slash = self.client.post(
                "/api/research",
                {"objective": "Trailing slash test"},
                format="json",
            )
            self.assertIn(res_without_slash.status_code, (301, 404, 500))
        except RuntimeError as exc:
            self.assertIn("APPEND_SLASH", str(exc))
