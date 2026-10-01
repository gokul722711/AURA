"""Tests for the Research REST API (M7)."""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from agent.planning.base import ActionType, AgentStep, Plan
from agent.research import ResearchEvidence, ResearchResult, ResearchRuntime
from agent.state import AgentState, AgentStatus
from agent.views import ResearchView
from gateway.base import LLMProvider
from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, GenerationResponse, ProviderMetadata, StructuredOutputRequest, StructuredOutputResponse, UsageInfo
from rag.models import Document
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document


from agent.tests.test_research import ScriptedLLMProvider


class ResearchAPITests(TestCase):
    """Test suite for POST /api/research/."""

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

    def test_valid_research_request(self) -> None:
        """Valid research request executes and returns HTTP 200 with completed status."""
        mock_result = ResearchResult(
            objective="How do components interact?",
            final_answer="The components interact via defined interfaces.",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=120.0,
        )

        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.return_value = mock_result
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "How do components interact?"},
                format="json",
            )

            self.assertEqual(response.status_code, 200)
            mock_runtime.run_research.assert_called_once_with("How do components interact?")
            data = response.json()
            self.assertEqual(data["objective"], "How do components interact?")
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["final_answer"], "The components interact via defined interfaces.")

    def test_successful_research_result_serialization(self) -> None:
        """Response exposes all canonical fields from ResearchResult.to_dict()."""
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

        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.return_value = mock_result
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "Analyze architecture"},
                format="json",
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify all canonical fields are present and match
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
        """API cleanly represents failure when unhandled exception or agent failure occurs."""
        # Case A: Uncaught runtime exception returns HTTP 500 with clean error payload
        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.side_effect = RuntimeError("Inference connection timed out")
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "Test failure handling"},
                format="json",
            )

            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["error"], "Research execution failed.")
            self.assertEqual(data["status"], "failed")
            self.assertIn("timed out", data["detail"])

        # Case B: Runtime returns a result with status FAILED and error messages
        failed_result = ResearchResult(
            objective="Exceed limits",
            final_answer="",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=5,
            has_evidence=False,
            status=AgentStatus.FAILED,
            duration_ms=500.0,
            errors=["Execution iteration limit (5) exceeded."],
        )
        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.return_value = failed_result
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "Exceed limits"},
                format="json",
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["errors"], ["Execution iteration limit (5) exceeded."])

    @override_settings(AI_GATEWAY={"API_KEY": "nvapi-confidential-secret-key-12345"})
    def test_api_does_not_expose_provider_secrets(self) -> None:
        """API must not leak configured or detected API keys in responses or errors."""
        secret_key = "nvapi-confidential-secret-key-12345"
        token_pattern = "sk-live-1234567890abcdef"

        # Case 1: Exception string contains API key and token pattern
        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.side_effect = RuntimeError(
                f"Connection failed for {secret_key} with header Bearer {token_pattern}"
            )
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "Test secret leakage"},
                format="json",
            )

            content = response.content.decode("utf-8")
            self.assertNotIn(secret_key, content)
            self.assertNotIn(token_pattern, content)
            self.assertIn("[REDACTED]", content)

        # Case 2: Result data contains accidental secret in errors or metadata
        secret_result = ResearchResult(
            objective="Secret result test",
            final_answer="Answer mentioning nothing secret.",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            errors=[f"Warn: failed probe with {secret_key}"],
            metadata={"auth": f"Bearer {token_pattern}"},
        )
        with patch.object(ResearchView, "get_runtime") as mock_get_runtime:
            mock_runtime = MagicMock()
            mock_runtime.run_research.return_value = secret_result
            mock_get_runtime.return_value = mock_runtime

            response = self.client.post(
                self.url,
                {"objective": "Secret result test"},
                format="json",
            )

            self.assertEqual(response.status_code, 200)
            content = response.content.decode("utf-8")
            self.assertNotIn(secret_key, content)
            self.assertNotIn(token_pattern, content)
            data = response.json()
            self.assertIn("[REDACTED]", data["errors"][0])
            self.assertIn("[REDACTED]", data["metadata"]["auth"])

    def test_end_to_end_research_execution(self) -> None:
        """End-to-end execution of ResearchRuntime via ResearchView with ingested knowledge."""
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

        with patch.object(ResearchView, "get_runtime", return_value=runtime):
            response = self.client.post(
                self.url,
                {"objective": "How do Model Gateway and RAG interact?"},
                format="json",
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "completed")
            self.assertTrue(data["has_evidence"])
            self.assertTrue(data["is_grounded"])
            self.assertGreaterEqual(len(data["evidence"]), 1)
            self.assertEqual(data["queries"], ["Model Gateway and RAG"])
            self.assertIn("Model Gateway routes requests", data["final_answer"])
