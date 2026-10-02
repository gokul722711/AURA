"""Automated tests for Knowledge Base Document REST API (M8)."""

import uuid
from unittest.mock import MagicMock, patch

from django.test import TestCase
from rest_framework.test import APIClient

from agent.research import create_research_runtime
from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, GenerationResponse, ProviderMetadata, UsageInfo
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import EmbeddingError
from rag.models import Document, DocumentChunk
from rag.retrieval import RetrievalConfig, retrieve_chunks


class DocumentAPITests(TestCase):
    """Test suite for /api/documents/ and /api/documents/<id>/ endpoints."""

    def setUp(self) -> None:
        self.client = APIClient()
        self.list_url = "/api/documents/"

    # -------------------------------------------------------------------------
    # POST /api/documents/ (Create / Ingestion)
    # -------------------------------------------------------------------------

    def test_create_document_success(self) -> None:
        """POST with valid title and content ingests document and creates chunks."""
        payload = {
            "title": "AURA Architecture",
            "content": "AURA is an autonomous research and engineering agent designed to be LLM-agnostic.",
        }
        response = self.client.post(self.list_url, payload, format="json")
        self.assertEqual(response.status_code, 201)
        data = response.json()

        self.assertIn("id", data)
        self.assertEqual(data["title"], "AURA Architecture")
        self.assertEqual(data["content"], payload["content"])
        self.assertEqual(data["status"], "ready")
        self.assertGreaterEqual(data["chunk_count"], 1)
        self.assertIn("created_at", data)
        self.assertIn("updated_at", data)

        # Ensure no embedding vector data is exposed
        self.assertNotIn("embedding", data)
        self.assertNotIn("vector", data)

        # Verify DB persistence
        doc = Document.objects.get(id=data["id"])
        self.assertEqual(doc.title, "AURA Architecture")
        self.assertEqual(doc.status, Document.STATUS_READY)
        chunks = DocumentChunk.objects.filter(document=doc)
        self.assertEqual(chunks.count(), data["chunk_count"])

    def test_create_document_with_source_and_metadata(self) -> None:
        """POST with optional source and metadata persists correctly."""
        payload = {
            "title": "Research Notes",
            "content": "Detailed notes on multi-step reasoning and retrieval pipelines.",
            "source": "notes/reasoning.md",
            "metadata": {"author": "AURA Team", "tags": ["research", "agent"]},
        }
        response = self.client.post(self.list_url, payload, format="json")
        self.assertEqual(response.status_code, 201)
        data = response.json()

        self.assertEqual(data["source"], "notes/reasoning.md")
        self.assertEqual(data["metadata"], {"author": "AURA Team", "tags": ["research", "agent"]})

        doc = Document.objects.get(id=data["id"])
        self.assertEqual(doc.source, "notes/reasoning.md")
        self.assertEqual(doc.metadata["author"], "AURA Team")

    def test_create_document_invalid_body(self) -> None:
        """Non-dict request body returns HTTP 400."""
        import json
        response = self.client.post(
            self.list_url, json.dumps("not a json object"), content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertIn("JSON object expected", response.json()["error"])

    def test_create_document_missing_title(self) -> None:
        """Missing title field returns HTTP 400."""
        response = self.client.post(self.list_url, {"content": "Some content"}, format="json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required field: 'title'", response.json()["error"])

    def test_create_document_empty_title(self) -> None:
        """Empty or whitespace-only title returns HTTP 400."""
        for empty_val in ["", "   ", "\t\n  "]:
            response = self.client.post(
                self.list_url, {"title": empty_val, "content": "Some content"}, format="json"
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Field 'title' must be a non-empty string", response.json()["error"])

    def test_create_document_non_string_title(self) -> None:
        """Non-string title returns HTTP 400."""
        for val in [123, ["title"], {"k": "v"}, True, None]:
            response = self.client.post(
                self.list_url, {"title": val, "content": "Some content"}, format="json"
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Field 'title' must be a non-empty string", response.json()["error"])

    def test_create_document_missing_content(self) -> None:
        """Missing content field returns HTTP 400."""
        response = self.client.post(self.list_url, {"title": "Title"}, format="json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required field: 'content'", response.json()["error"])

    def test_create_document_empty_content(self) -> None:
        """Empty or whitespace-only content returns HTTP 400."""
        for empty_val in ["", "   ", "\t\n  "]:
            response = self.client.post(
                self.list_url, {"title": "Title", "content": empty_val}, format="json"
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Field 'content' must be a non-empty string", response.json()["error"])

    def test_create_document_non_string_content(self) -> None:
        """Non-string content returns HTTP 400."""
        for val in [123, ["content"], {"k": "v"}, True, None]:
            response = self.client.post(
                self.list_url, {"title": "Title", "content": val}, format="json"
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Field 'content' must be a non-empty string", response.json()["error"])

    def test_create_document_invalid_source_type(self) -> None:
        """Non-string source field returns HTTP 400."""
        response = self.client.post(
            self.list_url, {"title": "Title", "content": "Content", "source": 123}, format="json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Field 'source' must be a string", response.json()["error"])

    def test_create_document_invalid_metadata_type(self) -> None:
        """Non-dict metadata field returns HTTP 400."""
        response = self.client.post(
            self.list_url, {"title": "Title", "content": "Content", "metadata": "not-a-dict"}, format="json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Field 'metadata' must be an object", response.json()["error"])

    def test_create_document_ingestion_failure_handling(self) -> None:
        """When embedding fails, endpoint returns HTTP 500 without leaking secrets."""
        failing_provider = MagicMock()
        failing_provider.embed_texts.side_effect = EmbeddingError("Embedding service connection timed out")

        with patch("rag.views.DocumentListCreateView.get_embedding_provider", return_value=failing_provider):
            response = self.client.post(
                self.list_url,
                {"title": "Fail Doc", "content": "Some text to embed"},
                format="json",
            )
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"], "Document ingestion failed.")
            self.assertIn("detail", data)

    # -------------------------------------------------------------------------
    # GET /api/documents/ (List)
    # -------------------------------------------------------------------------

    def test_list_documents_empty(self) -> None:
        """GET /api/documents/ returns an empty list when no documents exist."""
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_list_documents_multiple(self) -> None:
        """GET /api/documents/ returns all documents with metadata and chunk counts."""
        self.client.post(
            self.list_url,
            {"title": "Doc 1", "content": "Content for document 1."},
            format="json",
        )
        self.client.post(
            self.list_url,
            {"title": "Doc 2", "content": "Content for document 2."},
            format="json",
        )

        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)

        # Most recent document should appear first
        self.assertEqual(data[0]["title"], "Doc 2")
        self.assertEqual(data[1]["title"], "Doc 1")

        for item in data:
            self.assertIn("id", item)
            self.assertIn("title", item)
            self.assertIn("status", item)
            self.assertIn("chunk_count", item)
            self.assertGreaterEqual(item["chunk_count"], 1)
            # Full content and embeddings should NOT be exposed in list
            self.assertNotIn("content", item)
            self.assertNotIn("embedding", item)

    # -------------------------------------------------------------------------
    # GET /api/documents/{id}/ (Retrieve)
    # -------------------------------------------------------------------------

    def test_get_document_success(self) -> None:
        """GET /api/documents/<id>/ returns single document details and content."""
        create_res = self.client.post(
            self.list_url,
            {"title": "Detail Doc", "content": "Full detailed content text."},
            format="json",
        )
        doc_id = create_res.json()["id"]

        detail_url = f"/api/documents/{doc_id}/"
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["id"], doc_id)
        self.assertEqual(data["title"], "Detail Doc")
        self.assertEqual(data["content"], "Full detailed content text.")
        self.assertEqual(data["status"], "ready")
        self.assertGreaterEqual(data["chunk_count"], 1)
        self.assertNotIn("embedding", data)

    def test_get_document_not_found(self) -> None:
        """GET /api/documents/<id>/ with unknown UUID returns HTTP 404."""
        unknown_id = str(uuid.uuid4())
        response = self.client.get(f"/api/documents/{unknown_id}/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "Document not found."})

    def test_get_document_invalid_uuid(self) -> None:
        """GET /api/documents/<id>/ with non-UUID string returns HTTP 404."""
        response = self.client.get("/api/documents/invalid-uuid-string/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "Document not found."})

    # -------------------------------------------------------------------------
    # DELETE /api/documents/{id}/ (Delete)
    # -------------------------------------------------------------------------

    def test_delete_document_success(self) -> None:
        """DELETE /api/documents/<id>/ removes document and its chunks cleanly."""
        create_res = self.client.post(
            self.list_url,
            {"title": "To Delete", "content": "Content to be deleted."},
            format="json",
        )
        doc_id = create_res.json()["id"]

        # Verify chunks exist before delete
        self.assertGreater(DocumentChunk.objects.filter(document_id=doc_id).count(), 0)

        detail_url = f"/api/documents/{doc_id}/"
        del_res = self.client.delete(detail_url)
        self.assertEqual(del_res.status_code, 204)

        # Verify document is deleted
        self.assertFalse(Document.objects.filter(id=doc_id).exists())

        # Verify chunks are cascade-deleted cleanly (no orphaned chunks)
        self.assertEqual(DocumentChunk.objects.filter(document_id=doc_id).count(), 0)

        # Verify subsequent GET returns 404
        get_res = self.client.get(detail_url)
        self.assertEqual(get_res.status_code, 404)

    def test_delete_document_not_found(self) -> None:
        """DELETE /api/documents/<id>/ with non-existent UUID returns HTTP 404."""
        unknown_id = str(uuid.uuid4())
        response = self.client.delete(f"/api/documents/{unknown_id}/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "Document not found."})

    def test_delete_document_invalid_uuid(self) -> None:
        """DELETE /api/documents/<id>/ with non-UUID string returns HTTP 404."""
        response = self.client.delete("/api/documents/not-a-valid-uuid/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "Document not found."})

    # -------------------------------------------------------------------------
    # Integration: Ingested document is immediately usable in RAG / Research
    # -------------------------------------------------------------------------

    def test_ingested_document_retrievable_by_rag(self) -> None:
        """Ingested document via API is immediately retrievable by RAG retrieve_chunks."""
        title = "Quantum Navigation Systems"
        content = "Quantum gyroscopes allow submarine navigation without GPS signals using trapped ions."
        create_res = self.client.post(
            self.list_url,
            {"title": title, "content": content},
            format="json",
        )
        self.assertEqual(create_res.status_code, 201)
        doc_id = create_res.json()["id"]

        # Run RAG retrieval
        embedding_provider = MockEmbeddingProvider(dimensions=384)
        results = retrieve_chunks(
            query="quantum gyroscopes submarine navigation",
            embedding_provider=embedding_provider,
            config=RetrievalConfig(top_k=5, similarity_threshold=0.0),
        )

        matching_result = next((r for r in results if str(r.document_id) == doc_id), None)
        self.assertIsNotNone(matching_result, "Ingested document was not retrieved by RAG.")
        self.assertEqual(matching_result.document_title, title)
        self.assertIn("trapped ions", matching_result.content)

    def test_end_to_end_research_with_api_ingested_document(self) -> None:
        """Document ingested via POST /api/documents/ is used by ResearchView to produce grounded answers."""
        import json
        from agent.tests.test_research import ScriptedLLMProvider
        from agent.views import ResearchView

        # 1. Ingest document via API
        title = "AURA Knowledge System"
        content = "The Knowledge Base allows users to ingest documents for RAG retrieval and research grounding."
        create_res = self.client.post(
            self.list_url,
            {"title": title, "content": content, "source": "manual-entry"},
            format="json",
        )
        self.assertEqual(create_res.status_code, 201)
        doc_id = create_res.json()["id"]

        # 2. Setup scripted provider for research loop
        scripted_provider = ScriptedLLMProvider(
            responses=[
                json.dumps({"decision": "continue", "query": "Knowledge Base and RAG"}),
                json.dumps({"decision": "finish"}),
                "The Knowledge Base enables document ingestion for RAG retrieval and research grounding [AURA Knowledge System, Chunk: 1].",
            ]
        )
        gateway = ModelGateway(provider=scripted_provider)
        runtime = create_research_runtime(
            gateway=gateway,
            embedding_provider=MockEmbeddingProvider(dimensions=384),
        )

        # 3. Call Research API
        with patch.object(ResearchView, "get_runtime", return_value=runtime):
            research_res = self.client.post(
                "/api/research/",
                {"objective": "What is the role of the Knowledge Base in AURA?"},
                format="json",
            )
            self.assertEqual(research_res.status_code, 200)
            data = research_res.json()
            self.assertEqual(data["status"], "completed")
            self.assertTrue(data["is_grounded"])
            self.assertTrue(data["has_evidence"])
            self.assertGreaterEqual(len(data["evidence"]), 1)
            self.assertEqual(data["evidence"][0]["document_title"], title)
            self.assertEqual(data["evidence"][0]["document_id"], doc_id)
            self.assertIn("Knowledge Base", data["final_answer"])
