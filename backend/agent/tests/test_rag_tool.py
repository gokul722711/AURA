"""Tests for RAGSearchTool."""

from django.test import TestCase, override_settings

from agent.tools.builtin.rag import RAGSearchTool
from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalConfig


class RAGSearchToolTests(TestCase):
    """Tests for RAGSearchTool integration across boundary."""

    def setUp(self):
        self.provider = MockEmbeddingProvider(dimensions=384)
        self.doc = ingest_document(
            title="Quantum Computing Guide",
            content="Quantum computers utilize qubits for superposition and entanglement. " * 5,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=80, chunk_overlap=10),
            source="quantum.pdf",
        )
        self.tool = RAGSearchTool(
            embedding_provider=self.provider,
            config=RetrievalConfig(top_k=3),
        )

    def test_tool_metadata(self):
        self.assertEqual(self.tool.name, "rag_search")
        self.assertIn("knowledge base", self.tool.description)
        schema = self.tool.input_schema
        self.assertIn("query", schema["properties"])
        self.assertEqual(schema["required"], ["query"])

    def test_execute_successful_search(self):
        res = self.tool.execute(query="quantum computers qubits")
        self.assertFalse(res.is_error)
        self.assertIsInstance(res.output, list)
        self.assertGreater(len(res.output), 0)
        self.assertLessEqual(len(res.output), 3)

        first_chunk = res.output[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertEqual(first_chunk["document_title"], "Quantum Computing Guide")
        self.assertEqual(first_chunk["document_source"], "quantum.pdf")
        self.assertIn("score", first_chunk)
        self.assertIn("content", first_chunk)

    def test_execute_with_custom_top_k(self):
        res = self.tool.execute(query="superposition", top_k=1)
        self.assertFalse(res.is_error)
        self.assertEqual(len(res.output), 1)

    def test_missing_or_empty_query_rejected(self):
        res = self.tool.execute()
        self.assertTrue(res.is_error)
        self.assertIn("must be a non-empty string", res.error_message)

        res = self.tool.execute(query="")
        self.assertTrue(res.is_error)

    def test_invalid_top_k_rejected(self):
        res = self.tool.execute(query="test", top_k=0)
        self.assertTrue(res.is_error)
        self.assertIn("positive integer", res.error_message)

        res = self.tool.execute(query="test", top_k=-5)
        self.assertTrue(res.is_error)

    def test_rag_search_via_agent_runtime(self):
        """Verify RAGSearchTool executing end-to-end within AgentRuntime."""
        from agent.planning.base import ActionType, AgentStep
        from agent.planning.mock import MockPlanner
        from agent.runtime import AgentRuntime
        from agent.state import AgentStatus
        from agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self.tool)

        steps = [
            AgentStep(
                step_id="step-rag",
                action_type=ActionType.TOOL,
                description="Search documents",
                payload={"tool_name": "rag_search", "tool_input": {"query": "qubits"}},
            ),
            AgentStep(
                step_id="step-finish",
                action_type=ActionType.FINISH,
                description="Finish",
                payload={"final_answer": "RAG search completed."},
            ),
        ]
        planner = MockPlanner(steps=steps)
        runtime = AgentRuntime(planner=planner, registry=registry)

        result = runtime.run("Search quantum computing")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(len(result.state.tool_results), 1)
        self.assertEqual(result.state.tool_results[0]["tool_name"], "rag_search")
        self.assertGreater(len(result.state.tool_results[0]["output"]), 0)

    def test_non_uuid_document_identifier_resolution(self):
        """RAGSearchTool resolves non-UUID title identifiers cleanly."""
        res = self.tool.execute(query="quantum", document_ids=["Quantum Computing Guide"])
        self.assertFalse(res.is_error)
        self.assertGreater(len(res.output), 0)
        self.assertEqual(res.output[0]["document_title"], "Quantum Computing Guide")

    def test_url_metadata_surfaced(self):
        """RAGSearchTool surfaces URL metadata for web-sourced documents."""
        web_doc = ingest_document(
            title="Cristiano Ronaldo",
            content="Cristiano Ronaldo dos Santos Aveiro is a Portuguese professional footballer.",
            embedding_provider=self.provider,
            source="https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
            metadata={"url": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"},
        )
        res = self.tool.execute(query="footballer", document_ids=["Cristiano Ronaldo"])
        self.assertFalse(res.is_error)
        self.assertGreater(len(res.output), 0)
        first = res.output[0]
        self.assertEqual(first["url"], "https://en.wikipedia.org/wiki/Cristiano_Ronaldo")
        self.assertEqual(first["metadata"]["url"], "https://en.wikipedia.org/wiki/Cristiano_Ronaldo")

    @override_settings(AI_RAG={"TOP_K": 8, "SIMILARITY_THRESHOLD": 0.25, "MAX_CHUNKS_PER_DOC": 2})
    def test_default_retrieval_configuration_from_settings(self):
        """RAGSearchTool default configuration respects existing settings (AI_RAG)."""
        tool = RAGSearchTool(embedding_provider=self.provider)
        self.assertEqual(tool.config.top_k, 8)
        self.assertEqual(tool.config.similarity_threshold, 0.25)
        self.assertEqual(tool.config.max_chunks_per_document, 2)

    def test_per_document_fairness_applied(self):
        """RAGSearchTool default config applies MAX_CHUNKS_PER_DOC fairness."""
        # Ingest a second document with many chunks matching qubits
        ingest_document(
            title="Second Quantum Guide",
            content="Qubits and quantum states in physical quantum processors. " * 15,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=40, chunk_overlap=5),
        )
        # Default tool with fairness enabled
        tool = RAGSearchTool(embedding_provider=self.provider)
        res = tool.execute(query="qubits quantum", top_k=6)
        self.assertFalse(res.is_error)
        doc_counts = {}
        for chunk in res.output:
            doc_counts[chunk["document_id"]] = doc_counts.get(chunk["document_id"], 0) + 1
        for did, count in doc_counts.items():
            self.assertLessEqual(count, 2)
        # Verify both documents are represented
        self.assertGreater(len(doc_counts), 1)

    def test_single_document_identifier_bypasses_fairness_cap(self):
        """When document_ids restricts to one document, RAGSearchTool returns up to top_k chunks from it."""
        doc = ingest_document(
            title="Single Target Guide",
            content="Qubits and superposition principles in quantum circuits. " * 15,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=40, chunk_overlap=5),
        )
        # Default tool with fairness enabled in config (MAX_CHUNKS_PER_DOC=2)
        tool = RAGSearchTool(embedding_provider=self.provider)
        res = tool.execute(query="qubits quantum circuits", top_k=5, document_ids=[str(doc.id)])
        self.assertFalse(res.is_error)
        self.assertEqual(len(res.output), 5)
        for chunk in res.output:
            self.assertEqual(chunk["document_id"], str(doc.id))
