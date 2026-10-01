"""Tests for RAGSearchTool."""

from django.test import TestCase

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
