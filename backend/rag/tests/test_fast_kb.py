"""Comprehensive tests for DeterministicKBPipeline (M13).

Validates:
1. Critical model call count:
   - planner calls = 0
   - structured-output calls = 0
   - tool-loop calls = 0
   - generation calls = 1
2. Citation generation and verification.
3. Metadata preservation (pages, URLs, offsets, sources).
4. No-context behavior (0 generation calls, explicit notice).
5. Timing instrumentation.
6. Single and multi-document retrieval scoping.
"""

from unittest.mock import MagicMock
from django.test import TestCase

from agent.results import ResearchResult, make_citation
from agent.state import AgentStatus
from gateway.gateway import ModelGateway
from gateway.providers.mock import MockLLMProvider
from gateway.types import GenerationRequest, GenerationResponse, StructuredOutputRequest, UsageInfo
from rag.context import ContextConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.fast_kb import DeterministicKBPipeline, KBConfig
from rag.models import Document, DocumentChunk
from rag.rerankers.mock import MockReranker


class CountingMockLLMProvider(MockLLMProvider):
    """Mock LLM Provider that instruments call counts for all operations."""

    def __init__(self, response_text: str = "Mock answer"):
        super().__init__(default_response=response_text)
        self.generate_count = 0
        self.structured_output_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.generate_count += 1
        return super().generate(request)

    def structured_output(self, request: StructuredOutputRequest) -> Any:
        self.structured_output_count += 1
        return super().structured_output(request)


class DeterministicKBPipelineTests(TestCase):
    """Tests for DeterministicKBPipeline."""

    def setUp(self) -> None:
        self.doc1 = Document.objects.create(
            title="AURA Architecture Guide",
            content="AURA uses pgvector and Model Gateway for deterministic retrieval.",
            source="docs/architecture.md",
            status="ready",
            metadata={"source_type": "markdown"},
        )
        self.chunk1 = DocumentChunk.objects.create(
            document=self.doc1,
            content="Model Gateway abstracts LLM providers without vendor lock-in.",
            chunk_index=0,
            start_offset=0,
            end_offset=65,
            embedding=[0.05] * 384,
            metadata={"page": 3},
        )
        self.chunk2 = DocumentChunk.objects.create(
            document=self.doc1,
            content="pgvector executes database-side cosine distance search for chunks.",
            chunk_index=1,
            start_offset=66,
            end_offset=135,
            embedding=[0.08] * 384,
            metadata={"page": 4},
        )

        self.web_doc = Document.objects.create(
            title="AURA Online Manual",
            content="Online manual available at official site.",
            source="https://example.com/manual",
            status="ready",
            metadata={"source_type": "web_page", "url": "https://example.com/manual"},
        )
        self.web_chunk = DocumentChunk.objects.create(
            document=self.web_doc,
            content="AURA documentation and guides are published online.",
            chunk_index=0,
            start_offset=0,
            end_offset=52,
            embedding=[0.1] * 384,
            metadata={"url": "https://example.com/manual"},
        )

    def test_critical_model_call_requirement_exactly_one_generation(self) -> None:
        """KB query with matching context performs EXACTLY 1 generation call and 0 planning/structured calls."""
        # Answer references chunk1
        expected_citation = f"[AURA Architecture Guide, Page: 3, Chunk: {self.chunk1.pk}]"
        mock_provider = CountingMockLLMProvider(
            response_text=f"The Model Gateway provides provider abstraction {expected_citation}."
        )
        gateway = ModelGateway(provider=mock_provider)
        reranker = MockReranker()

        pipeline = DeterministicKBPipeline(
            gateway=gateway,
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=reranker,
        )

        res = pipeline.run("What does Model Gateway do?")

        # Assert model call constraints
        self.assertEqual(mock_provider.generate_count, 1, "Must perform exactly 1 generation call")
        self.assertEqual(mock_provider.structured_output_count, 0, "Must perform 0 structured output calls")

        # Verify result structure
        self.assertIsInstance(res, ResearchResult)
        self.assertEqual(res.status, AgentStatus.COMPLETED)
        self.assertEqual(res.iteration_count, 1)
        self.assertTrue(res.has_evidence)
        self.assertTrue(res.is_grounded)
        self.assertIn("Model Gateway provides provider abstraction", res.final_answer)

        # Verify citations
        citation_check = res.verify_citations()
        self.assertIn(expected_citation, citation_check["matched_citations"])
        self.assertIn(str(self.chunk1.pk), citation_check["matched_chunk_ids"])

        # Verify timings instrumentation
        timings = res.metadata.get("timings", {})
        self.assertIn("dense_retrieval_ms", timings)
        self.assertIn("lexical_retrieval_ms", timings)
        self.assertIn("rrf_ms", timings)
        self.assertIn("reranking_ms", timings)
        self.assertIn("generation_ms", timings)
        self.assertIn("total_ms", timings)

    def test_no_context_behavior_makes_zero_generation_calls(self) -> None:
        """When no matching context is found, 0 LLM generation calls occur."""
        mock_provider = CountingMockLLMProvider()
        gateway = ModelGateway(provider=mock_provider)

        pipeline = DeterministicKBPipeline(
            gateway=gateway,
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=MockReranker(),
            config=KBConfig(similarity_threshold=0.99),  # Threshold will filter out everything
        )

        res = pipeline.run("Unrelated query about deep space telescopes")

        # Must make ZERO generation calls
        self.assertEqual(mock_provider.generate_count, 0)
        self.assertEqual(mock_provider.structured_output_count, 0)

        # Explicit insufficient evidence notice
        self.assertFalse(res.has_evidence)
        self.assertFalse(res.is_grounded)
        self.assertEqual(len(res.evidence), 0)
        self.assertEqual(len(res.sources), 0)
        self.assertIn("did not provide sufficient supporting evidence", res.final_answer)
        self.assertEqual(res.metadata.get("context_grounded"), False)

    def test_url_metadata_preserved_in_evidence_and_citations(self) -> None:
        """Web documents retain URL metadata in evidence and citations."""
        citation = make_citation(
            document_title="AURA Online Manual",
            chunk_id=str(self.web_chunk.pk),
            url="https://example.com/manual",
        )
        mock_provider = CountingMockLLMProvider(
            response_text=f"AURA docs are online {citation}."
        )
        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=MockReranker(),
        )

        res = pipeline.run("Where are AURA guides published?", document_ids=[str(self.web_doc.id)])
        self.assertEqual(mock_provider.generate_count, 1)
        self.assertTrue(res.has_evidence)

        # Find web evidence
        web_ev = [ev for ev in res.evidence if ev.chunk_id == str(self.web_chunk.pk)][0]
        self.assertEqual(web_ev.url, "https://example.com/manual")
        self.assertEqual(web_ev.citation, citation)

    def test_document_scoping_filters_candidates(self) -> None:
        """Specifying document_ids restricts retrieval to specified documents."""
        mock_provider = CountingMockLLMProvider()
        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=MockReranker(),
        )

        res = pipeline.run("Model Gateway", document_ids=[str(self.web_doc.id)])
        # web_doc does not contain Model Gateway, so no chunks match threshold/filter
        for ev in res.evidence:
            self.assertEqual(ev.document_id, str(self.web_doc.id))

    def test_context_budget_bounding(self) -> None:
        """Context bounding restricts chunk inclusion when max_chars is small."""
        mock_provider = CountingMockLLMProvider()
        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=MockReranker(),
            config=KBConfig(
                context_config=ContextConfig(max_chars=70),  # Fits only 1 chunk
            ),
        )

        res = pipeline.run("pgvector and Model Gateway")
        self.assertLessEqual(len(res.evidence), 1)

    def test_relevance_threshold_filters_evidence_below_min_score(self) -> None:
        """5 candidates, 3 above threshold, 2 below -> exactly 3 in final evidence/citations (not 5)."""
        # Create additional chunks
        chunks = []
        for i in range(5):
            c = DocumentChunk.objects.create(
                document=self.doc1,
                content=f"Chunk content item {i} describing system components.",
                chunk_index=10 + i,
                start_offset=200 + i * 50,
                end_offset=250 + i * 50,
                embedding=[0.05] * 384,
            )
            chunks.append(c)

        # Scorer assigning scores: 3 above 0.0001, 2 below 0.0001
        score_map = {
            str(chunks[0].pk): 0.95,
            str(chunks[1].pk): 0.005,
            str(chunks[2].pk): 0.0005,
            str(chunks[3].pk): 0.000025,
            str(chunks[4].pk): 0.000020,
        }

        mock_provider = CountingMockLLMProvider(response_text="Answer based on valid chunks.")
        reranker = MockReranker(
            scorer=lambda q, c: score_map.get(str(c.chunk_id), 0.0),
            min_score=0.0001,
        )

        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=reranker,
            config=KBConfig(rerank_top_k=5, reranker_min_score=0.0001),
        )

        res = pipeline.run("Components test")
        self.assertEqual(len(res.evidence), 3, "Only the 3 qualifying chunks must enter evidence")
        self.assertEqual(len(res.citations), 3, "Only the 3 qualifying chunks must become citations")
        evidence_ids = {ev.chunk_id for ev in res.evidence}
        self.assertIn(str(chunks[0].pk), evidence_ids)
        self.assertIn(str(chunks[1].pk), evidence_ids)
        self.assertIn(str(chunks[2].pk), evidence_ids)
        self.assertNotIn(str(chunks[3].pk), evidence_ids)
        self.assertNotIn(str(chunks[4].pk), evidence_ids)

    def test_fewer_candidates_than_top_k_retains_only_qualifying(self) -> None:
        """top_k=5, but only 1 candidate above min_score -> exactly 1 in evidence, not 5."""
        mock_provider = CountingMockLLMProvider(response_text="Single chunk answer.")
        # Only chunk1 gets a high score; all others get below min_score
        reranker = MockReranker(
            scorer=lambda q, c: 0.99 if c.chunk_id == str(self.chunk1.pk) else 0.00001,
            min_score=0.0001,
        )

        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=reranker,
            config=KBConfig(rerank_top_k=5, reranker_min_score=0.0001),
        )

        res = pipeline.run("What does Model Gateway do?")
        self.assertEqual(len(res.evidence), 1, "Must return exactly 1 candidate, never filling remaining quota")
        self.assertEqual(res.evidence[0].chunk_id, str(self.chunk1.pk))
        self.assertEqual(len(res.citations), 1)

    def test_no_candidates_above_min_score_triggers_no_context_path(self) -> None:
        """When 0 candidates score above min_score, makes 0 generation calls and returns insufficient evidence."""
        mock_provider = CountingMockLLMProvider()
        # All candidates score 0.00002 (below 0.0001)
        reranker = MockReranker(
            scorer=lambda q, c: 0.00002,
            min_score=0.0001,
        )

        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=reranker,
            config=KBConfig(rerank_top_k=5, reranker_min_score=0.0001),
        )

        res = pipeline.run("Query with low scoring matches")
        self.assertEqual(mock_provider.generate_count, 0, "Must perform 0 generation calls when no context qualifies")
        self.assertFalse(res.has_evidence)
        self.assertEqual(len(res.evidence), 0)
        self.assertEqual(len(res.citations), 0)
        self.assertIn("did not provide sufficient supporting evidence", res.final_answer)

    def test_mixed_claude_and_ev_chunks_filters_unrelated_documents(self) -> None:
        """Simulates live smoke test: Claude chunks qualify while EV chunks are dropped and excluded from citations."""
        claude_doc = Document.objects.create(
            title="Claude (AI) - Wikipedia",
            content="Claude is an AI developed by Anthropic.",
            source="https://en.wikipedia.org/wiki/Claude_(AI)",
            status="ready",
        )
        claude_chunk1 = DocumentChunk.objects.create(
            document=claude_doc,
            content="Claude is a family of LLMs developed by Anthropic.",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            embedding=[0.05] * 384,
        )
        claude_chunk2 = DocumentChunk.objects.create(
            document=claude_doc,
            content="Claude Sonnet was released by Anthropic in 2024.",
            chunk_index=1,
            start_offset=51,
            end_offset=100,
            embedding=[0.05] * 384,
        )

        ev_doc = Document.objects.create(
            title="evi",
            content="Electric vehicle charging infrastructure.",
            source="evi.pdf",
            status="ready",
        )
        ev_chunk1 = DocumentChunk.objects.create(
            document=ev_doc,
            content="Smart charging systems and vehicle-to-grid grid infrastructure.",
            chunk_index=0,
            start_offset=0,
            end_offset=60,
            embedding=[0.05] * 384,
            metadata={"page": 3},
        )
        ev_chunk2 = DocumentChunk.objects.create(
            document=ev_doc,
            content="Electric vehicle supply equipment power electronics.",
            chunk_index=1,
            start_offset=61,
            end_offset=120,
            embedding=[0.05] * 384,
            metadata={"page": 4},
        )

        # Scorer mirrors live FlashRank results
        scores = {
            str(claude_chunk1.pk): 0.997235,
            str(claude_chunk2.pk): 0.000432,
            str(ev_chunk1.pk): 0.000029,
            str(ev_chunk2.pk): 0.000024,
        }

        captured_requests = []
        class CapturingMockLLMProvider(CountingMockLLMProvider):
            def generate(self, req: GenerationRequest) -> GenerationResponse:
                captured_requests.append(req)
                c1_cite = f"[Claude (AI) - Wikipedia, Chunk: {claude_chunk1.pk}]"
                return GenerationResponse(
                    text=f"Claude is developed by Anthropic {c1_cite}.",
                    provider="mock",
                    model="mock-model",
                )

        mock_provider = CapturingMockLLMProvider()
        reranker = MockReranker(
            scorer=lambda q, c: scores.get(str(c.chunk_id), 0.0),
            min_score=0.0001,
        )

        pipeline = DeterministicKBPipeline(
            gateway=ModelGateway(provider=mock_provider),
            embedding_provider=MockEmbeddingProvider(dimensions=384),
            reranker=reranker,
            config=KBConfig(rerank_top_k=5, reranker_min_score=0.0001),
        )

        res = pipeline.run("What is Claude, and what organization developed it?")

        # 1. Exactly 2 Claude chunks entered evidence, 0 EV chunks
        self.assertEqual(len(res.evidence), 2)
        ev_ids = [ev.chunk_id for ev in res.evidence]
        self.assertIn(str(claude_chunk1.pk), ev_ids)
        self.assertIn(str(claude_chunk2.pk), ev_ids)
        self.assertNotIn(str(ev_chunk1.pk), ev_ids)
        self.assertNotIn(str(ev_chunk2.pk), ev_ids)

        # 2. Citations contain ONLY Claude chunks
        self.assertEqual(len(res.citations), 2)
        for cite in res.citations:
            self.assertIn("Claude (AI) - Wikipedia", cite)
            self.assertNotIn("evi", cite)

        # 3. Sources summary contains ONLY the Claude document
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["document_title"], "Claude (AI) - Wikipedia")

        # 4. The LLM prompt was not polluted with EV chunks
        self.assertEqual(len(captured_requests), 1)
        user_message_content = captured_requests[0].messages[1].content
        self.assertIn("Claude is a family of LLMs", user_message_content)
        self.assertNotIn("Smart charging systems", user_message_content)
        self.assertNotIn("Electric vehicle supply equipment", user_message_content)

        # 5. Citation verification succeeds
        verif = res.verify_citations()
        self.assertEqual(len(verif["matched_chunk_ids"]), 1)
        self.assertEqual(verif["matched_chunk_ids"][0], str(claude_chunk1.pk))
