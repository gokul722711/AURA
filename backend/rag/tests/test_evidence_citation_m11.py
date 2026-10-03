"""Deterministic unit tests verifying evidence and citation metadata preservation across the pipeline."""

from django.test import TestCase

from agent.results import (
    ResearchEvidence,
    ResearchResult,
    _aggregate_sources,
    make_citation,
)
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_file
from rag.models import Document
from rag.retrieval import RetrievalConfig, retrieve_chunks
from rag.tests.test_extraction import _make_docx, _make_pdf


class EvidenceCitationPreservationM11Tests(TestCase):
    """Tests for complete metadata survival from Document to ResearchEvidence and Citations."""

    def setUp(self) -> None:
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        Document.objects.all().delete()

    def test_pdf_page_metadata_pipeline(self) -> None:
        """Verify: PDF -> page metadata -> retrieval -> evidence -> citation."""
        pdf_bytes = _make_pdf([
            "Page 1: Principles of quantum computing and superposition.",
            "Page 2: Quantum entanglement and Einstein-Podolsky-Rosen paradox.",
        ])

        doc = ingest_file(
            file_bytes=pdf_bytes,
            filename="physics_report.pdf",
            title="Quantum Principles",
            embedding_provider=self.embedding_provider,
        )

        # 1. RetrievalResult retains page
        retrieval_results = retrieve_chunks(
            query="entanglement",
            embedding_provider=self.embedding_provider,
            config=RetrievalConfig(top_k=2),
        )
        self.assertGreater(len(retrieval_results), 0)
        match_result = next(r for r in retrieval_results if "entanglement" in r.content.lower())
        self.assertEqual(match_result.chunk_metadata.get("page"), 2)

        # 2. ResearchEvidence retains page and forms canonical citation
        evidence = ResearchEvidence.from_retrieval_result(match_result)
        self.assertEqual(evidence.page, 2)
        self.assertIn("Page: 2", evidence.citation)
        self.assertTrue(evidence.citation.startswith("[Quantum Principles, Page: 2, Chunk:"))

        # 3. Serialization retains page
        ev_dict = evidence.to_dict()
        self.assertEqual(ev_dict["page"], 2)

        # 4. ResearchResult aggregates pages in sources
        sources = _aggregate_sources([evidence])
        result = ResearchResult(
            objective="Explain entanglement",
            final_answer=f"As demonstrated in {evidence.citation}, quantum entanglement correlates states.",
            evidence=[evidence],
            sources=sources,
            queries=["entanglement"],
            iteration_count=1,
            has_evidence=True,
        )

        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["document_title"], "Quantum Principles")
        self.assertIn(2, result.sources[0]["pages"])

        # 5. Citation verification recognizes the page-inclusive citation
        verification = result.verify_citations()
        self.assertTrue(verification["has_evidence"])
        self.assertIn(evidence.citation, verification["matched_citations"])
        self.assertIn(evidence.chunk_id, verification["matched_chunk_ids"])

    def test_docx_structural_metadata_pipeline(self) -> None:
        """Verify: DOCX -> source metadata -> retrieval -> evidence -> citation."""
        docx_bytes = _make_docx(
            elements=[
                ("heading", "Autonomous System Requirements", "Heading 1"),
                ("paragraph", "AURA requires deterministic tool execution.", None),
            ],
            tables=[
                [["Metric", "Target"], ["Hit Rate", "95%"]]
            ],
        )

        doc = ingest_file(
            file_bytes=docx_bytes,
            filename="specs.docx",
            title="System Specifications",
            embedding_provider=self.embedding_provider,
        )

        retrieval_results = retrieve_chunks(
            query="requirements hit rate",
            embedding_provider=self.embedding_provider,
            config=RetrievalConfig(top_k=3),
        )
        self.assertGreater(len(retrieval_results), 0)

        evidence_list = [ResearchEvidence.from_retrieval_result(r) for r in retrieval_results]
        self.assertGreater(len(evidence_list), 0)

        # Verify citation format without page for non-paged documents
        for ev in evidence_list:
            self.assertIsNone(ev.page)
            self.assertTrue(ev.citation.startswith(f"[{ev.document_title}, Chunk: {ev.chunk_id}]"))

    def test_make_citation_backward_compatibility(self) -> None:
        """make_citation without page preserves exact [Title, Chunk: ID] format."""
        cite = make_citation("Architecture Overview", "chunk-101")
        self.assertEqual(cite, "[Architecture Overview, Chunk: chunk-101]")

        cite_paged = make_citation("Architecture Overview", "chunk-101", page=7)
        self.assertEqual(cite_paged, "[Architecture Overview, Page: 7, Chunk: chunk-101]")
