"""Deterministic unit tests for M11 retrieval improvements and fairness."""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.context import ContextConfig, assemble_context
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document, ingest_file
from rag.models import Document
from rag.retrieval import RetrievalConfig, retrieve_chunks
from rag.tests.test_extraction import _make_pdf


class RetrievalM11Tests(TestCase):
    """Tests for document filtering, multi-document fairness, and metadata preservation."""

    def setUp(self) -> None:
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        Document.objects.all().delete()

        # Ingest Document A (PDF with 2 pages)
        pdf_bytes = _make_pdf([
            "Quantum processors utilize superconducting qubits in superposition.",
            "Decoherence and thermal noise limit quantum gate coherence times.",
        ])
        self.doc_a = ingest_file(
            file_bytes=pdf_bytes,
            filename="quantum.pdf",
            title="Quantum Architecture",
            embedding_provider=self.embedding_provider,
        )

        # Ingest Document B (Plain text)
        self.doc_b = ingest_document(
            title="Classical Computing",
            content="Classical computing operates deterministically with binary bits 0 and 1.",
            embedding_provider=self.embedding_provider,
            source="classical.txt",
        )

        # Ingest Document C (Multi-chunk document)
        self.doc_c = ingest_document(
            title="Quantum Noise Mitigation",
            content=(
                "Error correction codes protect quantum memory from phase flips. "
                "Surface codes require high physical qubit overhead for logical qubits. "
                "Dynamical decoupling sequences suppress low-frequency environmental noise."
            ),
            embedding_provider=self.embedding_provider,
            chunking_config=ChunkingConfig(chunk_size=90, chunk_overlap=10),
            source="noise.txt",
        )

    def test_document_id_filtering(self) -> None:
        """Retrieval restricted to document_ids only returns chunks from those documents."""
        config = RetrievalConfig(
            top_k=10,
            document_ids=[str(self.doc_b.id)],
        )
        results = retrieve_chunks(
            query="computing architecture",
            embedding_provider=self.embedding_provider,
            config=config,
        )

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_b.id))
            self.assertEqual(r.document_title, "Classical Computing")

    def test_multi_document_fairness_max_chunks_per_document(self) -> None:
        """max_chunks_per_document prevents a single document from dominating results."""
        # doc_c has multiple chunks that match quantum queries
        config_unfair = RetrievalConfig(top_k=5, max_chunks_per_document=None)
        results_unfair = retrieve_chunks(
            query="quantum error noise qubits",
            embedding_provider=self.embedding_provider,
            config=config_unfair,
        )

        # Now apply fairness limit: at most 1 chunk per document
        config_fair = RetrievalConfig(top_k=5, max_chunks_per_document=1)
        results_fair = retrieve_chunks(
            query="quantum error noise qubits",
            embedding_provider=self.embedding_provider,
            config=config_fair,
        )

        # Count chunks per doc in fair results
        doc_counts: dict[str, int] = {}
        for r in results_fair:
            doc_counts[r.document_id] = doc_counts.get(r.document_id, 0) + 1

        for did, count in doc_counts.items():
            self.assertLessEqual(count, 1)

        # Multiple documents should be represented
        self.assertGreater(len(doc_counts), 1)

    def test_pdf_page_metadata_preserved_in_retrieval_result(self) -> None:
        """RetrievalResult chunk_metadata preserves PDF page numbers."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=[str(self.doc_a.id)],
        )
        results = retrieve_chunks(
            query="superposition qubits",
            embedding_provider=self.embedding_provider,
            config=config,
        )

        self.assertGreater(len(results), 0)
        res = results[0]
        self.assertEqual(res.chunk_metadata.get("source_type"), "pdf")
        self.assertIn("page", res.chunk_metadata)
        self.assertIn(res.chunk_metadata["page"], [1, 2])

    def test_context_assembly_includes_page_attribution(self) -> None:
        """Context assembler formats page numbers in source block headers."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=[str(self.doc_a.id)],
        )
        results = retrieve_chunks(
            query="qubits",
            embedding_provider=self.embedding_provider,
            config=config,
        )

        context_str = assemble_context(results, query="qubits")
        self.assertIn("Source: quantum.pdf | Page:", context_str)

    def test_context_assembly_respects_max_chunks_per_document(self) -> None:
        """ContextConfig.max_chunks_per_document caps chunks per doc in assembled context."""
        config = RetrievalConfig(top_k=10)
        results = retrieve_chunks(
            query="quantum",
            embedding_provider=self.embedding_provider,
            config=config,
        )

        context_cfg = ContextConfig(max_chunks_per_document=1)
        context_str = assemble_context(results, query="quantum", config=context_cfg)

        # Count how many times doc_c is cited in context
        self.assertLessEqual(context_str.count("Source: noise.txt"), 1)

    def test_deterministic_ordering_with_tie_breaker(self) -> None:
        """Deterministic ordering: distance ASC, then chunk pk ASC."""
        config = RetrievalConfig(top_k=10)
        results1 = retrieve_chunks("quantum physics", self.embedding_provider, config=config)
        results2 = retrieve_chunks("quantum physics", self.embedding_provider, config=config)

        self.assertEqual(
            [r.chunk_id for r in results1],
            [r.chunk_id for r in results2],
        )
