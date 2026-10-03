"""Tests for PostgreSQL Full-Text Search lexical retrieval in AURA (M13)."""

from django.test import TestCase

from rag.exceptions import RetrievalError
from rag.models import Document, DocumentChunk
from rag.retrieval import RetrievalConfig, retrieve_lexical_chunks


class LexicalRetrievalTests(TestCase):
    """Tests for retrieve_lexical_chunks."""

    def setUp(self) -> None:
        # Create ready document with chunks
        self.doc1 = Document.objects.create(
            title="PostgreSQL Features Guide",
            content="PostgreSQL offers robust full text search, indexing, and pgvector extension.",
            source="docs/postgres.md",
            status="ready",
        )
        self.chunk1 = DocumentChunk.objects.create(
            document=self.doc1,
            content="PostgreSQL full text search allows fast linguistic search with dictionaries and stemming.",
            chunk_index=0,
            start_offset=0,
            end_offset=90,
            embedding=[0.1] * 384,
            metadata={"page": 1},
        )
        self.chunk2 = DocumentChunk.objects.create(
            document=self.doc1,
            content="The pgvector extension enables vector embeddings storage and similarity search in SQL.",
            chunk_index=1,
            start_offset=91,
            end_offset=180,
            embedding=[0.2] * 384,
            metadata={"page": 2},
        )

        # Create second ready document
        self.doc2 = Document.objects.create(
            title="Django Web Framework",
            content="Django provides an ORM, migrations, and built-in postgres search helpers.",
            source="docs/django.md",
            status="ready",
        )
        self.chunk3 = DocumentChunk.objects.create(
            document=self.doc2,
            content="Django ORM integrates directly with PostgreSQL search vectors and search queries.",
            chunk_index=0,
            start_offset=0,
            end_offset=85,
            embedding=[0.3] * 384,
            metadata={"page": 1},
        )

        # Create unready document (should never be returned)
        self.unready_doc = Document.objects.create(
            title="Pending Draft",
            content="Draft notes about PostgreSQL full text search.",
            source="draft.txt",
            status="pending",
        )
        self.unready_chunk = DocumentChunk.objects.create(
            document=self.unready_doc,
            content="Draft notes about PostgreSQL full text search.",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            embedding=[0.4] * 384,
        )

    def test_lexical_search_matches_relevant_terms(self) -> None:
        """Query matching terms in ready documents returns matching chunks ordered by rank."""
        results = retrieve_lexical_chunks("linguistic stemming search")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, str(self.chunk1.pk))
        self.assertEqual(results[0].document_title, "PostgreSQL Features Guide")
        self.assertGreater(results[0].score, 0.0)

    def test_lexical_search_excludes_unready_documents(self) -> None:
        """Unready documents (status != ready) are excluded even if text matches."""
        results = retrieve_lexical_chunks("Draft notes")
        # unready_doc has status="pending", so no results should match
        self.assertEqual(len(results), 0)

    def test_lexical_search_unrelated_query_returns_empty(self) -> None:
        """Query with no lexical match in documents returns empty list."""
        results = retrieve_lexical_chunks("quantum astrophysics asteroid telescope")
        self.assertEqual(results, [])

    def test_empty_or_whitespace_query_raises_error(self) -> None:
        """Empty or whitespace query raises RetrievalError."""
        with self.assertRaises(RetrievalError):
            retrieve_lexical_chunks("")

        with self.assertRaises(RetrievalError):
            retrieve_lexical_chunks("   \n\t  ")

    def test_document_ids_scoping(self) -> None:
        """Restricting search to document_ids excludes matches from other documents."""
        cfg = RetrievalConfig(document_ids=[str(self.doc2.id)])
        # Both doc1 and doc2 mention PostgreSQL search, but document_ids restricts to doc2
        results = retrieve_lexical_chunks("PostgreSQL search", config=cfg)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_id, str(self.doc2.id))
        self.assertEqual(results[0].document_title, "Django Web Framework")

    def test_top_k_limiting(self) -> None:
        """top_k limit slices the candidate result list."""
        cfg = RetrievalConfig(top_k=1)
        results = retrieve_lexical_chunks("search", config=cfg)
        self.assertEqual(len(results), 1)

    def test_max_chunks_per_document_fairness(self) -> None:
        """max_chunks_per_document restricts chunks returned per source document."""
        cfg = RetrievalConfig(max_chunks_per_document=1, top_k=5)
        results = retrieve_lexical_chunks("PostgreSQL", config=cfg)
        doc_counts = {}
        for r in results:
            doc_counts[r.document_id] = doc_counts.get(r.document_id, 0) + 1
        for doc_id, count in doc_counts.items():
            self.assertLessEqual(count, 1)
