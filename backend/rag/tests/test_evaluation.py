"""Tests for RAG evaluation framework.

Tests cover:
- Recall@K computation and edge cases
- Precision@K computation and edge cases
- Hit Rate@K computation and edge cases
- MRR computation and edge cases
- Empty retrieved results
- No expected relevant results
- Deterministic evaluation
- Evaluator integration with retrieval pipeline
- Evaluation dataset representation
"""

from django.test import SimpleTestCase, TestCase

from rag.evaluation.dataset import EvaluationDataset, EvaluationExample
from rag.evaluation.metrics import (
    hit_rate_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)
from rag.exceptions import EvaluationError


class RecallAtKTests(SimpleTestCase):
    """Tests for recall_at_k metric."""

    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 1.0)

    def test_partial_recall(self):
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b", "c", "d"}
        # 2 hits out of 4 relevant = 0.5
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.5)

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.0)

    def test_recall_with_k_less_than_retrieved(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        # Only top-2 considered: a, b → 2/2 = 1.0
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 1.0)

    def test_recall_with_k_greater_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # 2 hits out of 3 relevant = 0.667
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=10), 2 / 3)

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=5), 0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=5), 0.0)

    def test_duplicate_retrieved_ids_does_not_inflate_recall(self):
        retrieved = ["a", "a", "b"]
        relevant = {"a", "b"}
        # Top-2 has ["a", "a"] -> only 1 unique relevant item ("a"), out of 2 relevant = 0.5
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 0.5)

    def test_duplicate_retrieved_ids_cannot_exceed_one(self):
        retrieved = ["a", "a", "a"]
        relevant = {"a"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 1.0)

    def test_duplicate_relevant_ids_as_list(self):
        retrieved = ["a"]
        relevant = ["a", "a"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=1), 1.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(EvaluationError):
            recall_at_k(["a"], {"a"}, k=0)

    def test_negative_k_raises(self):
        with self.assertRaises(EvaluationError):
            recall_at_k(["a"], {"a"}, k=-1)


class PrecisionAtKTests(SimpleTestCase):
    """Tests for precision_at_k metric."""

    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 1.0)

    def test_partial_precision(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        # 2 hits / k=3 = 0.667
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 2 / 3)

    def test_zero_precision(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 0.0)

    def test_precision_with_fewer_retrieved_than_k(self):
        retrieved = ["a"]
        relevant = {"a", "b"}
        # 1 hit / k=5 = 0.2
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.2)

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a", "b"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.0)

    def test_duplicate_retrieved_ids_precision(self):
        retrieved = ["a", "a"]
        relevant = {"a"}
        # 1 unique relevant item retrieved across k=2 slots -> 1/2 = 0.5
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=2), 0.5)

    def test_duplicate_relevant_ids_as_list(self):
        retrieved = ["a"]
        relevant = ["a", "a"]
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=1), 1.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(EvaluationError):
            precision_at_k(["a"], {"a"}, k=0)


class HitRateAtKTests(SimpleTestCase):
    """Tests for hit_rate_at_k metric."""

    def test_hit(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=3), 1.0)

    def test_no_hit(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=3), 0.0)

    def test_hit_at_position_one(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=1), 1.0)

    def test_no_hit_at_k_one(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=1), 0.0)

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=5), 0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=5), 0.0)

    def test_duplicate_retrieved_ids(self):
        retrieved = ["a", "a"]
        relevant = {"a"}
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=2), 1.0)

    def test_duplicate_relevant_ids_as_list(self):
        retrieved = ["a"]
        relevant = ["a", "a"]
        self.assertAlmostEqual(hit_rate_at_k(retrieved, relevant, k=1), 1.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(EvaluationError):
            hit_rate_at_k(["a"], {"a"}, k=0)


class MRRTests(SimpleTestCase):
    """Tests for mrr (Mean Reciprocal Rank) metric."""

    def test_first_position(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 1.0)

    def test_second_position(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.5)

    def test_third_position(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 1 / 3)

    def test_no_relevant_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.0)

    def test_multiple_relevant_returns_first(self):
        retrieved = ["x", "a", "b", "y"]
        relevant = {"a", "b"}
        # First relevant is at position 2
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.5)

    def test_duplicate_retrieved_ids(self):
        retrieved = ["a", "a"]
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 1.0)

    def test_duplicate_relevant_ids_as_list(self):
        retrieved = ["x", "a"]
        relevant = ["a", "a"]
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.5)

    def test_empty_retrieved(self):
        retrieved = []
        relevant = {"a"}
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        self.assertAlmostEqual(mrr(retrieved, relevant), 0.0)


class DeterministicEvaluationTests(SimpleTestCase):
    """Tests verifying evaluation is deterministic."""

    def test_recall_deterministic(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        r1 = recall_at_k(retrieved, relevant, k=3)
        r2 = recall_at_k(retrieved, relevant, k=3)
        self.assertEqual(r1, r2)

    def test_precision_deterministic(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        r1 = precision_at_k(retrieved, relevant, k=3)
        r2 = precision_at_k(retrieved, relevant, k=3)
        self.assertEqual(r1, r2)

    def test_hit_rate_deterministic(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        r1 = hit_rate_at_k(retrieved, relevant, k=3)
        r2 = hit_rate_at_k(retrieved, relevant, k=3)
        self.assertEqual(r1, r2)

    def test_mrr_deterministic(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        r1 = mrr(retrieved, relevant)
        r2 = mrr(retrieved, relevant)
        self.assertEqual(r1, r2)


class EvaluationDatasetTests(SimpleTestCase):
    """Tests for evaluation dataset representation."""

    def test_example_creation(self):
        ex = EvaluationExample(
            query="test query",
            relevant_ids=frozenset({"id1", "id2"}),
            description="Test example",
        )
        self.assertEqual(ex.query, "test query")
        self.assertEqual(len(ex.relevant_ids), 2)
        self.assertIn("id1", ex.relevant_ids)

    def test_example_deduplicates_relevant_ids_from_list(self):
        ex = EvaluationExample(
            query="test query",
            relevant_ids=["id1", "id1", "id2"],
        )
        self.assertIsInstance(ex.relevant_ids, frozenset)
        self.assertEqual(len(ex.relevant_ids), 2)
        self.assertEqual(ex.relevant_ids, frozenset({"id1", "id2"}))

    def test_dataset_creation(self):
        examples = (
            EvaluationExample(
                query="q1",
                relevant_ids=frozenset({"a"}),
            ),
            EvaluationExample(
                query="q2",
                relevant_ids=frozenset({"b", "c"}),
            ),
        )
        ds = EvaluationDataset(name="test-ds", examples=examples)
        self.assertEqual(ds.name, "test-ds")
        self.assertEqual(len(ds), 2)

    def test_dataset_iteration(self):
        examples = (
            EvaluationExample(query="q1", relevant_ids=frozenset({"a"})),
        )
        ds = EvaluationDataset(name="test", examples=examples)
        items = list(ds)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].query, "q1")

    def test_example_is_frozen(self):
        ex = EvaluationExample(query="q", relevant_ids=frozenset({"a"}))
        with self.assertRaises(AttributeError):
            ex.query = "modified"

    def test_dataset_is_frozen(self):
        ds = EvaluationDataset(name="test", examples=())
        with self.assertRaises(AttributeError):
            ds.name = "modified"

    def test_test_dataset_loads(self):
        """The repository-local test dataset should load correctly."""
        from rag.evaluation.test_dataset import get_test_evaluation_dataset
        ds = get_test_evaluation_dataset()
        self.assertEqual(ds.name, "test-evaluation-v1")
        self.assertGreater(len(ds), 0)
        for ex in ds:
            self.assertIsInstance(ex.query, str)
            self.assertGreater(len(ex.query), 0)
            self.assertIsInstance(ex.relevant_ids, frozenset)
            self.assertGreater(len(ex.relevant_ids), 0)


class EvaluatorIntegrationTests(TestCase):
    """Integration tests for the evaluator with actual retrieval.

    Uses MockEmbeddingProvider. These tests verify the evaluation pipeline
    mechanics, NOT semantic retrieval quality.
    """

    def setUp(self):
        from rag.embeddings.mock import MockEmbeddingProvider
        from rag.ingestion import ingest_document
        from rag.chunking import ChunkingConfig

        self.provider = MockEmbeddingProvider(dimensions=384)

        # Ingest a document to have chunks available
        self.document = ingest_document(
            title="Eval Test Doc",
            content="Evaluation testing content for the evaluator. " * 10,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        )

    def test_evaluator_runs_without_error(self):
        """The evaluator should complete without errors."""
        from rag.evaluation.evaluator import evaluate_retrieval
        from rag.evaluation.dataset import EvaluationDataset, EvaluationExample
        from rag.models import DocumentChunk

        # Get actual chunk IDs from the ingested document
        chunk_ids = list(
            DocumentChunk.objects.filter(document=self.document)
            .order_by("chunk_index")
            .values_list("pk", flat=True)
        )

        if not chunk_ids:
            return

        # Use real chunk IDs as relevant (at least one exists)
        ds = EvaluationDataset(
            name="integration-test",
            examples=(
                EvaluationExample(
                    query="evaluation testing",
                    relevant_ids=frozenset({str(chunk_ids[0])}),
                ),
            ),
        )

        result = evaluate_retrieval(
            dataset=ds,
            embedding_provider=self.provider,
            k=5,
        )

        self.assertEqual(result.dataset_name, "integration-test")
        self.assertEqual(result.num_examples, 1)
        self.assertEqual(result.k, 5)
        self.assertIsInstance(result.mean_recall, float)
        self.assertIsInstance(result.mean_precision, float)
        self.assertIsInstance(result.mean_hit_rate, float)
        self.assertIsInstance(result.mean_mrr, float)

    def test_evaluator_empty_dataset_raises(self):
        """Evaluator should raise on empty dataset."""
        from rag.evaluation.evaluator import evaluate_retrieval
        from rag.evaluation.dataset import EvaluationDataset

        ds = EvaluationDataset(name="empty", examples=())
        with self.assertRaises(EvaluationError):
            evaluate_retrieval(ds, self.provider, k=5)

    def test_evaluator_invalid_k_raises(self):
        """Evaluator should raise on k <= 0."""
        from rag.evaluation.evaluator import evaluate_retrieval
        from rag.evaluation.dataset import EvaluationDataset, EvaluationExample

        ds = EvaluationDataset(
            name="test",
            examples=(
                EvaluationExample(query="q", relevant_ids=frozenset({"a"})),
            ),
        )
        with self.assertRaises(EvaluationError):
            evaluate_retrieval(ds, self.provider, k=0)

    def test_evaluator_deterministic(self):
        """Same dataset, same data should produce same results."""
        from rag.evaluation.evaluator import evaluate_retrieval
        from rag.evaluation.dataset import EvaluationDataset, EvaluationExample

        ds = EvaluationDataset(
            name="determinism",
            examples=(
                EvaluationExample(
                    query="deterministic eval test",
                    relevant_ids=frozenset({"nonexistent-id"}),
                ),
            ),
        )

        r1 = evaluate_retrieval(ds, self.provider, k=5)
        r2 = evaluate_retrieval(ds, self.provider, k=5)

        self.assertEqual(r1.mean_recall, r2.mean_recall)
        self.assertEqual(r1.mean_precision, r2.mean_precision)
        self.assertEqual(r1.mean_hit_rate, r2.mean_hit_rate)
        self.assertEqual(r1.mean_mrr, r2.mean_mrr)
