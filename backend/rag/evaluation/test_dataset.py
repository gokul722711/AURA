"""Repository-local deterministic evaluation dataset for AURA RAG.

This dataset is intentionally small and designed for testing the
evaluation framework mechanics, NOT for measuring semantic retrieval
quality. The MockEmbeddingProvider produces hash-based vectors that
are deterministic but do not have meaningful semantic properties.

The expected relevant identifiers in this dataset are synthetic
and used to verify that the evaluation pipeline (metrics, evaluator,
dataset representation) functions correctly.

Do not use this dataset to claim meaningful retrieval quality.
"""

from rag.evaluation.dataset import EvaluationDataset, EvaluationExample


def get_test_evaluation_dataset() -> EvaluationDataset:
    """Return a small deterministic evaluation dataset for framework testing.

    These examples use synthetic relevant IDs that must be mapped
    to actual chunk IDs in a test fixture. The purpose is to test
    evaluation pipeline mechanics, not semantic search quality.

    Returns:
        An EvaluationDataset with representative test examples.
    """
    return EvaluationDataset(
        name="test-evaluation-v1",
        description=(
            "Small synthetic dataset for testing the evaluation framework. "
            "Uses deterministic hash-based mock embeddings. "
            "Does NOT measure meaningful semantic retrieval quality."
        ),
        examples=(
            EvaluationExample(
                query="What is Python?",
                relevant_ids=frozenset({"chunk-python-0", "chunk-python-1"}),
                description="Basic Python query with two relevant chunks.",
            ),
            EvaluationExample(
                query="How does machine learning work?",
                relevant_ids=frozenset({"chunk-ml-0"}),
                description="ML query with one relevant chunk.",
            ),
            EvaluationExample(
                query="What is Django used for?",
                relevant_ids=frozenset({"chunk-django-0", "chunk-django-1"}),
                description="Django query with two relevant chunks.",
            ),
            EvaluationExample(
                query="Explain vector databases",
                relevant_ids=frozenset({"chunk-vector-0"}),
                description="Vector DB query with one relevant chunk.",
            ),
            EvaluationExample(
                query="What is retrieval-augmented generation?",
                relevant_ids=frozenset({"chunk-rag-0", "chunk-rag-1", "chunk-rag-2"}),
                description="RAG query with three relevant chunks.",
            ),
        ),
    )
