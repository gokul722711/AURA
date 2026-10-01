"""Evaluator for AURA RAG retrieval quality.

Runs evaluation examples against an embedding provider and computes
deterministic retrieval metrics. Does not use an LLM.

The evaluator compares retrieved chunk identifiers against
the expected relevant identifiers defined in each EvaluationExample.
"""

from dataclasses import dataclass, field

from rag.embeddings.base import EmbeddingProvider
from rag.evaluation.dataset import EvaluationDataset, EvaluationExample
from rag.evaluation.metrics import (
    hit_rate_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)
from rag.exceptions import EvaluationError
from rag.retrieval import RetrievalConfig, retrieve_chunks


@dataclass(frozen=True)
class ExampleResult:
    """Result of evaluating a single example.

    Attributes:
        query: The evaluation query.
        retrieved_ids: Ordered list of retrieved chunk identifiers.
        relevant_ids: Set of expected relevant identifiers.
        recall: Recall@K score.
        precision: Precision@K score.
        hit_rate: Hit Rate@K score.
        reciprocal_rank: Reciprocal rank of the first relevant result.
        k: The K value used for top-K metrics.
    """

    query: str
    retrieved_ids: list[str]
    relevant_ids: frozenset[str]
    recall: float
    precision: float
    hit_rate: float
    reciprocal_rank: float
    k: int


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate result of evaluating a dataset.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        example_results: Individual results per example.
        mean_recall: Average Recall@K across all examples.
        mean_precision: Average Precision@K across all examples.
        mean_hit_rate: Average Hit Rate@K across all examples.
        mean_mrr: Mean Reciprocal Rank across all examples.
        k: The K value used.
        num_examples: Total number of examples evaluated.
    """

    dataset_name: str
    example_results: tuple[ExampleResult, ...]
    mean_recall: float
    mean_precision: float
    mean_hit_rate: float
    mean_mrr: float
    k: int
    num_examples: int


def evaluate_retrieval(
    dataset: EvaluationDataset,
    embedding_provider: EmbeddingProvider,
    k: int = 5,
    similarity_threshold: float = 0.0,
) -> EvaluationResult:
    """Evaluate retrieval quality over a dataset.

    For each example, runs retrieval using the embedding provider
    and computes Recall@K, Precision@K, Hit Rate@K, and MRR.

    This function does NOT call any LLM. It only measures retrieval quality.

    Args:
        dataset: The evaluation dataset to run.
        embedding_provider: Provider to embed queries.
        k: Number of top results to consider for metrics.
        similarity_threshold: Minimum similarity for retrieval.

    Returns:
        EvaluationResult with per-example and aggregate metrics.

    Raises:
        EvaluationError: If evaluation fails.
    """
    if k <= 0:
        raise EvaluationError("k must be greater than 0.")

    if len(dataset) == 0:
        raise EvaluationError("Evaluation dataset must not be empty.")

    config = RetrievalConfig(top_k=k, similarity_threshold=similarity_threshold)

    example_results = []
    for example in dataset:
        try:
            results = retrieve_chunks(
                query=example.query,
                embedding_provider=embedding_provider,
                config=config,
            )
        except Exception as exc:
            raise EvaluationError(
                f"Retrieval failed for query '{example.query}': {exc}"
            ) from exc

        retrieved_ids = [r.chunk_id for r in results]
        relevant_set = set(example.relevant_ids)

        ex_recall = recall_at_k(retrieved_ids, relevant_set, k)
        ex_precision = precision_at_k(retrieved_ids, relevant_set, k)
        ex_hit_rate = hit_rate_at_k(retrieved_ids, relevant_set, k)
        ex_mrr = mrr(retrieved_ids, relevant_set)

        example_results.append(
            ExampleResult(
                query=example.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=example.relevant_ids,
                recall=ex_recall,
                precision=ex_precision,
                hit_rate=ex_hit_rate,
                reciprocal_rank=ex_mrr,
                k=k,
            )
        )

    n = len(example_results)
    mean_recall = sum(r.recall for r in example_results) / n
    mean_precision = sum(r.precision for r in example_results) / n
    mean_hit_rate = sum(r.hit_rate for r in example_results) / n
    mean_mrr_val = sum(r.reciprocal_rank for r in example_results) / n

    return EvaluationResult(
        dataset_name=dataset.name,
        example_results=tuple(example_results),
        mean_recall=mean_recall,
        mean_precision=mean_precision,
        mean_hit_rate=mean_hit_rate,
        mean_mrr=mean_mrr_val,
        k=k,
        num_examples=n,
    )
