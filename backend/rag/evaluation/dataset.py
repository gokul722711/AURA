"""Evaluation dataset representation for AURA RAG evaluation.

Provides a minimal, deterministic data structure for representing
evaluation examples (query + expected relevant identifiers).

This does NOT depend on an LLM. The dataset is used by the evaluator
to compute retrieval metrics.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationExample:
    """A single evaluation example.

    Attributes:
        query: The evaluation query text.
        relevant_ids: Set of chunk or document identifiers that are
            considered relevant for this query.
        description: Optional human-readable description of the example.
    """

    query: str
    relevant_ids: frozenset[str]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.relevant_ids, frozenset):
            object.__setattr__(self, "relevant_ids", frozenset(self.relevant_ids))


@dataclass(frozen=True)
class EvaluationDataset:
    """A collection of evaluation examples.

    Attributes:
        name: Human-readable name of the dataset.
        examples: List of EvaluationExample instances.
        description: Optional description of the dataset.
    """

    name: str
    examples: tuple[EvaluationExample, ...] = field(default_factory=tuple)
    description: str = ""

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)
