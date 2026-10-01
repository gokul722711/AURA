"""Structured research result and evidence models for AURA M6."""

from dataclasses import dataclass, field
import re
from typing import Any

from agent.state import AgentState, AgentStatus
from rag.retrieval import RetrievalResult


def make_citation(document_title: str, chunk_id: str) -> str:
    """Generate canonical citation string for an evidence chunk."""
    title = document_title.strip() if document_title else "Untitled"
    cid = chunk_id.strip() if chunk_id else "N/A"
    return f"[{title}, Chunk: {cid}]"


def _is_chunk_cited(citation: str, chunk_id: str, text: str) -> bool:
    """Check if an evidence chunk is explicitly cited in target text.

    Accepts:
    1. Exact canonical citation: e.g. '[Title, Chunk: <chunk_id>]'
    2. Boundary-delimited chunk reference: e.g. 'Chunk: <chunk_id>' or 'Chunk <chunk_id>'
    Does NOT match on arbitrary substrings or bare document titles.
    """
    if not text:
        return False

    # 1. Exact canonical citation match: [Document Title, Chunk: <chunk_id>]
    if citation and citation in text:
        return True

    # 2. Clearly boundary-delimited chunk citation (e.g., 'Chunk: 1', 'Chunk: cid-123')
    if chunk_id:
        escaped_cid = re.escape(chunk_id.strip())
        pattern = rf"(?:\bChunk:\s*|\bChunk\s+|\bchunk_id[:=]\s*){escaped_cid}(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _copy_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep copy sources list, dictionaries, and inner chunk_ids lists."""
    copied: list[dict[str, Any]] = []
    for s in sources:
        if isinstance(s, dict):
            s_dict = dict(s)
            cids = s.get("chunk_ids")
            if isinstance(cids, list):
                s_dict["chunk_ids"] = list(cids)
            copied.append(s_dict)
        else:
            copied.append(s)
    return copied


@dataclass(frozen=True)
class ResearchEvidence:
    """A unit of grounded evidence retrieved during autonomous research.

    Attributes:
        chunk_id: Unique identifier of the retrieved chunk.
        document_title: Human-readable title of the source document.
        document_source: Source origin/URI of the document.
        content: Grounded text content retrieved from the knowledge base.
        score: Relevance/similarity score from retrieval (if available).
        document_id: Unique identifier of the source document (if available).
        citation: Canonical citation string tracing this evidence.
        metadata: Additional metadata from retrieval or chunking.
    """

    chunk_id: str
    document_title: str
    document_source: str
    content: str
    score: float | None = None
    document_id: str | None = None
    citation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.citation:
            object.__setattr__(
                self,
                "citation",
                make_citation(self.document_title, self.chunk_id),
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence to a JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_source": self.document_source,
            "content": self.content,
            "score": self.score,
            "citation": self.citation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchEvidence":
        """Instantiate ResearchEvidence from a dictionary."""
        chunk_id = str(data.get("chunk_id", "") or "")
        doc_title = str(data.get("document_title") or "Untitled")
        doc_source = str(data.get("document_source") or "unknown")
        content = str(data.get("content") or "")
        score = data.get("score")
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = None
        doc_id = data.get("document_id")
        citation = str(data.get("citation") or "")
        meta = data.get("metadata")
        return cls(
            chunk_id=chunk_id,
            document_title=doc_title,
            document_source=doc_source,
            content=content,
            score=score,
            document_id=str(doc_id) if doc_id else None,
            citation=citation,
            metadata=dict(meta) if isinstance(meta, dict) else {},
        )

    @classmethod
    def from_retrieval_result(cls, r: RetrievalResult) -> "ResearchEvidence":
        """Instantiate ResearchEvidence from an M2/M3 RetrievalResult."""
        return cls(
            chunk_id=str(r.chunk_id),
            document_id=str(r.document_id) if r.document_id else None,
            document_title=str(r.document_title or "Untitled"),
            document_source=str(r.document_source or "unknown"),
            content=r.content,
            score=float(r.score) if r.score is not None else None,
            metadata=dict(r.chunk_metadata) if r.chunk_metadata else {},
        )


def _aggregate_sources(evidence: list[ResearchEvidence]) -> list[dict[str, Any]]:
    """Group unique document sources and summarize chunk coverage.

    Includes document_id in grouping identity so distinct documents with
    identical title and source metadata remain separate entries.
    """
    sources_map: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    for ev in evidence:
        key = (ev.document_id, ev.document_title, ev.document_source)
        if key not in sources_map:
            sources_map[key] = {
                "document_title": ev.document_title,
                "document_source": ev.document_source,
                "document_id": ev.document_id,
                "chunk_count": 0,
                "chunk_ids": [],
            }
        sources_map[key]["chunk_count"] += 1
        sources_map[key]["chunk_ids"].append(ev.chunk_id)
    return list(sources_map.values())


@dataclass
class ResearchResult:
    """Structured, citation-grounded result of an autonomous research run.

    Attributes:
        objective: Original research objective.
        final_answer: Grounded synthesized response or explicit no-context statement.
        evidence: Deduplicated list of ResearchEvidence items accumulated across all searches.
        sources: Aggregated list of source documents with titles, sources, and chunk IDs.
        queries: List of all search queries executed during the research loop.
        iteration_count: Number of iterations / planning steps taken.
        has_evidence: True if at least one evidence chunk was retrieved; False if none.
        status: AgentStatus of the run (completed, failed, cancelled).
        duration_ms: Total execution duration in milliseconds.
        metadata: Additional run and execution metadata.
    """

    objective: str
    final_answer: str
    evidence: list[ResearchEvidence]
    sources: list[dict[str, Any]]
    queries: list[str]
    iteration_count: int
    has_evidence: bool
    status: AgentStatus = AgentStatus.COMPLETED
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citations(self) -> list[str]:
        """List of all valid citation strings from the accumulated evidence."""
        return [ev.citation for ev in self.evidence]

    @property
    def is_grounded(self) -> bool:
        """True if the result is grounded in retrieved evidence."""
        return self.has_evidence and len(self.evidence) > 0

    def verify_citations(self, text: str | None = None) -> dict[str, Any]:
        """Verify which valid citations from the evidence appear in the answer."""
        target_text = text if text is not None else self.final_answer
        valid_citations = [ev.citation for ev in self.evidence]
        matched_citations: list[str] = []
        matched_chunk_ids: list[str] = []

        if target_text:
            for ev in self.evidence:
                if _is_chunk_cited(ev.citation, ev.chunk_id, target_text):
                    matched_citations.append(ev.citation)
                    matched_chunk_ids.append(ev.chunk_id)

        return {
            "has_evidence": self.has_evidence,
            "total_evidence_count": len(self.evidence),
            "valid_citations": valid_citations,
            "matched_citations": matched_citations,
            "matched_chunk_ids": matched_chunk_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to clean dictionary representation.

        Deep copies sources, queries, and metadata to prevent mutable-state leaks.
        """
        status_val = self.status.value if isinstance(self.status, AgentStatus) else str(self.status)
        return {
            "objective": self.objective,
            "final_answer": self.final_answer,
            "evidence": [ev.to_dict() for ev in self.evidence],
            "sources": _copy_sources(self.sources),
            "queries": list(self.queries),
            "iteration_count": self.iteration_count,
            "has_evidence": self.has_evidence,
            "citations": self.citations,
            "is_grounded": self.is_grounded,
            "status": status_val,
            "duration_ms": self.duration_ms,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchResult":
        """Instantiate ResearchResult from a dictionary representation.

        Safely handles explicit None / null values on nullable fields.
        """
        raw_evidence = data.get("evidence")
        if isinstance(raw_evidence, list):
            evidence = [
                ResearchEvidence.from_dict(e) if isinstance(e, dict) else e
                for e in raw_evidence
                if e is not None
            ]
        else:
            evidence = []

        raw_sources = data.get("sources")
        if raw_sources is None:
            if "sources" in data:
                sources = []
            else:
                sources = _aggregate_sources(evidence)
        elif isinstance(raw_sources, list):
            sources = _copy_sources(raw_sources)
        else:
            sources = []

        raw_queries = data.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(q) for q in raw_queries if q is not None]
        else:
            queries = []

        raw_iter = data.get("iteration_count")
        if raw_iter is not None:
            try:
                iteration_count = int(raw_iter)
            except (ValueError, TypeError):
                iteration_count = 0
        else:
            iteration_count = 0

        raw_dur = data.get("duration_ms")
        if raw_dur is not None:
            try:
                duration_ms = float(raw_dur)
            except (ValueError, TypeError):
                duration_ms = 0.0
        else:
            duration_ms = 0.0

        raw_has_ev = data.get("has_evidence")
        if raw_has_ev is not None:
            has_evidence = bool(raw_has_ev)
        else:
            has_evidence = len(evidence) > 0

        status_raw = data.get("status")
        if status_raw is not None:
            try:
                status = AgentStatus(status_raw)
            except ValueError:
                status = AgentStatus.COMPLETED
        else:
            status = AgentStatus.COMPLETED

        raw_errors = data.get("errors")
        if isinstance(raw_errors, list):
            errors = [str(e) for e in raw_errors if e is not None]
        else:
            errors = []

        raw_meta = data.get("metadata")
        if isinstance(raw_meta, dict):
            metadata = dict(raw_meta)
        else:
            metadata = {}

        return cls(
            objective=str(data.get("objective") or ""),
            final_answer=str(data.get("final_answer") or ""),
            evidence=evidence,
            sources=sources,
            queries=queries,
            iteration_count=iteration_count,
            has_evidence=has_evidence,
            status=status,
            duration_ms=duration_ms,
            errors=errors,
            metadata=metadata,
        )

    @classmethod
    def from_state(cls, state: AgentState) -> "ResearchResult":
        """Construct a structured ResearchResult from completed AgentState.

        Safely handles null metadata in tool results.
        """
        evidence: list[ResearchEvidence] = []
        seen_chunk_ids: set[str] = set()
        queries: list[str] = []

        for res in getattr(state, "tool_results", []):
            if not isinstance(res, dict):
                continue
            if res.get("tool_name") == "rag_search":
                meta = res.get("metadata")
                if isinstance(meta, dict):
                    q = meta.get("query")
                    if q and isinstance(q, str):
                        queries.append(q)
                if not res.get("is_error"):
                    chunks = res.get("output")
                    if isinstance(chunks, list):
                        for c in chunks:
                            if isinstance(c, dict):
                                cid = c.get("chunk_id")
                                if cid and cid in seen_chunk_ids:
                                    continue
                                if cid:
                                    seen_chunk_ids.add(cid)
                                evidence.append(ResearchEvidence.from_dict(c))
                            elif isinstance(c, ResearchEvidence):
                                if c.chunk_id and c.chunk_id in seen_chunk_ids:
                                    continue
                                if c.chunk_id:
                                    seen_chunk_ids.add(c.chunk_id)
                                evidence.append(c)

        sources = _aggregate_sources(evidence)
        step_history = getattr(state, "step_history", [])
        duration_ms = sum(getattr(s, "duration_ms", 0.0) for s in step_history)
        status = getattr(state, "status", AgentStatus.COMPLETED)
        final_answer = getattr(state, "final_output", "") or ""
        raw_state_errors = getattr(state, "errors", [])
        errors = [str(e) for e in raw_state_errors if e is not None] if isinstance(raw_state_errors, list) else []
        raw_state_meta = getattr(state, "metadata", {})
        metadata = dict(raw_state_meta) if isinstance(raw_state_meta, dict) else {}

        return cls(
            objective=getattr(state, "objective", ""),
            final_answer=final_answer,
            evidence=evidence,
            sources=sources,
            queries=queries,
            iteration_count=getattr(state, "iteration", 0),
            has_evidence=len(evidence) > 0,
            status=status,
            duration_ms=duration_ms,
            errors=errors,
            metadata=metadata,
        )

    @classmethod
    def from_run_result(cls, run_result: Any) -> "ResearchResult":
        """Construct a structured ResearchResult from an AgentRunResult."""
        state = getattr(run_result, "state", run_result)
        return cls.from_state(state)

