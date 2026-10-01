"""LLM-driven Autonomous Research Planner for M5/M6."""

import json
import uuid
from typing import Any

from agent.exceptions import InvalidPlanError
from agent.planning.base import ActionType, AgentStep, Plan, Planner
from agent.results import ResearchEvidence
from agent.state import AgentState
from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, Message, StructuredOutputRequest

RESEARCH_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["continue", "finish"],
            "description": "Decision whether to continue research by querying the knowledge base, or finish and synthesize an answer.",
        },
        "query": {
            "type": "string",
            "description": "Specific search query to execute if decision is continue.",
        },
    },
    "required": ["decision"],
}


class ResearchPlanner(Planner):
    """Iterative, LLM-driven planner for autonomous research.

    Evaluates accumulated evidence, autonomously decides whether to continue
    searching or finish, and produces grounded final synthesis with citation tracing.
    """

    supports_replanning: bool = True

    def __init__(
        self,
        gateway: ModelGateway,
        decision_temperature: float = 0.0,
        synthesis_temperature: float = 0.2,
        max_queries: int | None = None,
        decision_max_tokens: int = 256,
        synthesis_max_tokens: int = 1024,
    ) -> None:
        self.gateway = gateway
        self.decision_temperature = decision_temperature
        self.synthesis_temperature = synthesis_temperature
        self.max_queries = max_queries
        self.decision_max_tokens = decision_max_tokens
        self.synthesis_max_tokens = synthesis_max_tokens

    def plan(self, objective: str, state: AgentState) -> Plan:
        """Produce the next research step or grounded final synthesis."""
        evidence = self._extract_evidence(state)
        past_queries = self._get_past_queries(state)

        # If max query count reached, force finish and synthesize
        if self.max_queries is not None and len(past_queries) >= self.max_queries:
            decision_data = {"decision": "finish"}
        else:
            decision_data = self._get_model_decision(objective, past_queries, evidence)

        decision = decision_data.get("decision", "").lower().strip()

        if decision == "continue":
            query = decision_data.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                raise InvalidPlanError(
                    "Model decision 'continue' must include a non-empty string 'query'."
                )
            query = query.strip()
            step_num = len(state.step_history) + 1
            step = AgentStep(
                step_id=f"step-research-{step_num}",
                action_type=ActionType.TOOL,
                description=f"Search knowledge base for: {query}",
                payload={
                    "tool_name": "rag_search",
                    "tool_input": {"query": query},
                },
                metadata={"query": query, "step_num": step_num},
            )
            return Plan(
                plan_id=str(uuid.uuid4()),
                objective=objective,
                steps=(step,),
                metadata={
                    "planner": "ResearchPlanner",
                    "decision": "continue",
                    "query": query,
                },
            )

        elif decision == "finish":
            synthesis = self._synthesize_grounded_answer(objective, evidence, past_queries)
            step_num = len(state.step_history) + 1
            evidence_dicts = [
                ev.to_dict() if hasattr(ev, "to_dict") else ev for ev in evidence
            ]
            step = AgentStep(
                step_id=f"step-finish-{step_num}",
                action_type=ActionType.FINISH,
                description="Synthesize grounded final research response",
                payload={
                    "final_answer": synthesis,
                    "evidence": evidence_dicts,
                    "evidence_count": len(evidence),
                    "has_evidence": len(evidence) > 0,
                    "queries": past_queries,
                },
                metadata={
                    "evidence_count": len(evidence),
                    "has_evidence": len(evidence) > 0,
                },
            )
            return Plan(
                plan_id=str(uuid.uuid4()),
                objective=objective,
                steps=(step,),
                metadata={
                    "planner": "ResearchPlanner",
                    "decision": "finish",
                    "evidence_count": len(evidence),
                    "has_evidence": len(evidence) > 0,
                },
            )

        else:
            raise InvalidPlanError(
                f"Invalid research decision '{decision}'. Expected 'continue' or 'finish'."
            )

    def _extract_evidence(self, state: AgentState) -> list[ResearchEvidence]:
        """Accumulate unique chunks retrieved from all RAG searches in this run."""
        evidence: list[ResearchEvidence] = []
        seen_chunk_ids: set[str] = set()

        for res in state.tool_results:
            if res.get("tool_name") == "rag_search" and not res.get("is_error"):
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
        return evidence

    def _get_past_queries(self, state: AgentState) -> list[str]:
        """Extract all queries executed so far in this run."""
        queries: list[str] = []
        for res in state.tool_results:
            if res.get("tool_name") == "rag_search":
                q = res.get("metadata", {}).get("query")
                if q and isinstance(q, str):
                    queries.append(q)
        return queries

    def _get_model_decision(
        self,
        objective: str,
        past_queries: list[str],
        evidence: list[Any],
    ) -> dict[str, Any]:
        """Prompt LLM via ModelGateway.structured_output to decide next research step or finish."""
        system_prompt = (
            "You are AURA's Autonomous Research Agent. Your goal is to thoroughly "
            "investigate a research objective by querying the knowledge base.\n\n"
            "At each step, examine the objective and any already collected evidence.\n"
            "Produce a structured decision in one of two formats:\n\n"
            "1. If more information is required from the knowledge base:\n"
            '   decision="continue", query="<specific search query>"\n\n'
            "2. If sufficient evidence has been collected to synthesize a grounded answer, "
            "or if further searching will not yield new evidence:\n"
            '   decision="finish"\n\n'
            "Rules:\n"
            "- Formulate focused, specific search queries.\n"
            "- When decision is 'continue', query must be a non-empty string."
        )

        if not past_queries:
            user_content = (
                f"Research Objective:\n{objective}\n\n"
                f"No research queries have been executed yet.\n"
                f"What information should we search for first?"
            )
        else:
            queries_str = "\n".join(f"- {q}" for q in past_queries)
            if evidence:
                evidence_items = []
                for idx, ev in enumerate(evidence, start=1):
                    title = getattr(ev, "document_title", None) or (
                        ev.get("document_title") if isinstance(ev, dict) else "Untitled"
                    )
                    cid = getattr(ev, "chunk_id", None) or (
                        ev.get("chunk_id") if isinstance(ev, dict) else "N/A"
                    )
                    content = getattr(ev, "content", None) or (
                        ev.get("content") if isinstance(ev, dict) else ""
                    )
                    evidence_items.append(
                        f"[{idx}] Source: {title} (Chunk: {cid})\n{content.strip()}"
                    )
                evidence_str = "\n\n".join(evidence_items)
            else:
                evidence_str = "No relevant chunks were found by previous queries."

            user_content = (
                f"Research Objective:\n{objective}\n\n"
                f"Previous Queries Executed:\n{queries_str}\n\n"
                f"Collected Evidence:\n{evidence_str}\n\n"
                f"Decide whether to 'continue' with another search query or 'finish'."
            )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        req = StructuredOutputRequest(
            messages=messages,
            schema=RESEARCH_DECISION_SCHEMA,
            temperature=self.decision_temperature,
            max_tokens=self.decision_max_tokens,
        )

        try:
            resp = self.gateway.structured_output(req)
        except Exception as exc:
            raise InvalidPlanError(
                f"Structured decision generation failed: {exc}"
            ) from exc

        data = resp.data
        if not isinstance(data, dict):
            raise InvalidPlanError(
                f"Model decision must be a JSON object, got {type(data).__name__}."
            )

        return data

    def _synthesize_grounded_answer(
        self,
        objective: str,
        evidence: list[Any],
        past_queries: list[str],
    ) -> str:
        """Synthesize final grounded response using retrieved evidence and citations."""
        if not evidence:
            return (
                f"The available knowledge base did not provide sufficient supporting evidence "
                f"to answer the research objective: '{objective}'. "
                f"Queries executed: {past_queries}."
            )

        evidence_items = []
        for idx, ev in enumerate(evidence, start=1):
            title = getattr(ev, "document_title", None) or (
                ev.get("document_title") if isinstance(ev, dict) else "Untitled"
            )
            source = getattr(ev, "document_source", None) or (
                ev.get("document_source") if isinstance(ev, dict) else "Unknown"
            )
            cid = getattr(ev, "chunk_id", None) or (
                ev.get("chunk_id") if isinstance(ev, dict) else "N/A"
            )
            content = getattr(ev, "content", None) or (
                ev.get("content") if isinstance(ev, dict) else ""
            )
            evidence_items.append(
                f"[{idx}] Source: {title} | Document: {source} | Chunk: {cid}\n{content.strip()}"
            )
        evidence_block = "\n\n".join(evidence_items)

        system_prompt = (
            "You are AURA's Autonomous Research Agent. You synthesize comprehensive, "
            "factually grounded answers based strictly on retrieved evidence.\n\n"
            "Grounding Rules:\n"
            "1. Rely strictly on the provided evidence. Do NOT extrapolate, hallucinate, "
            "or assume facts not supported by the evidence.\n"
            "2. If the evidence is insufficient to answer parts of the objective, clearly "
            "indicate that the available knowledge base did not provide sufficient supporting evidence.\n"
            "3. Cite supporting sources using references like [Source Title, Chunk: ID] where appropriate. "
            "Never invent citations or reference sources not present in the provided evidence."
        )

        user_content = (
            f"Research Objective:\n{objective}\n\n"
            f"Retrieved Evidence:\n{evidence_block}\n\n"
            f"Provide a grounded, comprehensive answer addressing the research objective "
            f"based strictly on the evidence above."
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        req = GenerationRequest(
            messages=messages,
            temperature=self.synthesis_temperature,
            max_tokens=self.synthesis_max_tokens,
        )

        resp = self.gateway.generate(req)
        return resp.text.strip()
