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
        "document_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of document IDs to restrict search to.",
        },
    },
    "required": ["decision"],
}


class ResearchPlanner(Planner):
    """Iterative, LLM-driven planner for autonomous research with multiple research modes.

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
        mode: str = "knowledge_base",
        available_tools: set[str] | list[str] | None = None,
    ) -> None:
        self.gateway = gateway
        self.decision_temperature = decision_temperature
        self.synthesis_temperature = synthesis_temperature
        self.max_queries = max_queries
        self.decision_max_tokens = decision_max_tokens
        self.synthesis_max_tokens = synthesis_max_tokens
        self.mode = mode

        if available_tools is not None:
            self.available_tools = set(available_tools)
        elif mode == "model_knowledge":
            self.available_tools = set()
        elif mode == "web":
            self.available_tools = {"web_search"}
        elif mode == "web_knowledge_base":
            self.available_tools = {"rag_search", "web_search"}
        else:
            self.available_tools = {"rag_search"}

    def _synthesize_model_knowledge_answer(
        self, objective: str, remaining_seconds: float | None = None
    ) -> str:
        """Synthesize answer using only the model's pretrained/general knowledge."""
        messages = [
            Message(
                role="system",
                content=(
                    "You are AURA's Autonomous Research Agent operating in Model Knowledge mode. "
                    "Answer the user's research objective comprehensively and accurately using "
                    "your general pretrained knowledge. Do not cite external documents or chunks."
                ),
            ),
            Message(role="user", content=f"Research Objective:\n{objective}"),
        ]
        metadata = {}
        if remaining_seconds is not None:
            metadata["timeout"] = remaining_seconds
        resp = self.gateway.generate(
            GenerationRequest(
                messages=messages,
                temperature=self.synthesis_temperature,
                max_tokens=self.synthesis_max_tokens,
                metadata=metadata,
            )
        )
        return resp.text.strip()

    def plan(
        self,
        objective: str,
        state: AgentState,
        tracker: Any = None,
    ) -> Plan:
        """Produce the next research step or grounded final synthesis."""
        remaining_budget: float | None = None
        if tracker is not None and hasattr(tracker, "remaining_seconds"):
            remaining_budget = tracker.remaining_seconds()


        # 1. Model Knowledge mode: direct synthesis without tool querying
        if self.mode == "model_knowledge" or not self.available_tools:
            synthesis = self._synthesize_model_knowledge_answer(
                objective, remaining_seconds=remaining_budget
            )
            step = AgentStep(
                step_id="step-model-knowledge-1",
                action_type=ActionType.FINISH,
                description="Synthesize answer using model pretrained knowledge",
                payload={
                    "final_answer": synthesis,
                    "evidence": [],
                    "evidence_count": 0,
                    "has_evidence": False,
                    "queries": [],
                },
                metadata={"mode": "model_knowledge"},
            )
            return Plan(
                plan_id=str(uuid.uuid4()),
                objective=objective,
                steps=(step,),
                metadata={
                    "planner": "ResearchPlanner",
                    "mode": "model_knowledge",
                    "decision": "finish",
                    "has_evidence": False,
                },
            )

        # 2. Tool-based research modes (Knowledge Base, Web, Web + Knowledge Base)
        evidence = self._extract_evidence(state)
        past_queries = self._get_past_queries(state)

        # If max query count reached, force finish and synthesize
        if self.max_queries is not None and len(past_queries) >= self.max_queries:
            decision_data = {"decision": "finish"}
        else:
            decision_data = self._get_model_decision(
                objective, past_queries, evidence, remaining_seconds=remaining_budget
            )

        decision = decision_data.get("decision", "").lower().strip()

        if decision == "continue":
            query = decision_data.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                raise InvalidPlanError(
                    "Model decision 'continue' must include a non-empty string 'query'."
                )
            query = query.strip()

            # Select tool: if multiple tools are available, check model decision; else use the single available tool
            if len(self.available_tools) > 1:
                selected_tool = decision_data.get("tool")
                if not selected_tool or selected_tool not in self.available_tools:
                    selected_tool = "rag_search" if "rag_search" in self.available_tools else next(iter(self.available_tools))
            else:
                selected_tool = next(iter(self.available_tools))

            tool_input: dict[str, Any] = {"query": query}
            if "document_ids" in decision_data and decision_data["document_ids"]:
                tool_input["document_ids"] = decision_data["document_ids"]

            step_num = len(state.step_history) + 1
            step = AgentStep(
                step_id=f"step-research-{step_num}",
                action_type=ActionType.TOOL,
                description=f"Search {selected_tool} for: {query}",
                payload={
                    "tool_name": selected_tool,
                    "tool_input": tool_input,
                },
                metadata={"query": query, "step_num": step_num, "tool": selected_tool},
            )
            return Plan(
                plan_id=str(uuid.uuid4()),
                objective=objective,
                steps=(step,),
                metadata={
                    "planner": "ResearchPlanner",
                    "decision": "continue",
                    "query": query,
                    "tool": selected_tool,
                },
            )

        elif decision == "finish":
            synthesis = self._synthesize_grounded_answer(
                objective, evidence, past_queries, remaining_seconds=remaining_budget
            )
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
        """Accumulate unique chunks retrieved from all RAG or Web searches in this run."""
        evidence: list[ResearchEvidence] = []
        seen_chunk_ids: set[str] = set()

        for res in state.tool_results:
            if res.get("tool_name") in ("rag_search", "web_search") and not res.get("is_error"):
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
        """Extract all queries executed so far in this run across all search tools."""
        queries: list[str] = []
        for res in state.tool_results:
            if res.get("tool_name") in ("rag_search", "web_search"):
                q = res.get("metadata", {}).get("query")
                if q and isinstance(q, str):
                    queries.append(q)
        return queries

    def _get_model_decision(
        self,
        objective: str,
        past_queries: list[str],
        evidence: list[Any],
        remaining_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Prompt LLM via ModelGateway.structured_output to decide next research step or finish."""
        tool_descriptions = []
        if "rag_search" in self.available_tools:
            tool_descriptions.append("- 'rag_search': Search internal indexed Knowledge Base for project documentation and uploaded records.")
        if "web_search" in self.available_tools:
            tool_descriptions.append("- 'web_search': Search live web for public documentation, external articles, and web resources.")
        tools_list_text = "\n".join(tool_descriptions)

        multi_tool = len(self.available_tools) > 1

        if multi_tool:
            decision_schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["continue", "finish"],
                        "description": "Decision whether to continue research by querying an information source, or finish and synthesize an answer.",
                    },
                    "tool": {
                        "type": "string",
                        "enum": sorted(list(self.available_tools)),
                        "description": "Source tool to search: 'rag_search' for indexed knowledge base, or 'web_search' for live web search.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific search query to execute if decision is continue.",
                    },
                },
                "required": ["decision"],
            }
            system_prompt = (
                f"You are AURA's Autonomous Research Agent. Your goal is to thoroughly "
                f"investigate a research objective using the permitted information sources:\n"
                f"{tools_list_text}\n\n"
                f"At each step, examine the objective and any already collected evidence.\n"
                f"Produce a structured decision:\n\n"
                f"1. If more information is required:\n"
                f'   decision="continue", tool="<rag_search|web_search>", query="<specific search query>"\n\n'
                f"2. If sufficient evidence has been collected to synthesize a grounded answer, "
                f"or if further searching will not yield new evidence:\n"
                f'   decision="finish"\n\n'
                f"Rules:\n"
                f"- Formulate focused, specific search queries.\n"
                f"- Formulate new search queries; do not repeat queries that have already been executed.\n"
                f"- Choose the most appropriate tool based on the objective and evidence.\n"
                f"- When decision is 'continue', query must be a non-empty string."
            )
        else:
            decision_schema = RESEARCH_DECISION_SCHEMA
            single_tool = next(iter(self.available_tools)) if self.available_tools else "rag_search"
            system_prompt = (
                f"You are AURA's Autonomous Research Agent. Your goal is to thoroughly "
                f"investigate a research objective by querying: {single_tool}.\n\n"
                f"At each step, examine the objective and any already collected evidence.\n"
                f"Produce a structured decision in one of two formats:\n\n"
                f"1. If more information is required:\n"
                f'   decision="continue", query="<specific search query>"\n\n'
                f"2. If sufficient evidence has been collected to synthesize a grounded answer, "
                f"or if further searching will not yield new evidence:\n"
                f'   decision="finish"\n\n'
                f"Rules:\n"
                f"- Formulate focused, specific search queries.\n"
                f"- Formulate new search queries; do not repeat queries that have already been executed.\n"
                f"- When decision is 'continue', query must be a non-empty string."
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
                    page = (
                        getattr(ev, "page", None)
                        or (ev.get("page") if isinstance(ev, dict) else None)
                        or (
                            ev.get("metadata", {}).get("page")
                            if isinstance(ev, dict) and isinstance(ev.get("metadata"), dict)
                            else None
                        )
                    )
                    url = (
                        getattr(ev, "url", None)
                        or (ev.get("url") if isinstance(ev, dict) else None)
                        or (
                            ev.get("metadata", {}).get("url")
                            if isinstance(ev, dict) and isinstance(ev.get("metadata"), dict)
                            else None
                        )
                        or (
                            ev.get("metadata", {}).get("canonical_url")
                            if isinstance(ev, dict) and isinstance(ev.get("metadata"), dict)
                            else None
                        )
                    )
                    if not url and isinstance(ev, dict):
                        ds = ev.get("document_source", "")
                        if ds and str(ds).startswith(("http://", "https://")):
                            url = str(ds)

                    page_str = f" (Page: {page})" if page is not None else ""
                    url_str = f" (URL: {url})" if url else ""
                    evidence_items.append(
                        f"[{idx}] Source: {title}{page_str}{url_str} (Chunk: {cid})\n{content.strip()}"
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

        metadata = {}
        if remaining_seconds is not None:
            metadata["timeout"] = remaining_seconds

        req = StructuredOutputRequest(
            messages=messages,
            schema=decision_schema,
            temperature=self.decision_temperature,
            max_tokens=self.decision_max_tokens,
            metadata=metadata,
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
        remaining_seconds: float | None = None,
    ) -> str:
        """Synthesize final grounded response using retrieved evidence and citations."""
        if not evidence:
            return (
                f"The available information sources did not provide sufficient supporting evidence "
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
            page = (
                getattr(ev, "page", None)
                or (ev.get("page") if isinstance(ev, dict) else None)
                or (
                    ev.get("metadata", {}).get("page")
                    if isinstance(ev, dict) and isinstance(ev.get("metadata"), dict)
                    else None
                )
            )
            page_str = f" | Page: {page}" if page is not None else ""
            evidence_items.append(
                f"[{idx}] Source: {title} | Document: {source}{page_str} | Chunk: {cid}\n{content.strip()}"
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

        metadata = {}
        if remaining_seconds is not None:
            metadata["timeout"] = remaining_seconds

        req = GenerationRequest(
            messages=messages,
            temperature=self.synthesis_temperature,
            max_tokens=self.synthesis_max_tokens,
            metadata=metadata,
        )

        resp = self.gateway.generate(req)
        return resp.text.strip()
