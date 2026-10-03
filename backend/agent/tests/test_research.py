"""Deterministic offline unit and integration tests for M5 Autonomous Research."""

import json
from collections.abc import Iterator
from typing import Any

from django.test import TestCase

from agent.execution.limits import ExecutionLimits
from agent.planning.research import RESEARCH_DECISION_SCHEMA, ResearchPlanner
from agent.research import (
    ResearchEvidence,
    ResearchResult,
    ResearchRuntime,
    create_research_runtime,
)
from agent.results import _aggregate_sources
from agent.state import AgentState, AgentStatus
from rag.models import Document
from agent.tools.builtin.rag import RAGSearchTool
from agent.tools.policy import DefaultToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.base import LLMProvider
from gateway.exceptions import GenerationError
from gateway.gateway import ModelGateway
from gateway.types import (
    ALL_CAPABILITIES,
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalConfig


class ScriptedLLMProvider(LLMProvider):
    """Deterministic LLM Provider returning a scripted sequence of responses."""

    def __init__(
        self,
        responses: list[str] | None = None,
        fail_on_call: int | None = None,
        failure_exc: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.call_count = 0
        self.recorded_requests: list[GenerationRequest] = []
        self.recorded_structured_requests: list[StructuredOutputRequest] = []
        self.fail_on_call = fail_on_call
        self.failure_exc = failure_exc or GenerationError("Simulated provider failure")

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider="scripted",
            model="scripted-model",
            capabilities=ALL_CAPABILITIES,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.recorded_requests.append(request)
        self.call_count += 1

        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise self.failure_exc

        if self.call_count <= len(self.responses):
            text = self.responses[self.call_count - 1]
        else:
            text = '{"decision": "finish"}'

        return GenerationResponse(
            text=text,
            provider="scripted",
            model="scripted-model",
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        resp = self.generate(request)
        yield StreamChunk(text=resp.text, index=0, finish_reason="stop")

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        self.recorded_structured_requests.append(request)
        gen_req = GenerationRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        resp = self.generate(gen_req)
        try:
            data = json.loads(resp.text)
        except Exception as exc:
            raise GenerationError(
                f"Model returned invalid JSON decision: {exc}. Raw output: {resp.text}"
            ) from exc
        return StructuredOutputResponse(
            data=data,
            raw_text=resp.text,
            provider="scripted",
            model="scripted-model",
        )


class AutonomousResearchTests(TestCase):
    """Test suite verifying M5 Autonomous Research capabilities."""

    def setUp(self) -> None:
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        self.doc = ingest_document(
            title="Quantum Architecture Report",
            content=(
                "Quantum processors rely on qubits to perform computations in superposition. "
                "Entanglement enables correlated states between distant qubits. "
                "Current limitations include decoherence and thermal noise."
            ),
            embedding_provider=self.embedding_provider,
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
            source="quantum_report.pdf",
        )

    def _build_runtime(
        self,
        provider: LLMProvider,
        limits: ExecutionLimits | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ):
        gateway = ModelGateway(provider=provider)
        return create_research_runtime(
            gateway=gateway,
            embedding_provider=self.embedding_provider,
            limits=limits or ExecutionLimits(max_iterations=10, max_tool_calls=10, max_time_seconds=30.0),
            retrieval_config=retrieval_config or RetrievalConfig(top_k=3),
        )

    def test_one_research_query_evidence_finish(self) -> None:
        """Requirement 10.1: One research query -> evidence -> finish."""
        responses = [
            '{"decision": "continue", "query": "qubits superposition"}',
            '{"decision": "finish"}',
            "Grounded Synthesis: Qubits operate in superposition. [Source: Quantum Architecture Report, Chunk: chunk-1]",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Investigate how qubits operate in quantum computing.")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("Qubits operate in superposition", result.final_output)
        self.assertEqual(len(result.state.tool_results), 1)
        self.assertEqual(result.state.tool_results[0]["tool_name"], "rag_search")
        self.assertGreater(len(result.state.tool_results[0]["output"]), 0)

    def test_two_queries_insufficient_evidence_then_finish(self) -> None:
        """Requirement 10.2: Research -> insufficient evidence -> second query -> finish."""
        responses = [
            '{"decision": "continue", "query": "quantum processors superposition"}',
            '{"decision": "continue", "query": "decoherence thermal noise limitations"}',
            '{"decision": "finish"}',
            "Grounded Synthesis: Quantum processors use qubits in superposition, but suffer from decoherence.",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Investigate quantum processors and their key limitations.")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(len(result.state.tool_results), 2)
        self.assertEqual(result.state.tool_results[0]["metadata"]["query"], "quantum processors superposition")
        self.assertEqual(result.state.tool_results[1]["metadata"]["query"], "decoherence thermal noise limitations")
        self.assertIn("decoherence", result.final_output)

    def test_no_useful_evidence(self) -> None:
        """Requirement 10.3: No useful evidence in knowledge base."""
        responses = [
            '{"decision": "continue", "query": "astronomy telescopes mars orbit"}',
            '{"decision": "finish"}',
        ]
        provider = ScriptedLLMProvider(responses=responses)
        # Use high similarity threshold so no chunks match
        runtime = self._build_runtime(
            provider,
            retrieval_config=RetrievalConfig(top_k=3, similarity_threshold=0.999),
        )

        result = runtime.run("What are the orbital coordinates of Mars?")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("did not provide sufficient supporting evidence", result.final_output)

    def test_maximum_research_iterations(self) -> None:
        """Requirement 10.4: Maximum research iterations limit reached."""
        # Provider perpetually wants to continue
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "query 1"}',
                '{"decision": "continue", "query": "query 2"}',
                '{"decision": "continue", "query": "query 3"}',
                '{"decision": "continue", "query": "query 4"}',
            ]
        )
        limits = ExecutionLimits(max_iterations=2, max_tool_calls=10, max_time_seconds=30.0)
        runtime = self._build_runtime(provider, limits=limits)

        result = runtime.run("Unbounded research topic")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("Iteration limit" in err for err in result.state.errors))
        failed_events = result.trace.get_events_by_type("AgentRunFailed")
        self.assertEqual(len(failed_events), 1)

    def test_invalid_structured_model_decision(self) -> None:
        """Requirement 10.5: Invalid structured model decision handled safely."""
        responses = [
            '{"decision": "unknown_action_xyz"}',
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Test invalid decision")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("Invalid research decision" in err for err in result.state.errors))

    def test_invalid_json_decision(self) -> None:
        """Requirement 10.5b: Non-JSON model decision handled safely."""
        responses = [
            "I recommend searching for quantum computers.",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Test non-JSON decision")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("invalid JSON" in err for err in result.state.errors))

    def test_model_provider_failure(self) -> None:
        """Requirement 10.6: Model/provider failure during execution."""
        provider = ScriptedLLMProvider(
            responses=['{"decision": "continue", "query": "qubits"}'],
            fail_on_call=1,
            failure_exc=GenerationError("NVIDIA API 503 Service Unavailable"),
        )
        runtime = self._build_runtime(provider)

        result = runtime.run("Test provider failure")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("503 Service Unavailable" in err for err in result.state.errors))

    def test_final_synthesis_uses_collected_evidence(self) -> None:
        """Requirement 10.7: Final synthesis receives and uses collected evidence."""
        responses = [
            '{"decision": "continue", "query": "entanglement correlated states"}',
            '{"decision": "finish"}',
            "Grounded: Entanglement creates correlated states between distant qubits.",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Explain entanglement.")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)

        # Verify evidence was injected into the synthesis request
        synthesis_req = provider.recorded_requests[-1]
        user_msg = synthesis_req.messages[-1].content
        self.assertIn("Retrieved Evidence:", user_msg)
        self.assertIn("Quantum Architecture Report", user_msg)
        self.assertIn("correlated states", user_msg)

    def test_cancellation_works(self) -> None:
        """Requirement 10.8: Pre-run and mid-run cancellation works."""
        provider = ScriptedLLMProvider(
            responses=['{"decision": "continue", "query": "qubits"}']
        )
        runtime = self._build_runtime(provider)
        runtime.cancel("Research cancelled before start.")

        result = runtime.run("Test cancellation")

        self.assertEqual(result.status, AgentStatus.CANCELLED)
        self.assertFalse(result.is_success)
        self.assertEqual(len(result.trace.get_events_by_type("AgentRunCancelled")), 1)

    def test_execution_limits_tool_calls(self) -> None:
        """Requirement 10.9: Tool call limit works safely."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "query 1"}',
                '{"decision": "continue", "query": "query 2"}',
            ]
        )
        limits = ExecutionLimits(max_iterations=10, max_tool_calls=1, max_time_seconds=30.0)
        runtime = self._build_runtime(provider, limits=limits)

        result = runtime.run("Test tool call limit")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("Tool call limit" in err for err in result.state.errors))

    def test_execution_limits_timeout(self) -> None:
        """Requirement 10.9b: Timeout limit works safely."""
        provider = ScriptedLLMProvider(
            responses=['{"decision": "continue", "query": "query 1"}']
        )
        limits = ExecutionLimits(max_iterations=10, max_tool_calls=10, max_time_seconds=0.0001)
        runtime = self._build_runtime(provider, limits=limits)

        result = runtime.run("Test timeout limit")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("Time limit" in err for err in result.state.errors))

    def test_research_requests_receive_expected_max_tokens(self) -> None:
        """Regression test: verify GenerationRequest receives expected max_tokens for decision and synthesis."""
        responses = [
            '{"decision": "continue", "query": "qubits superposition"}',
            '{"decision": "finish"}',
            "Grounded Synthesis: Qubits operate in superposition.",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        runtime = self._build_runtime(provider)

        result = runtime.run("Investigate how qubits operate in quantum computing.")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        # We had 3 LLM calls:
        # 1: _get_model_decision (initial step) -> max_tokens == 256
        # 2: _get_model_decision (replanning after evidence) -> max_tokens == 256
        # 3: _synthesize_grounded_answer -> max_tokens == 1024
        self.assertEqual(len(provider.recorded_requests), 3)

        decision_req_1 = provider.recorded_requests[0]
        self.assertEqual(decision_req_1.max_tokens, 256)

        decision_req_2 = provider.recorded_requests[1]
        self.assertEqual(decision_req_2.max_tokens, 256)

        synthesis_req = provider.recorded_requests[2]
        self.assertEqual(synthesis_req.max_tokens, 1024)

    def test_research_planner_configurable_max_tokens(self) -> None:
        """Verify ResearchPlanner honors custom decision_max_tokens and synthesis_max_tokens."""
        responses = [
            '{"decision": "continue", "query": "qubits superposition"}',
            '{"decision": "finish"}',
            "Custom max tokens grounded answer.",
        ]
        provider = ScriptedLLMProvider(responses=responses)
        gateway = ModelGateway(provider=provider)
        planner = ResearchPlanner(
            gateway=gateway,
            decision_max_tokens=128,
            synthesis_max_tokens=512,
        )
        registry = ToolRegistry()
        rag_tool = RAGSearchTool(
            embedding_provider=self.embedding_provider,
            config=RetrievalConfig(top_k=3),
        )
        registry.register(rag_tool)
        policy = DefaultToolPolicy(allowed_tools={"rag_search"})
        from agent.runtime import AgentRuntime
        runtime = AgentRuntime(
            planner=planner,
            gateway=gateway,
            registry=registry,
            policy=policy,
        )

        result = runtime.run("Custom max tokens test")
        self.assertTrue(result.is_success)

        self.assertEqual(len(provider.recorded_requests), 3)
        self.assertEqual(provider.recorded_requests[0].max_tokens, 128)
        self.assertEqual(provider.recorded_requests[1].max_tokens, 128)
        self.assertEqual(provider.recorded_requests[2].max_tokens, 512)

    def test_valid_continue_structured_decision(self) -> None:
        """Requirement: valid CONTINUE structured decision produces rag_search tool step."""
        provider = ScriptedLLMProvider(
            responses=['{"decision": "continue", "query": "qubits superposition"}']
        )
        gateway = ModelGateway(provider=provider)
        planner = ResearchPlanner(gateway=gateway)
        from agent.state import AgentState
        state = AgentState.create("Investigate quantum computing")

        plan = planner.plan(state.objective, state)

        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        from agent.planning.base import ActionType
        self.assertEqual(step.action_type, ActionType.TOOL)
        self.assertEqual(step.payload.get("tool_name"), "rag_search")
        self.assertEqual(step.payload.get("tool_input"), {"query": "qubits superposition"})
        self.assertEqual(len(provider.recorded_structured_requests), 1)
        req = provider.recorded_structured_requests[0]
        self.assertEqual(req.schema, RESEARCH_DECISION_SCHEMA)
        self.assertEqual(req.max_tokens, 256)

    def test_valid_finish_structured_decision(self) -> None:
        """Requirement: valid FINISH structured decision produces finish step with grounded synthesis."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "finish"}',
                "Grounded Synthesis: Qubits operate in superposition.",
            ]
        )
        gateway = ModelGateway(provider=provider)
        planner = ResearchPlanner(gateway=gateway)
        from agent.state import AgentState
        state = AgentState.create("Investigate quantum computing")
        state.record_tool_result({
            "tool_name": "rag_search",
            "output": [
                {
                    "chunk_id": "chunk-1",
                    "document_title": "Quantum Architecture Report",
                    "document_source": "quantum_report.pdf",
                    "content": "Qubits operate in superposition.",
                }
            ],
            "metadata": {"query": "qubits superposition"},
        })

        plan = planner.plan(state.objective, state)

        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        from agent.planning.base import ActionType
        self.assertEqual(step.action_type, ActionType.FINISH)
        self.assertIn("Qubits operate in superposition", step.payload.get("final_answer", ""))
        self.assertEqual(len(provider.recorded_structured_requests), 1)
        self.assertEqual(provider.recorded_structured_requests[0].schema, RESEARCH_DECISION_SCHEMA)
        self.assertEqual(provider.recorded_structured_requests[0].max_tokens, 256)

    def test_structured_decision_with_accumulated_evidence(self) -> None:
        """Requirement: structured decision receives accumulated evidence in prompt."""
        provider = ScriptedLLMProvider(
            responses=['{"decision": "continue", "query": "entanglement"}']
        )
        gateway = ModelGateway(provider=provider)
        planner = ResearchPlanner(gateway=gateway)
        from agent.state import AgentState
        state = AgentState.create("Quantum analysis")
        state.record_tool_result({
            "tool_name": "rag_search",
            "output": [
                {
                    "chunk_id": "c-42",
                    "document_title": "Entanglement Overview",
                    "document_source": "doc.pdf",
                    "content": "Entanglement correlates quantum states.",
                }
            ],
            "metadata": {"query": "initial query"},
        })

        plan = planner.plan(state.objective, state)

        self.assertEqual(len(provider.recorded_structured_requests), 1)
        struct_req = provider.recorded_structured_requests[0]
        user_content = struct_req.messages[-1].content
        self.assertIn("Entanglement Overview", user_content)
        self.assertIn("c-42", user_content)
        self.assertIn("Entanglement correlates quantum states.", user_content)
        self.assertIn("initial query", user_content)

    def test_malformed_structured_response_handled_as_controlled_planner_failure(self) -> None:
        """Requirement: malformed structured output is caught and handled safely as controlled failure."""
        class MalformedStructuredProvider(LLMProvider):
            def metadata(self) -> ProviderMetadata:
                return ProviderMetadata(provider="malformed", model="m", capabilities=ALL_CAPABILITIES)

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return GenerationResponse(text="", provider="malformed", model="m")

            def stream(self, request: GenerationRequest):
                yield StreamChunk(text="")

            def structured_output(self, request: StructuredOutputRequest) -> StructuredOutputResponse:
                return StructuredOutputResponse(
                    data="invalid non-dict data",
                    raw_text="invalid non-dict data",
                    provider="malformed",
                    model="m",
                )

        gateway = ModelGateway(provider=MalformedStructuredProvider())
        runtime = create_research_runtime(
            gateway=gateway,
            embedding_provider=self.embedding_provider,
        )

        result = runtime.run("Test malformed response")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertTrue(any("must be a JSON object" in err for err in result.state.errors))


class ResearchEvidenceAndResultTests(TestCase):
    """Focused offline unit and regression tests for M6 Research Result & Evidence Quality."""

    def setUp(self) -> None:
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        Document.objects.all().delete()
        self.doc_a = ingest_document(
            title="AURA Agent Architecture",
            content="Agent Runtime coordinates stateful execution and replanning cycles.",
            embedding_provider=self.embedding_provider,
            source="aura://docs/agent-arch",
        )
        self.doc_b = ingest_document(
            title="AURA RAG Vector Store",
            content="Vector store utilizes PostgreSQL pgvector for deterministic cosine similarity retrieval.",
            embedding_provider=self.embedding_provider,
            source="aura://docs/vector-store",
        )

    def _build_runtime(
        self,
        provider: LLMProvider,
        limits: ExecutionLimits | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> ResearchRuntime:
        gateway = ModelGateway(provider=provider)
        return create_research_runtime(
            gateway=gateway,
            embedding_provider=self.embedding_provider,
            limits=limits or ExecutionLimits(max_iterations=10, max_tool_calls=10, max_time_seconds=30.0),
            retrieval_config=retrieval_config or RetrievalConfig(top_k=3),
        )

    def test_empty_evidence(self) -> None:
        """Requirement: empty evidence produces a structured no-evidence result."""
        Document.objects.all().delete()
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "nonexistent"}',
                '{"decision": "finish"}',
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Find missing documentation")

        self.assertFalse(res.has_evidence)
        self.assertFalse(res.is_grounded)
        self.assertEqual(res.evidence, [])
        self.assertEqual(res.sources, [])
        self.assertEqual(res.citations, [])
        self.assertIn("did not provide sufficient supporting evidence", res.final_answer)

    def test_single_evidence_item(self) -> None:
        """Requirement: single evidence item preserves title, chunk, citation, and grounding."""
        Document.objects.all().delete()
        doc = ingest_document(
            title="Quantum Superposition",
            content="Qubits remain in superposition until measured.",
            embedding_provider=self.embedding_provider,
            source="quantum.pdf",
        )
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "superposition"}',
                '{"decision": "finish"}',
                "Grounded answer citing [Quantum Superposition, Chunk: test-chunk].",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("What is superposition?")

        self.assertTrue(res.has_evidence)
        self.assertTrue(res.is_grounded)
        self.assertEqual(len(res.evidence), 1)
        ev = res.evidence[0]
        self.assertEqual(ev.document_title, "Quantum Superposition")
        self.assertEqual(ev.document_source, "quantum.pdf")
        self.assertEqual(ev.content, "Qubits remain in superposition until measured.")
        self.assertEqual(ev.citation, f"[Quantum Superposition, Chunk: {ev.chunk_id}]")
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["document_title"], "Quantum Superposition")
        self.assertEqual(res.sources[0]["chunk_count"], 1)

    def test_multiple_evidence_items(self) -> None:
        """Requirement: multiple evidence items are accumulated and grouped by source."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "architecture vector store"}',
                '{"decision": "finish"}',
                "Comprehensive answer synthesizing both agent architecture and vector store.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Explain AURA architecture and storage")

        self.assertTrue(res.has_evidence)
        self.assertEqual(len(res.evidence), 2)
        self.assertEqual(len(res.sources), 2)
        source_titles = {s["document_title"] for s in res.sources}
        self.assertIn("AURA Agent Architecture", source_titles)
        self.assertIn("AURA RAG Vector Store", source_titles)

    def test_duplicate_chunk_ids_deduplicated(self) -> None:
        """Requirement: duplicate chunk IDs across searches are deduplicated while preserving order."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "agent architecture"}',
                '{"decision": "continue", "query": "agent architecture again"}',
                '{"decision": "finish"}',
                "Final synthesis after duplicate queries.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("How does agent architecture work?")

        chunk_ids = [e.chunk_id for e in res.evidence]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertGreaterEqual(len(chunk_ids), 1)

    def test_evidence_retained_across_iterations(self) -> None:
        """Requirement: evidence collected in early iterations is retained for replanning and synthesis."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "agent architecture"}',
                '{"decision": "continue", "query": "vector store"}',
                '{"decision": "finish"}',
                "Final answer combining evidence across multiple iterations.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Explain entire AURA system")

        self.assertEqual(len(res.queries), 2)
        self.assertEqual(len(res.evidence), 2)
        # Check that replanning prompt for second decision call included evidence from first call
        self.assertGreaterEqual(len(provider.recorded_structured_requests), 2)
        second_decision_prompt = provider.recorded_structured_requests[1].messages[-1].content
        self.assertIn("AURA Agent Architecture", second_decision_prompt)
        # Check that final synthesis request included both sources
        synthesis_req = provider.recorded_requests[-1]
        self.assertIn("AURA Agent Architecture", synthesis_req.messages[-1].content)
        self.assertIn("AURA RAG Vector Store", synthesis_req.messages[-1].content)

    def test_citation_source_metadata_preservation(self) -> None:
        """Requirement: citation and source metadata are fully preserved and traceable."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "architecture"}',
                '{"decision": "finish"}',
                "Final answer grounded in architecture.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Investigate architecture")

        self.assertGreaterEqual(len(res.evidence), 1)
        ev = res.evidence[0]
        self.assertTrue(len(ev.chunk_id) > 0)
        self.assertTrue(len(ev.document_title) > 0)
        self.assertTrue(len(ev.document_source) > 0)
        self.assertIsNotNone(ev.score)
        self.assertTrue(ev.citation.startswith(f"[{ev.document_title}, Chunk: {ev.chunk_id}]"))

        # Verify citation verification helper
        answer_with_citation = f"According to {ev.citation}, stateful execution is coordinated."
        verification = res.verify_citations(answer_with_citation)
        self.assertTrue(verification["has_evidence"])
        self.assertIn(ev.citation, verification["matched_citations"])
        self.assertIn(ev.chunk_id, verification["matched_chunk_ids"])

    def test_final_result_containing_structured_evidence(self) -> None:
        """Requirement: final result contains structured evidence, serializes, and round-trips."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "vector store"}',
                '{"decision": "finish"}',
                "Grounded response on vector store.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Storage mechanisms")

        data = res.to_dict()
        self.assertEqual(data["objective"], "Storage mechanisms")
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["has_evidence"])
        self.assertTrue(data["is_grounded"])
        self.assertIsInstance(data["evidence"], list)
        self.assertIsInstance(data["sources"], list)
        self.assertIsInstance(data["queries"], list)
        self.assertIsInstance(data["citations"], list)

        # JSON serializability check
        json_output = json.dumps(data)
        self.assertIsInstance(json_output, str)

        # Round-trip deserialization
        reconstructed = ResearchResult.from_dict(json.loads(json_output))
        self.assertEqual(reconstructed.objective, res.objective)
        self.assertEqual(reconstructed.final_answer, res.final_answer)
        self.assertEqual(len(reconstructed.evidence), len(res.evidence))
        self.assertEqual(reconstructed.has_evidence, res.has_evidence)
        self.assertEqual(reconstructed.status, res.status)

    def test_no_context_research_result(self) -> None:
        """Requirement: no-context research explicitly reflects ungrounded status without inventing citations."""
        Document.objects.all().delete()
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "unobtainium"}',
                '{"decision": "finish"}',
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Properties of unobtainium")

        self.assertFalse(res.has_evidence)
        self.assertFalse(res.is_grounded)
        self.assertEqual(res.evidence, [])
        self.assertEqual(res.sources, [])
        self.assertEqual(res.citations, [])
        self.assertIn("did not provide sufficient supporting evidence", res.final_answer)
        # Ensure verification reports zero matched citations
        verif = res.verify_citations()
        self.assertFalse(verif["has_evidence"])
        self.assertEqual(verif["matched_citations"], [])

    def test_successful_multi_iteration_research(self) -> None:
        """Requirement: successful multi-iteration research loop executes cleanly and returns ResearchResult."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "agent architecture"}',
                '{"decision": "continue", "query": "vector store"}',
                '{"decision": "finish"}',
                "Synthesis after multiple search iterations.",
            ]
        )
        runtime = self._build_runtime(provider)
        res = runtime.run_research("Multi-step AURA research")

        self.assertEqual(res.status, AgentStatus.COMPLETED)
        self.assertEqual(len(res.queries), 2)
        self.assertGreaterEqual(res.iteration_count, 3)
        self.assertTrue(res.is_grounded)
        self.assertGreaterEqual(len(res.evidence), 2)
        self.assertEqual(res.objective, "Multi-step AURA research")

    def test_existing_m5_behavior_remaining_compatible(self) -> None:
        """Requirement: existing AgentRuntime.run() remains fully compatible and exposes research_result."""
        provider = ScriptedLLMProvider(
            responses=[
                '{"decision": "continue", "query": "agent architecture"}',
                '{"decision": "finish"}',
                "Compatible final output answer.",
            ]
        )
        runtime = self._build_runtime(provider)
        # Calling standard AgentRuntime.run()
        run_res = runtime.run("Compatibility test objective")

        self.assertTrue(run_res.is_success)
        self.assertEqual(run_res.status, AgentStatus.COMPLETED)
        self.assertIn("Compatible final output answer", run_res.final_output)

        # Accessing research_result property on AgentRunResult
        research_res = run_res.research_result
        self.assertIsInstance(research_res, ResearchResult)
        self.assertEqual(research_res.objective, "Compatibility test objective")
        self.assertEqual(research_res.final_answer, run_res.final_output)
        self.assertTrue(research_res.has_evidence)


class ResearchResultIntegrityTests(TestCase):
    """Regression tests for ResearchResult integrity, citation verification, and serialization safety."""

    def test_citation_verification_strict_matching(self) -> None:
        """Requirement: citation verification requires actual citation, not bare title or chunk ID."""
        ev1 = ResearchEvidence(
            chunk_id="c1",
            document_title="Architecture Overview",
            document_source="docs/arch.md",
            content="Django backend routes requests.",
        )
        ev2 = ResearchEvidence(
            chunk_id="c2",
            document_title="Architecture Overview",
            document_source="docs/arch.md",
            content="Agent runtime handles replanning.",
        )
        ev_short = ResearchEvidence(
            chunk_id="1",
            document_title="Numeric Guide",
            document_source="docs/num.md",
            content="Step one is initialization.",
        )
        res = ResearchResult(
            objective="Analyze architecture",
            final_answer="",
            evidence=[ev1, ev2, ev_short],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=True,
        )

        # 1. Title mentioned without citation -> NOT verified
        title_only_text = "The Architecture Overview and Numeric Guide describe our system."
        verif = res.verify_citations(title_only_text)
        self.assertEqual(verif["matched_citations"], [])
        self.assertEqual(verif["matched_chunk_ids"], [])

        # 2. Short chunk ID '1' appearing naturally in text -> NOT treated as citation
        natural_num_text = "We have 1 primary reason to build this."
        verif = res.verify_citations(natural_num_text)
        self.assertEqual(verif["matched_citations"], [])
        self.assertEqual(verif["matched_chunk_ids"], [])

        # 3. One chunk cited (canonical) -> only that chunk verified; chunk 2 from same doc remains unverified
        one_chunk_text = f"As noted in {ev1.citation}, Django backend routes requests."
        verif = res.verify_citations(one_chunk_text)
        self.assertEqual(verif["matched_citations"], [ev1.citation])
        self.assertEqual(verif["matched_chunk_ids"], ["c1"])
        self.assertNotIn(ev2.citation, verif["matched_citations"])
        self.assertNotIn("c2", verif["matched_chunk_ids"])

        # 4. Boundary-delimited chunk citation -> verified
        boundary_chunk_text = "Refer to Chunk: c2 for runtime details."
        verif = res.verify_citations(boundary_chunk_text)
        self.assertEqual(verif["matched_citations"], [ev2.citation])
        self.assertEqual(verif["matched_chunk_ids"], ["c2"])

        # 5. Multiple canonical citations -> all correctly verified
        multi_citation_text = f"Synthesizing {ev1.citation} and {ev2.citation} with {ev_short.citation}."
        verif = res.verify_citations(multi_citation_text)
        self.assertEqual(len(verif["matched_citations"]), 3)
        self.assertIn(ev1.citation, verif["matched_citations"])
        self.assertIn(ev2.citation, verif["matched_citations"])
        self.assertIn(ev_short.citation, verif["matched_citations"])

    def test_mutable_state_leak_prevention(self) -> None:
        """Requirement: to_dict() must not expose internal mutable structures like chunk_ids."""
        ev = ResearchEvidence(
            chunk_id="c1",
            document_title="Overview",
            document_source="doc.md",
            content="content",
        )
        sources = [
            {
                "document_title": "Overview",
                "document_source": "doc.md",
                "document_id": "doc-1",
                "chunk_count": 1,
                "chunk_ids": ["c1"],
            }
        ]
        res = ResearchResult(
            objective="Test state leak",
            final_answer="Answer",
            evidence=[ev],
            sources=sources,
            queries=["q1"],
            iteration_count=1,
            has_evidence=True,
            metadata={"tag": "initial"},
        )

        serialized = res.to_dict()
        # Mutate serialized chunk_ids and other structures
        serialized["sources"][0]["chunk_ids"].append("c2-mutated")
        serialized["sources"][0]["document_title"] = "Mutated Title"
        serialized["queries"].append("q2-mutated")
        serialized["metadata"]["tag"] = "mutated"

        # Verify original ResearchResult instance is unchanged
        self.assertEqual(res.sources[0]["chunk_ids"], ["c1"])
        self.assertEqual(res.sources[0]["document_title"], "Overview")
        self.assertEqual(res.queries, ["q1"])
        self.assertEqual(res.metadata["tag"], "initial")

    def test_from_dict_null_handling(self) -> None:
        """Requirement: from_dict() must handle explicit null/None values on nullable fields."""
        data_with_nulls = {
            "objective": "Nullable test",
            "final_answer": "Final",
            "evidence": None,
            "sources": None,
            "queries": None,
            "iteration_count": None,
            "has_evidence": None,
            "status": None,
            "duration_ms": None,
            "metadata": None,
        }
        res = ResearchResult.from_dict(data_with_nulls)
        self.assertEqual(res.objective, "Nullable test")
        self.assertEqual(res.final_answer, "Final")
        self.assertEqual(res.evidence, [])
        self.assertEqual(res.sources, [])
        self.assertEqual(res.queries, [])
        self.assertEqual(res.iteration_count, 0)
        self.assertEqual(res.has_evidence, False)
        self.assertEqual(res.status, AgentStatus.COMPLETED)
        self.assertEqual(res.duration_ms, 0.0)
        self.assertEqual(res.metadata, {})

        # Ensure to_dict() works without TypeError on null-constructed result
        data = res.to_dict()
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["queries"], [])
        self.assertEqual(data["metadata"], {})

    def test_from_dict_defensive_copying(self) -> None:
        """Requirement: from_dict() must not expose input dictionary collections to mutation."""
        input_sources = [{"document_title": "T", "chunk_ids": ["c1"]}]
        input_queries = ["q1"]
        input_meta = {"key": "val"}
        data = {
            "objective": "Test",
            "final_answer": "Answer",
            "sources": input_sources,
            "queries": input_queries,
            "metadata": input_meta,
        }
        res = ResearchResult.from_dict(data)

        # Mutate input structures
        input_sources[0]["chunk_ids"].append("c2")
        input_queries.append("q2")
        input_meta["key"] = "changed"

        # Verify res is unaffected
        self.assertEqual(res.sources[0]["chunk_ids"], ["c1"])
        self.assertEqual(res.queries, ["q1"])
        self.assertEqual(res.metadata["key"], "val")

    def test_from_state_null_metadata(self) -> None:
        """Requirement: from_state() safely handles 'metadata': None in tool results without AttributeError."""
        state = AgentState(run_id="run-null-meta", objective="Investigate null metadata")
        state.record_tool_result({
            "tool_name": "rag_search",
            "output": [
                {
                    "chunk_id": "c1",
                    "document_title": "Doc",
                    "document_source": "doc.md",
                    "content": "content",
                }
            ],
            "is_error": False,
            "metadata": None,  # Explicitly None
        })
        state.metadata = None  # Explicitly None

        res = ResearchResult.from_state(state)
        self.assertEqual(res.objective, "Investigate null metadata")
        self.assertEqual(len(res.evidence), 1)
        self.assertEqual(res.queries, [])
        self.assertEqual(res.metadata, {})

    def test_source_aggregation_distinct_document_ids(self) -> None:
        """Requirement: documents with identical title and source but different IDs produce separate sources."""
        doc_a_chunk = ResearchEvidence(
            chunk_id="chunk-a1",
            document_title="Overview",
            document_source="unknown",
            document_id="doc-a",
            content="Content of document A",
        )
        doc_b_chunk = ResearchEvidence(
            chunk_id="chunk-b1",
            document_title="Overview",
            document_source="unknown",
            document_id="doc-b",
            content="Content of document B",
        )
        sources = _aggregate_sources([doc_a_chunk, doc_b_chunk])
        self.assertEqual(len(sources), 2)
        doc_ids = {s["document_id"] for s in sources}
        self.assertEqual(doc_ids, {"doc-a", "doc-b"})
        for s in sources:
            self.assertEqual(s["document_title"], "Overview")
            self.assertEqual(s["document_source"], "unknown")
            self.assertEqual(s["chunk_count"], 1)

    def test_research_planner_propagates_timeout_budget_to_structured_decision(self) -> None:
        """Verify ResearchPlanner includes remaining timeout budget in structured_output metadata."""
        from unittest.mock import MagicMock
        from agent.execution.limits import LimitTracker

        mock_gateway = MagicMock()
        mock_gateway.structured_output.return_value = StructuredOutputResponse(
            data={"decision": "continue", "query": "quantum algorithms", "tool": "rag_search"},
            raw_text='{"decision": "continue", "query": "quantum algorithms", "tool": "rag_search"}',
            provider="mock",
            model="mock-model",
        )

        planner = ResearchPlanner(gateway=mock_gateway, mode="knowledge_base")
        state = AgentState.create("Research quantum computing")
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=45.0))

        planner.plan("Research quantum computing", state, tracker=tracker)

        mock_gateway.structured_output.assert_called_once()
        req = mock_gateway.structured_output.call_args[0][0]
        self.assertIn("timeout", req.metadata)
        self.assertAlmostEqual(req.metadata["timeout"], 45.0, delta=0.5)

    def test_research_planner_propagates_timeout_budget_to_grounded_synthesis(self) -> None:
        """Verify ResearchPlanner includes remaining timeout budget in synthesis generate metadata."""
        from unittest.mock import MagicMock
        from agent.execution.limits import LimitTracker

        mock_gateway = MagicMock()
        mock_gateway.structured_output.return_value = StructuredOutputResponse(
            data={"decision": "finish"},
            raw_text='{"decision": "finish"}',
            provider="mock",
            model="mock-model",
        )
        mock_gateway.generate.return_value = GenerationResponse(
            text="Grounded final answer with citations",
            provider="mock",
            model="mock-model",
        )

        planner = ResearchPlanner(gateway=mock_gateway, mode="knowledge_base")
        state = AgentState.create("Research quantum computing")
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=50.0))

        # Add evidence chunk to state
        state.record_tool_result({
            "tool_name": "rag_search",
            "is_error": False,
            "output": [{
                "chunk_id": "c1",
                "document_title": "Quantum Paper",
                "document_source": "arxiv",
                "content": "Quantum computing uses qubits.",
            }],
        })

        plan = planner.plan("Research quantum computing", state, tracker=tracker)

        mock_gateway.generate.assert_called_once()
        req = mock_gateway.generate.call_args[0][0]
        self.assertIn("timeout", req.metadata)
        self.assertAlmostEqual(req.metadata["timeout"], 50.0, delta=0.5)
        self.assertEqual(plan.steps[0].payload["final_answer"], "Grounded final answer with citations")

    def test_research_planner_propagates_timeout_budget_to_model_knowledge_synthesis(self) -> None:
        """Verify ResearchPlanner includes remaining timeout budget in model_knowledge synthesis."""
        from unittest.mock import MagicMock
        from agent.execution.limits import LimitTracker

        mock_gateway = MagicMock()
        mock_gateway.generate.return_value = GenerationResponse(
            text="Pretrained knowledge answer",
            provider="mock",
            model="mock-model",
        )

        planner = ResearchPlanner(gateway=mock_gateway, mode="model_knowledge")
        state = AgentState.create("What is gravity?")
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=28.0))

        plan = planner.plan("What is gravity?", state, tracker=tracker)


        mock_gateway.generate.assert_called_once()
        req = mock_gateway.generate.call_args[0][0]
        self.assertIn("timeout", req.metadata)
        self.assertAlmostEqual(req.metadata["timeout"], 28.0, delta=0.5)
        self.assertEqual(plan.steps[0].payload["final_answer"], "Pretrained knowledge answer")

    def test_planner_exposes_web_url_in_evidence_representation(self) -> None:
        """Planner includes web document URL in evidence items for model decision."""
        from unittest.mock import MagicMock
        from gateway.types import StructuredOutputResponse

        mock_gateway = MagicMock()
        mock_gateway.structured_output.return_value = StructuredOutputResponse(
            data={"decision": "finish"},
            raw_text='{"decision": "finish"}',
            provider="mock",
            model="mock-model",
        )
        mock_gateway.generate.return_value = GenerationResponse(
            text="Final grounded answer",
            provider="mock",
            model="mock-model",
        )

        planner = ResearchPlanner(gateway=mock_gateway, mode="knowledge_base")
        state = AgentState.create("Tell me about Ronaldo")
        state.record_tool_result({
            "tool_name": "rag_search",
            "is_error": False,
            "output": [{
                "chunk_id": "c-ronaldo-1",
                "document_title": "ronaldo",
                "document_source": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
                "url": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
                "content": "Cristiano Ronaldo is a Portuguese footballer.",
                "metadata": {"url": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"},
            }],
            "metadata": {"query": "ronaldo early career"},
        })

        planner.plan("Tell me about Ronaldo", state)

        mock_gateway.structured_output.assert_called_once()
        req = mock_gateway.structured_output.call_args[0][0]
        user_msg = next(m.content for m in req.messages if m.role == "user")
        self.assertIn("Source: ronaldo (URL: https://en.wikipedia.org/wiki/Cristiano_Ronaldo)", user_msg)

    def test_planner_system_prompt_includes_query_non_repetition_guidance(self) -> None:
        """Planner system prompt includes guidance to formulate new queries and avoid repeats."""
        from unittest.mock import MagicMock
        from gateway.types import StructuredOutputResponse

        mock_gateway = MagicMock()
        mock_gateway.structured_output.return_value = StructuredOutputResponse(
            data={"decision": "finish"},
            raw_text='{"decision": "finish"}',
            provider="mock",
            model="mock-model",
        )
        mock_gateway.generate.return_value = GenerationResponse(
            text="Final answer",
            provider="mock",
            model="mock-model",
        )

        # Single tool mode
        planner_single = ResearchPlanner(gateway=mock_gateway, mode="knowledge_base")
        state = AgentState.create("Test objective")
        planner_single.plan("Test objective", state)
        req_single = mock_gateway.structured_output.call_args[0][0]
        sys_msg_single = next(m.content for m in req_single.messages if m.role == "system")
        self.assertIn("Formulate new search queries; do not repeat queries that have already been executed.", sys_msg_single)

        # Multi-tool mode
        mock_gateway.reset_mock()
        planner_multi = ResearchPlanner(gateway=mock_gateway, mode="web_knowledge_base")
        planner_multi.plan("Test objective", state)
        req_multi = mock_gateway.structured_output.call_args[0][0]
        sys_msg_multi = next(m.content for m in req_multi.messages if m.role == "system")
        self.assertIn("Formulate new search queries; do not repeat queries that have already been executed.", sys_msg_multi)
