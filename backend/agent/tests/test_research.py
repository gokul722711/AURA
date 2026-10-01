"""Deterministic offline unit and integration tests for M5 Autonomous Research."""

import json
from collections.abc import Iterator
from typing import Any

from django.test import TestCase

from agent.execution.limits import ExecutionLimits
from agent.planning.research import RESEARCH_DECISION_SCHEMA, ResearchPlanner
from agent.research import create_research_runtime
from agent.state import AgentStatus
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
