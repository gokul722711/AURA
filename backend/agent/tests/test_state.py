"""Tests for AgentState and state transitions."""

from django.test import SimpleTestCase

from agent.exceptions import InvalidStateTransitionError
from agent.state import AgentState, AgentStatus, StepExecutionRecord


class AgentStateTests(SimpleTestCase):
    """Unit tests for AgentState creation, mutation, and serialization."""

    def test_state_creation(self):
        state = AgentState.create(objective="Calculate total revenue")
        self.assertIsNotNone(state.run_id)
        self.assertEqual(state.objective, "Calculate total revenue")
        self.assertEqual(state.status, AgentStatus.PENDING)
        self.assertEqual(state.iteration, 0)
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.current_step_index, 0)
        self.assertEqual(len(state.step_history), 0)
        self.assertEqual(len(state.tool_results), 0)
        self.assertIsNone(state.final_output)

    def test_valid_transitions(self):
        state = AgentState.create(objective="Test")
        # PENDING -> RUNNING
        state.transition_to(AgentStatus.RUNNING)
        self.assertEqual(state.status, AgentStatus.RUNNING)
        self.assertFalse(state.is_terminal())

        # RUNNING -> WAITING
        state.transition_to(AgentStatus.WAITING)
        self.assertEqual(state.status, AgentStatus.WAITING)

        # WAITING -> RUNNING
        state.transition_to(AgentStatus.RUNNING)
        self.assertEqual(state.status, AgentStatus.RUNNING)

        # RUNNING -> COMPLETED
        state.transition_to(AgentStatus.COMPLETED)
        self.assertEqual(state.status, AgentStatus.COMPLETED)
        self.assertTrue(state.is_terminal())

    def test_direct_completion_transition(self):
        state = AgentState.create(objective="Quick finish")
        state.transition_to(AgentStatus.RUNNING)
        state.transition_to(AgentStatus.COMPLETED)
        self.assertEqual(state.status, AgentStatus.COMPLETED)

    def test_failure_transition(self):
        state = AgentState.create(objective="Error case")
        state.transition_to(AgentStatus.RUNNING)
        state.transition_to(AgentStatus.FAILED)
        self.assertEqual(state.status, AgentStatus.FAILED)
        self.assertTrue(state.is_terminal())

    def test_cancellation_transition_from_pending(self):
        state = AgentState.create(objective="Cancel immediately")
        state.transition_to(AgentStatus.CANCELLED)
        self.assertEqual(state.status, AgentStatus.CANCELLED)
        self.assertTrue(state.is_terminal())

    def test_cancellation_transition_from_running(self):
        state = AgentState.create(objective="Cancel while running")
        state.transition_to(AgentStatus.RUNNING)
        state.transition_to(AgentStatus.CANCELLED)
        self.assertEqual(state.status, AgentStatus.CANCELLED)
        self.assertTrue(state.is_terminal())

    def test_invalid_transition_from_pending_to_completed_raises(self):
        state = AgentState.create(objective="Invalid jump")
        with self.assertRaises(InvalidStateTransitionError):
            state.transition_to(AgentStatus.COMPLETED)

    def test_terminal_state_cannot_transition(self):
        state = AgentState.create(objective="Finished")
        state.transition_to(AgentStatus.RUNNING)
        state.transition_to(AgentStatus.COMPLETED)
        with self.assertRaises(InvalidStateTransitionError):
            state.transition_to(AgentStatus.RUNNING)

    def test_record_step_and_tool_result(self):
        state = AgentState.create(objective="Record test")
        record = StepExecutionRecord(
            step_id="step-1",
            action_type="tool",
            status="completed",
            output={"result": 42},
            duration_ms=12.5,
        )
        state.record_step(record)
        self.assertEqual(len(state.step_history), 1)
        self.assertEqual(state.step_history[0].step_id, "step-1")

        state.record_tool_result({"tool_name": "calculator", "output": 42})
        self.assertEqual(len(state.tool_results), 1)

        state.add_error("Minor warning")
        self.assertEqual(state.errors, ["Minor warning"])

    def test_serialization_roundtrip(self):
        state = AgentState.create(
            objective="Serialize me",
            metadata={"priority": "high"},
        )
        state.transition_to(AgentStatus.RUNNING)
        state.iteration = 2
        state.current_step_index = 1
        state.final_output = "Final answer"
        record = StepExecutionRecord(
            step_id="s1",
            action_type="model",
            status="completed",
            output="Generated text",
            metadata={"tokens": 15},
        )
        state.record_step(record)
        state.record_tool_result({"calc": 100})

        data = state.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["objective"], "Serialize me")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["metadata"]["priority"], "high")

        restored = AgentState.from_dict(data)
        self.assertEqual(restored.run_id, state.run_id)
        self.assertEqual(restored.objective, "Serialize me")
        self.assertEqual(restored.status, AgentStatus.RUNNING)
        self.assertEqual(restored.iteration, 2)
        self.assertEqual(restored.current_step_index, 1)
        self.assertEqual(restored.final_output, "Final answer")
        self.assertEqual(len(restored.step_history), 1)
        self.assertEqual(restored.step_history[0].metadata["tokens"], 15)
        self.assertEqual(len(restored.tool_results), 1)
