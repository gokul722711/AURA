"""End-to-end and lifecycle tests for AgentRuntime."""

import time
from django.test import SimpleTestCase

from agent.events.types import ExecutionEvent
from agent.execution.limits import ExecutionLimits
from agent.planning.base import ActionType, AgentStep
from agent.planning.mock import MockPlanner
from agent.runtime import AgentRunResult, AgentRuntime
from agent.state import AgentStatus
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.policy import DefaultToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.config import GatewayConfig
from gateway.gateway import ModelGateway
from gateway.providers.mock import MockLLMProvider


class AgentRuntimeTests(SimpleTestCase):
    """End-to-end integration tests for the agent execution loop."""

    def setUp(self):
        self.mock_provider = MockLLMProvider()
        self.gateway = ModelGateway(
            config=GatewayConfig(provider="mock", model="mock-model"),
            provider=self.mock_provider,
        )
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.registry.register(MockEchoTool())
        self.policy = DefaultToolPolicy()

    def test_successful_multi_step_run(self):
        """Test a full 3-step run: TOOL -> MODEL -> FINISH."""
        steps = [
            AgentStep(
                step_id="step-calc",
                action_type=ActionType.TOOL,
                description="Compute value",
                payload={"tool_name": "calculator", "tool_input": {"expression": "20 + 22"}},
            ),
            AgentStep(
                step_id="step-model",
                action_type=ActionType.MODEL,
                description="Synthesize answer",
                payload={"prompt": "Summarize result 42."},
            ),
            AgentStep(
                step_id="step-finish",
                action_type=ActionType.FINISH,
                description="Conclude",
                payload={"final_answer": "Final computation complete."},
            ),
        ]
        planner = MockPlanner(steps=steps)
        runtime = AgentRuntime(
            planner=planner,
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
            limits=ExecutionLimits(max_iterations=10),
        )

        result = runtime.run("Calculate and summarize")

        self.assertIsInstance(result, AgentRunResult)
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.final_output, "Final computation complete.")
        self.assertEqual(len(result.state.step_history), 3)
        self.assertEqual(len(result.state.tool_results), 1)
        self.assertEqual(result.state.tool_results[0]["output"], 42)

        # Verify trace event sequence
        event_types = [e.event_type for e in result.trace.get_events()]
        self.assertIn("AgentRunStarted", event_types)
        self.assertIn("PlanCreated", event_types)
        self.assertIn("StepStarted", event_types)
        self.assertIn("ToolCallRequested", event_types)
        self.assertIn("ToolCallCompleted", event_types)
        self.assertIn("ModelCallRequested", event_types)
        self.assertIn("ModelCallCompleted", event_types)
        self.assertIn("StepCompleted", event_types)
        self.assertIn("AgentRunCompleted", event_types)

    def test_default_rule_based_calculation_run(self):
        """Test run with default MockPlanner for a math objective."""
        runtime = AgentRuntime(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
        )
        result = runtime.run("calculate 10 + 20")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.final_output, "Calculation complete.")
        self.assertEqual(len(result.state.tool_results), 1)

    def test_default_rule_based_echo_run(self):
        """Test run with default MockPlanner for an echo objective."""
        runtime = AgentRuntime(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
        )
        result = runtime.run("echo sample input")

        self.assertTrue(result.is_success)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("Echoed: echo sample input", result.final_output)

    def test_iteration_limit_exceeded_fails_safely(self):
        """Runtime should terminate safely as FAILED when iteration limit is reached."""
        steps = [
            AgentStep(
                step_id="step-1",
                action_type=ActionType.TOOL,
                description="Step 1",
                payload={"tool_name": "echo", "tool_input": {"message": "1"}},
            ),
            AgentStep(
                step_id="step-2",
                action_type=ActionType.TOOL,
                description="Step 2",
                payload={"tool_name": "echo", "tool_input": {"message": "2"}},
            ),
        ]
        planner = MockPlanner(steps=steps)
        # Limit to 1 iteration only
        limits = ExecutionLimits(max_iterations=1)
        runtime = AgentRuntime(
            planner=planner,
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
            limits=limits,
        )

        result = runtime.run("Testing iteration limit")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertGreater(len(result.state.errors), 0)
        self.assertIn("Iteration limit", result.state.errors[0])

        failed_events = result.trace.get_events_by_type("AgentRunFailed")
        self.assertEqual(len(failed_events), 1)

    def test_tool_call_limit_exceeded_fails_safely(self):
        """Runtime should terminate safely as FAILED when tool call limit is reached."""
        steps = [
            AgentStep(
                step_id="tool-1",
                action_type=ActionType.TOOL,
                description="Tool 1",
                payload={"tool_name": "echo", "tool_input": {"message": "1"}},
            ),
            AgentStep(
                step_id="tool-2",
                action_type=ActionType.TOOL,
                description="Tool 2",
                payload={"tool_name": "echo", "tool_input": {"message": "2"}},
            ),
        ]
        planner = MockPlanner(steps=steps)
        # Allow at most 1 tool call
        limits = ExecutionLimits(max_iterations=10, max_tool_calls=1)
        runtime = AgentRuntime(
            planner=planner,
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
            limits=limits,
        )

        result = runtime.run("Testing tool call limit")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Tool call limit", result.state.errors[0])

    def test_timeout_limit_fails_safely(self):
        """Runtime should terminate safely as FAILED when time budget is exceeded."""
        from unittest.mock import patch

        limits = ExecutionLimits(max_time_seconds=1.0)
        runtime = AgentRuntime(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
            limits=limits,
        )
        # Simulate monotonic time jumping past max_time_seconds on check
        with patch("time.monotonic", side_effect=[0.0, 0.0, 5.0, 10.0, 15.0]):
            result = runtime.run("Testing timeout")

        self.assertFalse(result.is_success)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Time limit", result.state.errors[0])

    def test_cancellation_before_execution(self):
        """Synchronous cancellation requested before run() marks state CANCELLED."""
        runtime = AgentRuntime(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
        )
        runtime.cancel("User cancelled task.")

        result = runtime.run("Any task")

        self.assertEqual(result.status, AgentStatus.CANCELLED)
        self.assertIn("User cancelled task", result.state.errors[0])

        cancel_events = result.trace.get_events_by_type("AgentRunCancelled")
        self.assertEqual(len(cancel_events), 1)

    def test_cancellation_during_execution(self):
        """Synchronous cancellation requested mid-run halts subsequent steps."""
        from agent.tools.base import Tool, ToolResult

        runtime = None

        class CancellingTool(Tool):
            @property
            def name(self) -> str:
                return "cancelling_tool"

            @property
            def description(self) -> str:
                return "Cancels the runtime."

            @property
            def input_schema(self) -> dict:
                return {"type": "object"}

            def execute(self, **kwargs) -> ToolResult:
                runtime.cancel("Cancelled mid-run")
                return ToolResult(tool_name=self.name, output="cancelled")

        registry = ToolRegistry()
        registry.register(CancellingTool())
        registry.register(MockEchoTool())

        steps = [
            AgentStep(
                step_id="s1",
                action_type=ActionType.TOOL,
                description="Step 1 cancels",
                payload={"tool_name": "cancelling_tool", "tool_input": {}},
            ),
            AgentStep(
                step_id="s2",
                action_type=ActionType.TOOL,
                description="Step 2 should never run",
                payload={"tool_name": "echo", "tool_input": {"message": "should not appear"}},
            ),
        ]
        planner = MockPlanner(steps=steps)
        runtime = AgentRuntime(planner=planner, registry=registry)

        result = runtime.run("Run with mid-cancellation")

        self.assertEqual(result.status, AgentStatus.CANCELLED)
        self.assertIn("Cancelled mid-run", result.state.errors[0])
        # Only step 1 should have been executed
        self.assertEqual(len(result.state.step_history), 1)
        self.assertEqual(result.state.step_history[0].step_id, "s1")

    def test_result_to_dict(self):
        """AgentRunResult serialization."""
        runtime = AgentRuntime(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
        )
        result = runtime.run("calculate 1 + 1")
        data = result.to_dict()

        self.assertIn("state", data)
        self.assertIn("trace", data)
        self.assertEqual(data["state"]["status"], "completed")
        self.assertGreater(data["trace"]["event_count"], 0)
