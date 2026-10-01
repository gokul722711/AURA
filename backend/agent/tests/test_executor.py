"""Tests for StepExecutor."""

from django.test import SimpleTestCase

from agent.events.trace import ExecutionTrace
from agent.exceptions import (
    ModelExecutionError,
    ToolNotFoundError,
    ToolPolicyViolationError,
)
from agent.execution.executor import StepExecutor
from agent.execution.limits import ExecutionLimits, LimitTracker
from agent.planning.base import ActionType, AgentStep
from agent.state import AgentState
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.policy import DefaultToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.config import GatewayConfig
from gateway.gateway import ModelGateway
from gateway.providers.mock import MockLLMProvider


class StepExecutorTests(SimpleTestCase):
    """Tests for StepExecutor step dispatching and boundary integration."""

    def setUp(self):
        self.mock_provider = MockLLMProvider()
        self.gateway = ModelGateway(
            config=GatewayConfig(provider="mock", model="mock-model"),
            provider=self.mock_provider,
        )
        self.registry = ToolRegistry()
        self.calc = CalculatorTool()
        self.echo = MockEchoTool()
        self.registry.register(self.calc)
        self.registry.register(self.echo)

        self.policy = DefaultToolPolicy()
        self.executor = StepExecutor(
            gateway=self.gateway,
            registry=self.registry,
            policy=self.policy,
        )
        self.state = AgentState.create("Executor test")
        self.trace = ExecutionTrace(self.state.run_id)
        self.tracker = LimitTracker(ExecutionLimits())

    def test_execute_model_step(self):
        step = AgentStep(
            step_id="m1",
            action_type=ActionType.MODEL,
            description="Generate text",
            payload={"prompt": "What is 2+2?"},
        )
        record = self.executor.execute_step(step, self.state, self.trace, self.tracker)

        self.assertEqual(record.status, "completed")
        self.assertIn("Mock response to:", record.output)
        self.assertIn("usage", record.metadata)

        # Check trace recorded events
        req_events = self.trace.get_events_by_type("ModelCallRequested")
        comp_events = self.trace.get_events_by_type("ModelCallCompleted")
        self.assertEqual(len(req_events), 1)
        self.assertEqual(len(comp_events), 1)

    def test_execute_tool_step_calculator(self):
        step = AgentStep(
            step_id="t1",
            action_type=ActionType.TOOL,
            description="Compute value",
            payload={"tool_name": "calculator", "tool_input": {"expression": "15 * 3"}},
        )
        record = self.executor.execute_step(step, self.state, self.trace, self.tracker)

        self.assertEqual(record.status, "completed")
        self.assertEqual(record.output, 45)
        self.assertEqual(self.tracker.tool_calls, 1)

        # Check state tool results
        self.assertEqual(len(self.state.tool_results), 1)
        self.assertEqual(self.state.tool_results[0]["output"], 45)

        # Check trace
        self.assertEqual(self.trace.tool_call_count(), 1)

    def test_execute_finish_step(self):
        step = AgentStep(
            step_id="f1",
            action_type=ActionType.FINISH,
            description="Complete",
            payload={"final_answer": "42 is the answer"},
        )
        record = self.executor.execute_step(step, self.state, self.trace, self.tracker)

        self.assertEqual(record.status, "completed")
        self.assertEqual(record.output, "42 is the answer")

    def test_execute_tool_policy_denial(self):
        strict_policy = DefaultToolPolicy(denied_tools={"calculator"})
        executor = StepExecutor(
            gateway=self.gateway,
            registry=self.registry,
            policy=strict_policy,
        )
        step = AgentStep(
            step_id="denied-step",
            action_type=ActionType.TOOL,
            description="Denied calc",
            payload={"tool_name": "calculator", "tool_input": {"expression": "1+1"}},
        )
        with self.assertRaises(ToolPolicyViolationError):
            executor.execute_step(step, self.state, self.trace, self.tracker)

    def test_execute_unknown_tool_raises(self):
        step = AgentStep(
            step_id="unknown-step",
            action_type=ActionType.TOOL,
            description="Missing tool",
            payload={"tool_name": "nonexistent_tool", "tool_input": {}},
        )
        with self.assertRaises(ToolNotFoundError):
            self.executor.execute_step(step, self.state, self.trace, self.tracker)

    def test_model_step_missing_prompt_raises(self):
        step = AgentStep(
            step_id="bad-model",
            action_type=ActionType.MODEL,
            description="No prompt",
            payload={},
        )
        with self.assertRaises(ModelExecutionError):
            self.executor.execute_step(step, self.state, self.trace, self.tracker)
