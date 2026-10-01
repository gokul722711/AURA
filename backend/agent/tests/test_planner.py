"""Tests for planning contracts and MockPlanner."""

from django.test import SimpleTestCase

from agent.exceptions import InvalidPlanError
from agent.planning.base import ActionType, AgentStep, Plan
from agent.planning.mock import MockPlanner
from agent.state import AgentState


class PlanningContractsTests(SimpleTestCase):
    """Tests for AgentStep and Plan contracts."""

    def test_agent_step_creation(self):
        step = AgentStep(
            step_id="step-1",
            action_type=ActionType.TOOL,
            description="Run calculator",
            payload={"tool_name": "calculator", "tool_input": {"expression": "1 + 1"}},
        )
        self.assertEqual(step.step_id, "step-1")
        self.assertEqual(step.action_type, ActionType.TOOL)
        self.assertEqual(step.description, "Run calculator")

        d = step.to_dict()
        self.assertEqual(d["action_type"], "tool")

        restored = AgentStep.from_dict(d)
        self.assertEqual(restored, step)

    def test_agent_step_string_action_type_normalized(self):
        step = AgentStep(
            step_id="step-2",
            action_type="model",  # string converted to ActionType
            description="Run model",
        )
        self.assertEqual(step.action_type, ActionType.MODEL)

    def test_agent_step_invalid_action_type(self):
        with self.assertRaises(ValueError):
            AgentStep(
                step_id="step-3",
                action_type="invalid_type",
                description="desc",
            )

    def test_plan_creation_and_indexing(self):
        s1 = AgentStep(step_id="s1", action_type=ActionType.TOOL, description="Step 1")
        s2 = AgentStep(step_id="s2", action_type=ActionType.FINISH, description="Step 2")
        plan = Plan(plan_id="p1", objective="Test plan", steps=(s1, s2))

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan.get_step(0), s1)
        self.assertEqual(plan.get_step(1), s2)
        self.assertIsNone(plan.get_step(2))
        self.assertIsNone(plan.get_step(-1))

        d = plan.to_dict()
        restored = Plan.from_dict(d)
        self.assertEqual(restored, plan)

    def test_plan_empty_steps_raises(self):
        with self.assertRaises(InvalidPlanError):
            Plan(plan_id="p-empty", objective="No steps", steps=())


class MockPlannerTests(SimpleTestCase):
    """Tests for MockPlanner behavior."""

    def setUp(self):
        self.state = AgentState.create("Test objective")

    def test_preconfigured_steps(self):
        steps = [
            AgentStep(step_id="custom-1", action_type=ActionType.TOOL, description="Custom tool"),
            AgentStep(step_id="custom-2", action_type=ActionType.FINISH, description="Custom finish"),
        ]
        planner = MockPlanner(steps=steps)
        plan = planner.plan("Arbitrary objective", self.state)

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan.get_step(0).step_id, "custom-1")
        self.assertEqual(plan.get_step(1).step_id, "custom-2")
        self.assertEqual(planner.call_count, 1)

    def test_rule_based_calculation_plan(self):
        planner = MockPlanner()
        plan = planner.plan("Please calculate 10 + 20", self.state)

        self.assertGreaterEqual(len(plan), 2)
        first_step = plan.get_step(0)
        self.assertEqual(first_step.action_type, ActionType.TOOL)
        self.assertEqual(first_step.payload.get("tool_name"), "calculator")

        last_step = plan.get_step(len(plan) - 1)
        self.assertEqual(last_step.action_type, ActionType.FINISH)

    def test_rule_based_echo_plan(self):
        planner = MockPlanner()
        plan = planner.plan("echo hello world", self.state)

        self.assertGreaterEqual(len(plan), 2)
        first_step = plan.get_step(0)
        self.assertEqual(first_step.action_type, ActionType.TOOL)
        self.assertEqual(first_step.payload.get("tool_name"), "echo")

    def test_rule_based_default_model_plan(self):
        planner = MockPlanner()
        plan = planner.plan("Synthesize research notes", self.state)

        self.assertGreaterEqual(len(plan), 2)
        first_step = plan.get_step(0)
        self.assertEqual(first_step.action_type, ActionType.MODEL)
        self.assertIn("Synthesize research notes", first_step.payload.get("prompt", ""))
