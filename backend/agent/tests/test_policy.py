"""Tests for ToolPolicy and DefaultToolPolicy."""

from django.test import SimpleTestCase

from agent.state import AgentState
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.policy import DefaultToolPolicy, PolicyDecision
from agent.tools.registry import ToolRegistry


class ToolPolicyTests(SimpleTestCase):
    """Tests for DefaultToolPolicy evaluation."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.calc = CalculatorTool()
        self.echo = MockEchoTool()
        self.registry.register(self.calc)
        self.registry.register(self.echo)
        self.state = AgentState.create("Policy test")

    def test_unknown_tool(self):
        policy = DefaultToolPolicy()
        result = policy.evaluate("unknown_tool", {}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.UNKNOWN)
        self.assertIn("does not exist", result.reason)

    def test_allowed_by_default_when_no_filters(self):
        policy = DefaultToolPolicy()
        result = policy.evaluate("calculator", {"expression": "2 + 2"}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.ALLOWED)

    def test_explicit_denylist(self):
        policy = DefaultToolPolicy(denied_tools={"calculator"})
        result = policy.evaluate("calculator", {"expression": "2 + 2"}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.DENIED)
        self.assertIn("explicitly denied", result.reason)

    def test_explicit_allowlist_allowed(self):
        policy = DefaultToolPolicy(allowed_tools={"echo"})
        result = policy.evaluate("echo", {"message": "hi"}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.ALLOWED)

    def test_explicit_allowlist_denied(self):
        policy = DefaultToolPolicy(allowed_tools={"echo"})
        result = policy.evaluate("calculator", {"expression": "1 + 1"}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.DENIED)
        self.assertIn("not in the allowed tools", result.reason)

    def test_missing_required_input_rejected(self):
        policy = DefaultToolPolicy()
        # Calculator requires 'expression'
        result = policy.evaluate("calculator", {}, self.registry, self.state)
        self.assertEqual(result.decision, PolicyDecision.DENIED)
        self.assertIn("Missing required input parameter 'expression'", result.reason)

    def test_non_dict_input_rejected(self):
        policy = DefaultToolPolicy()
        result = policy.evaluate("calculator", "not a dict", self.registry, self.state)  # type: ignore
        self.assertEqual(result.decision, PolicyDecision.DENIED)
        self.assertIn("must be provided as a dictionary", result.reason)
