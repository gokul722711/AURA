"""Tests for ToolRegistry."""

from django.test import SimpleTestCase

from agent.exceptions import DuplicateToolError, ToolNotFoundError
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.registry import ToolRegistry


class ToolRegistryTests(SimpleTestCase):
    """Tests for ToolRegistry operations."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.calc = CalculatorTool()
        self.echo = MockEchoTool()

    def test_register_and_get(self):
        self.registry.register(self.calc)
        self.assertTrue(self.registry.has_tool("calculator"))
        retrieved = self.registry.get("calculator")
        self.assertIs(retrieved, self.calc)

    def test_duplicate_registration_raises(self):
        self.registry.register(self.calc)
        with self.assertRaises(DuplicateToolError):
            self.registry.register(self.calc)

    def test_get_missing_tool_raises(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.get("nonexistent")

    def test_list_tools(self):
        self.registry.register(self.calc)
        self.registry.register(self.echo)
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 2)
        names = {t.name for t in tools}
        self.assertEqual(names, {"calculator", "echo"})

    def test_get_schemas(self):
        self.registry.register(self.calc)
        schemas = self.registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "calculator")
        self.assertIn("parameters", schemas[0])
        self.assertEqual(schemas[0]["parameters"]["required"], ["expression"])

    def test_invalid_tool_registration(self):
        with self.assertRaises(TypeError):
            self.registry.register("not a tool")  # type: ignore
