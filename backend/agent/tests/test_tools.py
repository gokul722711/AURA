"""Tests for Tool contract and builtin tools."""

from django.test import SimpleTestCase

from agent.tools.base import ToolResult
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool


class ToolResultTests(SimpleTestCase):
    """Tests for ToolResult serialization and properties."""

    def test_tool_result_creation_and_dict(self):
        res = ToolResult(
            tool_name="test_tool",
            output={"val": 123},
            is_error=False,
            metadata={"source": "unit_test"},
        )
        self.assertEqual(res.tool_name, "test_tool")
        self.assertEqual(res.output, {"val": 123})
        self.assertFalse(res.is_error)
        self.assertIsNone(res.error_message)

        d = res.to_dict()
        self.assertEqual(d["tool_name"], "test_tool")
        self.assertEqual(d["output"], {"val": 123})

        restored = ToolResult.from_dict(d)
        self.assertEqual(restored, res)


class CalculatorToolTests(SimpleTestCase):
    """Tests for CalculatorTool safe AST evaluation."""

    def setUp(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        res = self.calc.execute(expression="2 + 3")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 5)

        res = self.calc.execute(expression="10 - 4")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 6)

        res = self.calc.execute(expression="3 * 4")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 12)

        res = self.calc.execute(expression="15 / 3")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 5.0)

    def test_complex_expressions(self):
        res = self.calc.execute(expression="(2 + 3) * (4 - 1)")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 15)

        res = self.calc.execute(expression="10 // 3")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 3)

        res = self.calc.execute(expression="10 % 3")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 1)

        res = self.calc.execute(expression="2 ** 4")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, 16)

    def test_unary_operators(self):
        res = self.calc.execute(expression="-5 + +3")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, -2)

    def test_floating_point(self):
        res = self.calc.execute(expression="3.5 * 2.0")
        self.assertFalse(res.is_error)
        self.assertAlmostEqual(res.output, 7.0)

    def test_zero_division(self):
        res = self.calc.execute(expression="10 / 0")
        self.assertTrue(res.is_error)
        self.assertIn("Division by zero", res.error_message)

    def test_empty_or_whitespace_expression(self):
        res = self.calc.execute(expression="")
        self.assertTrue(res.is_error)

        res = self.calc.execute(expression="   ")
        self.assertTrue(res.is_error)

    def test_syntax_error(self):
        res = self.calc.execute(expression="2 + ")
        self.assertTrue(res.is_error)

    def test_rejection_of_names_and_builtins(self):
        """Names like os, __import__, abs must be rejected."""
        res = self.calc.execute(expression="__import__('os').system('ls')")
        self.assertTrue(res.is_error)
        self.assertIn("Forbidden", res.error_message)

        res = self.calc.execute(expression="abs(-5)")
        self.assertTrue(res.is_error)

    def test_rejection_of_boolean_literals(self):
        res = self.calc.execute(expression="True + 1")
        self.assertTrue(res.is_error)

    def test_rejection_of_strings(self):
        res = self.calc.execute(expression="'hello' + 'world'")
        self.assertTrue(res.is_error)

    def test_rejection_of_attributes(self):
        res = self.calc.execute(expression="(1).__class__")
        self.assertTrue(res.is_error)

    def test_rejection_of_excessive_exponent_dos(self):
        res = self.calc.execute(expression="2 ** 1001")
        self.assertTrue(res.is_error)
        self.assertIn("safety limit", res.error_message)


class MockEchoToolTests(SimpleTestCase):
    """Tests for MockEchoTool."""

    def setUp(self):
        self.echo = MockEchoTool()

    def test_echo_success(self):
        res = self.echo.execute(message="hello world")
        self.assertFalse(res.is_error)
        self.assertEqual(res.output, {"echo": "hello world"})

    def test_echo_missing_parameter(self):
        res = self.echo.execute()
        self.assertTrue(res.is_error)
        self.assertIn("Missing required parameter", res.error_message)
