"""Deterministic, safe calculator tool using strict AST allowlisting."""

import ast
import operator
from typing import Any

from agent.tools.base import Tool, ToolResult

# Permitted binary operators
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Permitted unary operators
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPONENT = 1000


def _safe_eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node against strict allowlist."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        # Reject booleans (bool is a subclass of int in Python)
        if type(node.value) in (int, float):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _ALLOWED_UNARYOPS[op_type](operand)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        # DoS guard for power calculations
        if op_type is ast.Pow:
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(
                    f"Exponent {right} exceeds safety limit of {_MAX_EXPONENT}."
                )

        return _ALLOWED_BINOPS[op_type](left, right)

    # Reject any other node type
    raise ValueError(f"Forbidden expression construct: {type(node).__name__}")


class CalculatorTool(Tool):
    """Safe arithmetic calculator tool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate arithmetic expressions safely. "
            "Supports +, -, *, /, //, %, ** and parentheses with numbers."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '(10 + 5) * 2')",
                }
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression")
        if not expression or not isinstance(expression, str):
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Parameter 'expression' must be a non-empty string.",
            )

        stripped = expression.strip()
        if not stripped:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Expression cannot be empty or whitespace.",
            )

        try:
            tree = ast.parse(stripped, mode="eval")
            result = _safe_eval_node(tree)
            return ToolResult(
                tool_name=self.name,
                output=result,
                is_error=False,
                metadata={"expression": stripped},
            )
        except ZeroDivisionError:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Division by zero.",
                metadata={"expression": stripped},
            )
        except (ValueError, SyntaxError, OverflowError) as exc:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=f"Evaluation failed: {exc}",
                metadata={"expression": stripped},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=f"Unexpected evaluation error: {exc}",
                metadata={"expression": stripped},
            )
