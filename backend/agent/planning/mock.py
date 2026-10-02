"""Deterministic MockPlanner for offline, repeatable testing."""

import uuid
from typing import Any, Sequence

from agent.planning.base import ActionType, AgentStep, Plan, Planner
from agent.state import AgentState


class MockPlanner(Planner):
    """Deterministic, provider-independent planner for tests.

    Can be initialized with a preconfigured list of steps, or dynamically
    generates a simple deterministic plan based on the objective string.
    """

    def __init__(self, steps: Sequence[AgentStep] | None = None) -> None:
        self._preconfigured_steps = tuple(steps) if steps is not None else None
        self.call_count = 0

    def plan(
        self,
        objective: str,
        state: AgentState,
        tracker: Any = None,
    ) -> Plan:
        self.call_count += 1

        if self._preconfigured_steps is not None:
            return Plan(
                plan_id=str(uuid.uuid4()),
                objective=objective,
                steps=self._preconfigured_steps,
                metadata={"planner": "MockPlanner", "mode": "preconfigured"},
            )

        # Rule-based fallback generation for testing convenience
        lower_obj = objective.lower()
        steps: list[AgentStep] = []

        if "calculate" in lower_obj or any(op in objective for op in ("+", "-", "*", "/")):
            steps.append(
                AgentStep(
                    step_id="step-calc",
                    action_type=ActionType.TOOL,
                    description="Perform calculation",
                    payload={"tool_name": "calculator", "tool_input": {"expression": "10 + 20"}},
                )
            )
            steps.append(
                AgentStep(
                    step_id="step-finish",
                    action_type=ActionType.FINISH,
                    description="Complete run with calculation result",
                    payload={"final_answer": "Calculation complete."},
                )
            )
        elif "echo" in lower_obj:
            steps.append(
                AgentStep(
                    step_id="step-echo",
                    action_type=ActionType.TOOL,
                    description="Echo test message",
                    payload={"tool_name": "echo", "tool_input": {"message": objective}},
                )
            )
            steps.append(
                AgentStep(
                    step_id="step-finish",
                    action_type=ActionType.FINISH,
                    description="Finish echo",
                    payload={"final_answer": f"Echoed: {objective}"},
                )
            )
        else:
            steps.append(
                AgentStep(
                    step_id="step-model",
                    action_type=ActionType.MODEL,
                    description="Generate response using model",
                    payload={"prompt": f"Address: {objective}"},
                )
            )
            steps.append(
                AgentStep(
                    step_id="step-finish",
                    action_type=ActionType.FINISH,
                    description="Finish execution",
                    payload={"final_answer": f"Completed objective: {objective}"},
                )
            )

        return Plan(
            plan_id=str(uuid.uuid4()),
            objective=objective,
            steps=tuple(steps),
            metadata={"planner": "MockPlanner", "mode": "rule_based"},
        )
