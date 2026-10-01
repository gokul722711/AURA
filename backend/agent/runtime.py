"""AgentRuntime orchestrating state, planning, step execution, and boundaries."""

from dataclasses import dataclass
from typing import Any

from agent.events.trace import ExecutionTrace
from agent.events.types import (
    agent_run_cancelled,
    agent_run_completed,
    agent_run_failed,
    agent_run_started,
    plan_created,
    step_completed,
    step_started,
)
from agent.exceptions import (
    ExecutionCancelledError,
    LimitExceededError,
)
from agent.execution.executor import StepExecutor
from agent.execution.limits import ExecutionLimits, LimitTracker
from agent.planning.base import ActionType, Planner
from agent.planning.mock import MockPlanner
from agent.state import AgentState, AgentStatus
from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.policy import DefaultToolPolicy, ToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.gateway import ModelGateway, get_gateway


@dataclass(frozen=True)
class AgentRunResult:
    """Immutable result returned at the end of an agent run."""

    state: AgentState
    trace: ExecutionTrace

    @property
    def final_output(self) -> str | None:
        return self.state.final_output

    @property
    def status(self) -> AgentStatus:
        return self.state.status

    @property
    def is_success(self) -> bool:
        return self.state.status == AgentStatus.COMPLETED

    @property
    def research_result(self) -> Any:
        """Return structured ResearchResult if available for this run."""
        from agent.results import ResearchResult

        return ResearchResult.from_run_result(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "trace": self.trace.to_dict(),
        }


def _get_default_registry() -> ToolRegistry:
    """Create a default ToolRegistry with safe builtins."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(MockEchoTool())
    return registry


class AgentRuntime:
    """Coordinates the deterministic, bounded execution of an agent."""

    def __init__(
        self,
        planner: Planner | None = None,
        executor: StepExecutor | None = None,
        limits: ExecutionLimits | None = None,
        gateway: ModelGateway | None = None,
        registry: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.planner = planner or MockPlanner()
        self.limits = limits or ExecutionLimits.from_settings()

        if executor is not None:
            self.executor = executor
        else:
            active_gateway = gateway or get_gateway()
            active_registry = registry or _get_default_registry()
            active_policy = policy or DefaultToolPolicy()
            self.executor = StepExecutor(
                gateway=active_gateway,
                registry=active_registry,
                policy=active_policy,
            )

        self._cancelled: bool = False
        self._cancel_reason: str = "Execution cancelled."

    def cancel(self, reason: str = "Execution cancelled by user.") -> None:
        """Request synchronous cancellation of the current or next run."""
        self._cancelled = True
        self._cancel_reason = reason

    def reset_cancellation(self) -> None:
        """Reset the cancellation state."""
        self._cancelled = False
        self._cancel_reason = "Execution cancelled."

    def _check_cancellation(self, state: AgentState, trace: ExecutionTrace) -> None:
        """Check if cancellation has been requested at a safe boundary."""
        if self._cancelled:
            raise ExecutionCancelledError(self._cancel_reason)

    def run(self, objective: str, metadata: dict[str, Any] | None = None) -> AgentRunResult:
        """Execute an agent run for the specified objective.

        Lifecycle:
        1. Initialize state (PENDING) and trace.
        2. Check cancellation.
        3. Transition to RUNNING.
        4. Planning phase -> produces Plan.
        5. Step loop:
           - check cancellation
           - check execution limits (time, iteration)
           - dispatch step to executor (MODEL, TOOL, FINISH)
           - record step execution and tool results
           - check cancellation
        6. Complete with AgentRunResult.
        """
        state = AgentState.create(objective=objective, metadata=metadata)
        trace = ExecutionTrace(run_id=state.run_id)
        tracker = LimitTracker(self.limits)

        trace.emit(agent_run_started(state.run_id, objective))

        try:
            # 1. Pre-execution cancellation check
            self._check_cancellation(state, trace)

            # 2. Transition to RUNNING
            state.transition_to(AgentStatus.RUNNING)

            # 3. Pre-planning cancellation check
            self._check_cancellation(state, trace)

            # 4. Planning phase
            plan = self.planner.plan(objective, state)
            state.plan = plan

            steps_payload = [
                {
                    "step_id": s.step_id,
                    "action_type": s.action_type.value,
                    "description": s.description,
                }
                for s in plan.steps
            ]
            trace.emit(plan_created(state.run_id, plan.plan_id, len(plan), steps_payload))

            # 5. Execution loop
            while not state.is_terminal():
                # Boundary check: cancellation
                self._check_cancellation(state, trace)

                # Boundary check: time limit
                tracker.check_time()

                # Boundary check: iteration limit
                tracker.increment_iteration()
                state.iteration = tracker.iterations

                step = plan.get_step(state.current_step_index)
                if step is None and getattr(self.planner, "supports_replanning", False):
                    # Planner supports dynamic replanning based on accumulated state
                    new_plan = self.planner.plan(objective, state)
                    if new_plan is not None and len(new_plan.steps) > 0:
                        plan = new_plan
                        state.plan = plan
                        state.current_step_index = 0
                        step = plan.get_step(0)
                        steps_payload = [
                            {
                                "step_id": s.step_id,
                                "action_type": s.action_type.value,
                                "description": s.description,
                            }
                            for s in plan.steps
                        ]
                        trace.emit(plan_created(state.run_id, plan.plan_id, len(plan), steps_payload))

                if step is None:
                    # Plan finished all steps without explicit FINISH step
                    state.transition_to(AgentStatus.COMPLETED)
                    trace.emit(agent_run_completed(state.run_id, state.final_output, state.iteration))
                    break

                trace.emit(step_started(state.run_id, step.step_id, step.action_type.value, step.description))

                # Dispatch step
                record = self.executor.execute_step(step, state, trace, tracker)
                state.record_step(record)

                trace.emit(step_completed(state.run_id, step.step_id, record.status, record.duration_ms))

                # Handle FINISH action
                if step.action_type == ActionType.FINISH:
                    state.final_output = record.output
                    state.transition_to(AgentStatus.COMPLETED)
                    trace.emit(agent_run_completed(state.run_id, state.final_output, state.iteration))
                    break

                # Advance to next step
                state.current_step_index += 1

                # Boundary check: post-step cancellation
                self._check_cancellation(state, trace)

        except ExecutionCancelledError as exc:
            state.transition_to(AgentStatus.CANCELLED)
            state.add_error(str(exc))
            trace.emit(agent_run_cancelled(state.run_id, str(exc), state.iteration))

        except LimitExceededError as exc:
            state.transition_to(AgentStatus.FAILED)
            state.add_error(str(exc))
            trace.emit(agent_run_failed(state.run_id, str(exc), state.iteration))

        except Exception as exc:
            state.transition_to(AgentStatus.FAILED)
            state.add_error(str(exc))
            trace.emit(agent_run_failed(state.run_id, str(exc), state.iteration))

        return AgentRunResult(state=state, trace=trace)
