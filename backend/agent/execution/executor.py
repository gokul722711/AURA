"""StepExecutor for dispatching model, tool, and finish actions."""

import time
from typing import Any

from agent.events.trace import ExecutionTrace
from agent.events.types import (
    model_call_completed,
    model_call_requested,
    tool_call_completed,
    tool_call_requested,
)
from agent.exceptions import (
    ModelExecutionError,
    ToolNotFoundError,
    ToolPolicyViolationError,
    UnhandledActionTypeError,
)
from agent.execution.limits import LimitTracker
from agent.planning.base import ActionType, AgentStep
from agent.state import AgentState, StepExecutionRecord
from agent.tools.policy import PolicyDecision, ToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, Message


class StepExecutor:
    """Dispatches plan steps to ModelGateway, ToolRegistry, or completes execution."""

    def __init__(
        self,
        gateway: ModelGateway,
        registry: ToolRegistry,
        policy: ToolPolicy,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.policy = policy

    def execute_step(
        self,
        step: AgentStep,
        state: AgentState,
        trace: ExecutionTrace,
        tracker: LimitTracker,
    ) -> StepExecutionRecord:
        """Execute a single agent step and return its execution record."""
        start_time = time.monotonic()

        if step.action_type == ActionType.MODEL:
            record = self._execute_model_step(step, state, trace)
        elif step.action_type == ActionType.TOOL:
            record = self._execute_tool_step(step, state, trace, tracker)
        elif step.action_type == ActionType.FINISH:
            record = self._execute_finish_step(step)
        else:
            raise UnhandledActionTypeError(
                f"Unrecognized action_type '{step.action_type}' for step '{step.step_id}'."
            )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        record.duration_ms = duration_ms
        return record

    def _execute_model_step(
        self,
        step: AgentStep,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> StepExecutionRecord:
        prompt = step.payload.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise ModelExecutionError(
                f"Step '{step.step_id}' missing required string parameter 'prompt' in payload."
            )

        trace.emit(model_call_requested(state.run_id, step.step_id, prompt))

        # ModelGateway integration: call generate with Message abstraction
        temperature = step.payload.get("temperature", 0.1)
        max_tokens = step.payload.get("max_tokens")

        messages = [
            Message(role="system", content="You are a helpful research agent assisting with task execution."),
            Message(role="user", content=prompt),
        ]

        try:
            req = GenerationRequest(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = self.gateway.generate(req)
        except Exception as exc:
            raise ModelExecutionError(f"Model generation failed: {exc}") from exc

        # Usage information is preserved if available, but not mandatory
        usage_dict = {}
        if hasattr(response, "usage") and response.usage:
            usage_dict = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        metadata_dict = {
            "provider": getattr(response, "provider", "unknown"),
            "model": getattr(response, "model", "unknown"),
        }

        trace.emit(
            model_call_completed(
                state.run_id,
                step.step_id,
                response.text,
                usage=usage_dict,
                metadata=metadata_dict,
            )
        )

        return StepExecutionRecord(
            step_id=step.step_id,
            action_type=step.action_type.value,
            status="completed",
            output=response.text,
            metadata={
                "usage": usage_dict,
                "provider": metadata_dict["provider"],
                "model": metadata_dict["model"],
            },
        )

    def _execute_tool_step(
        self,
        step: AgentStep,
        state: AgentState,
        trace: ExecutionTrace,
        tracker: LimitTracker,
    ) -> StepExecutionRecord:
        tool_name = step.payload.get("tool_name")
        tool_input = step.payload.get("tool_input", {})

        if not tool_name or not isinstance(tool_name, str):
            return StepExecutionRecord(
                step_id=step.step_id,
                action_type=step.action_type.value,
                status="failed",
                error="Missing or invalid 'tool_name' in step payload.",
            )

        # Authoritative policy evaluation
        policy_res = self.policy.evaluate(tool_name, tool_input, self.registry, state)

        if policy_res.decision == PolicyDecision.UNKNOWN:
            raise ToolNotFoundError(policy_res.reason)

        if policy_res.decision == PolicyDecision.DENIED:
            raise ToolPolicyViolationError(policy_res.reason)

        # Permitted tool: increment tool calls against execution limit
        tracker.increment_tool_call()

        trace.emit(tool_call_requested(state.run_id, step.step_id, tool_name, tool_input))

        tool = self.registry.get(tool_name)
        tool_res = tool.execute(**tool_input)

        trace.emit(
            tool_call_completed(
                state.run_id,
                step.step_id,
                tool_name,
                tool_res.output,
                tool_res.is_error,
                tool_res.error_message,
            )
        )

        # Record tool result in state
        state.record_tool_result(tool_res.to_dict())

        return StepExecutionRecord(
            step_id=step.step_id,
            action_type=step.action_type.value,
            status="failed" if tool_res.is_error else "completed",
            output=tool_res.output,
            error=tool_res.error_message,
            metadata={"tool_name": tool_name},
        )

    def _execute_finish_step(self, step: AgentStep) -> StepExecutionRecord:
        final_answer = step.payload.get("final_answer", "")
        return StepExecutionRecord(
            step_id=step.step_id,
            action_type=step.action_type.value,
            status="completed",
            output=final_answer,
        )
