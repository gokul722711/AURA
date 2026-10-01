"""Tests for ExecutionLimits and LimitTracker."""

import time
from django.test import SimpleTestCase

from agent.exceptions import (
    IterationLimitExceededError,
    TimeoutLimitExceededError,
    ToolCallLimitExceededError,
)
from agent.execution.limits import ExecutionLimits, LimitTracker


class ExecutionLimitsTests(SimpleTestCase):
    """Tests for limit validation and tracking."""

    def test_default_limits(self):
        limits = ExecutionLimits()
        self.assertEqual(limits.max_iterations, 10)
        self.assertEqual(limits.max_tool_calls, 15)
        self.assertEqual(limits.max_time_seconds, 60.0)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            ExecutionLimits(max_iterations=0)

        with self.assertRaises(ValueError):
            ExecutionLimits(max_tool_calls=-1)

        with self.assertRaises(ValueError):
            ExecutionLimits(max_time_seconds=0.0)

    def test_iteration_limit_exceeded(self):
        tracker = LimitTracker(ExecutionLimits(max_iterations=2))
        tracker.increment_iteration()  # 1
        tracker.increment_iteration()  # 2
        with self.assertRaises(IterationLimitExceededError):
            tracker.increment_iteration()  # 3 -> raises

    def test_tool_call_limit_exceeded(self):
        tracker = LimitTracker(ExecutionLimits(max_tool_calls=1))
        tracker.increment_tool_call()  # 1
        with self.assertRaises(ToolCallLimitExceededError):
            tracker.increment_tool_call()  # 2 -> raises

    def test_timeout_limit_exceeded(self):
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=0.01))
        time.sleep(0.02)
        with self.assertRaises(TimeoutLimitExceededError):
            tracker.check_time()
