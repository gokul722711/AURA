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

    def test_remaining_seconds_initial(self):
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=10.0))
        rem = tracker.remaining_seconds()
        self.assertGreater(rem, 9.9)
        self.assertLessEqual(rem, 10.0)

    def test_remaining_seconds_decreases_as_execution_progresses(self):
        tracker = LimitTracker(ExecutionLimits(max_time_seconds=10.0))
        rem1 = tracker.remaining_seconds()
        time.sleep(0.01)
        rem2 = tracker.remaining_seconds()
        self.assertLess(rem2, rem1)

    def test_remaining_seconds_zero_when_budget_exhausted(self):
        from unittest.mock import patch

        tracker = LimitTracker(ExecutionLimits(max_time_seconds=5.0))
        with patch("time.monotonic", return_value=tracker.start_time + 10.0):
            self.assertEqual(tracker.remaining_seconds(), 0.0)

    def test_remaining_seconds_monotonic_mocked(self):
        from unittest.mock import patch

        tracker = LimitTracker(ExecutionLimits(max_time_seconds=60.0))
        # After 20s elapsed -> 40s remaining
        with patch("time.monotonic", return_value=tracker.start_time + 20.0):
            self.assertAlmostEqual(tracker.remaining_seconds(), 40.0, places=2)
        # After 59s elapsed -> 1s remaining
        with patch("time.monotonic", return_value=tracker.start_time + 59.0):
            self.assertAlmostEqual(tracker.remaining_seconds(), 1.0, places=2)
        # After 65s elapsed -> 0.0 remaining (non-positive clamped to 0.0)
        with patch("time.monotonic", return_value=tracker.start_time + 65.0):
            self.assertEqual(tracker.remaining_seconds(), 0.0)
