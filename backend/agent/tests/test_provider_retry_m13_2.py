"""Comprehensive tests for M13.2 — Transient Model Provider Retry Resilience.

Validates:
1. transient 503 -> first task attempt schedules retry (countdown=5s)
2. transient 503 -> second attempt schedules retry (countdown=15s)
3. transient 503 -> final attempt fails ResearchRun
4. transient 503 -> next attempt succeeds -> ResearchRun completed
5. same ResearchRun ID is preserved across retries
6. non-transient authentication/configuration error is NOT retried
7. cancellation prevents retry execution
8. no duplicate ResearchRun records are created
9. existing successful execution behavior remains unchanged
10. classifier unit tests for transient vs non-transient exceptions
"""

from unittest.mock import MagicMock, patch
from celery.exceptions import Retry
from django.test import TestCase
from django.utils import timezone

from agent.models import ResearchRun
from agent.results import ResearchResult
from agent.state import AgentStatus
from agent.tasks import execute_research_run
from gateway.exceptions import (
    GenerationError,
    ProviderConfigurationError,
    TransientModelProviderError,
    is_transient_provider_error,
)


class TransientErrorClassifierTests(TestCase):
    """Unit tests for is_transient_provider_error classification."""

    def test_transient_status_codes(self) -> None:
        """HTTP 429, 500, 502, 503, 504 are classified as transient."""
        for code in [429, 500, 502, 503, 504]:
            exc = TransientModelProviderError(f"HTTP {code} error", status_code=code)
            self.assertTrue(is_transient_provider_error(exc), f"Code {code} should be transient")

    def test_non_transient_status_codes(self) -> None:
        """HTTP 400, 401, 403, 404, 422 are classified as non-transient."""
        for code in [400, 401, 403, 404, 422]:
            exc = GenerationError(f"HTTP {code} error")
            exc.status_code = code
            self.assertFalse(is_transient_provider_error(exc), f"Code {code} should not be transient")

    def test_transient_exception_types_and_messages(self) -> None:
        """Connection resets, timeouts, and overload messages are transient."""
        self.assertTrue(is_transient_provider_error(ConnectionResetError("Connection reset by peer")))
        self.assertTrue(is_transient_provider_error(TimeoutError("Read timed out")))
        self.assertTrue(
            is_transient_provider_error(
                GenerationError("NVIDIA model generation failed: Error code: 503 - {'error': {'message': 'Service temporarily overloaded'}}")
            )
        )
        self.assertTrue(
            is_transient_provider_error("Rate limit reached for model nvidia/nemotron-3-ultra-550b-a55b")
        )
        self.assertTrue(
            is_transient_provider_error("502 Bad Gateway: upstream server unavailable")
        )

    def test_non_transient_exception_types_and_messages(self) -> None:
        """Auth, bad request, and programming errors are not transient."""
        self.assertFalse(
            is_transient_provider_error(ProviderConfigurationError("NVIDIA API key not configured"))
        )
        self.assertFalse(
            is_transient_provider_error(
                GenerationError("NVIDIA model generation failed: Error code: 401 - Invalid API key")
            )
        )
        self.assertFalse(
            is_transient_provider_error(
                GenerationError("Error code: 400 - Invalid request format")
            )
        )
        self.assertFalse(is_transient_provider_error(ValueError("Invalid argument")))
        self.assertFalse(is_transient_provider_error(KeyError("missing_key")))
        self.assertFalse(is_transient_provider_error(None))


class CeleryModelProviderRetryTests(TestCase):
    """Integration and unit tests for Celery-level transient model provider retries (M13.2)."""

    def setUp(self) -> None:
        self.run = ResearchRun.objects.create(
            objective="What is Claude, and what organization developed it?",
            mode=ResearchRun.MODE_KNOWLEDGE_BASE,
            status=ResearchRun.STATUS_QUEUED,
        )

    def test_transient_503_first_attempt_schedules_retry(self) -> None:
        """Transient 503 on attempt 0 schedules retry 1/2 with initial backoff countdown."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError(
            "NVIDIA transient provider failure: Error code: 503", status_code=503
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                # Simulate Celery's retry raising Retry exception
                mock_retry.side_effect = Retry("Task rescheduled")

                with self.assertRaises(Retry):
                    execute_research_run(str(self.run.id))

                # Verify retry scheduled with exponential backoff base delay (5s) and max_retries=2
                mock_retry.assert_called_once()
                call_kwargs = mock_retry.call_args[1]
                self.assertEqual(call_kwargs["countdown"], 5)
                self.assertEqual(call_kwargs["max_retries"], 2)

        # ResearchRun record MUST remain in running state across retries (NOT failed)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_RUNNING)
        self.assertIsNone(self.run.completed_at)
        self.assertEqual(self.run.error_message, "")

    def test_transient_503_second_attempt_schedules_retry(self) -> None:
        """Transient 503 on attempt 1 schedules retry 2/2 with increased backoff countdown."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError(
            "NVIDIA transient provider failure: Error code: 503", status_code=503
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                mock_retry.side_effect = Retry("Task rescheduled")

                # Simulate Celery task running as retry #1
                with patch.object(execute_research_run.request, "retries", 1):
                    with self.assertRaises(Retry):
                        execute_research_run(str(self.run.id))

                    mock_retry.assert_called_once()
                    call_kwargs = mock_retry.call_args[1]
                    # Backoff factor 3.0: 5 * (3 ** 1) = 15s
                    self.assertEqual(call_kwargs["countdown"], 15)
                    self.assertEqual(call_kwargs["max_retries"], 2)

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_RUNNING)

    def test_transient_503_final_attempt_fails_research_run(self) -> None:
        """When retries are exhausted (attempt count reached max_retries=2), ResearchRun fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError(
            "NVIDIA transient provider failure: Error code: 503 - Service temporarily overloaded",
            status_code=503,
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                # Simulate Celery task running on attempt 2 (max_retries reached)
                with patch.object(execute_research_run.request, "retries", 2):
                    res = execute_research_run(str(self.run.id))
                    # retry must NOT be called when retries are exhausted
                    mock_retry.assert_not_called()
                    self.assertEqual(res["status"], "failed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_FAILED)
        self.assertIn("Service temporarily overloaded", self.run.error_message)
        self.assertIsNotNone(self.run.completed_at)
        self.assertIsNotNone(self.run.duration_ms)

    def test_transient_503_next_attempt_succeeds_research_run_completed(self) -> None:
        """Subsequent retry attempt succeeds and updates the ResearchRun to completed."""
        # Simulate attempt 1 (retry) succeeding with valid result
        successful_result = ResearchResult(
            objective=self.run.objective,
            final_answer="Claude is an AI developed by Anthropic [Source: Claude, Chunk: 123].",
            evidence=[],
            sources=[],
            queries=[self.run.objective],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=1250.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = successful_result

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run.request, "retries", 1):
                res = execute_research_run(str(self.run.id))
                self.assertEqual(res["status"], "completed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_COMPLETED)
        self.assertIn("Claude is an AI developed by Anthropic", self.run.result.get("final_answer", ""))
        self.assertEqual(self.run.duration_ms, 1250.0)
        self.assertIsNotNone(self.run.completed_at)
        self.assertEqual(self.run.error_message, "")

    def test_same_research_run_id_preserved_across_retries(self) -> None:
        """The exact same ResearchRun record is updated across attempts; no new record is created."""
        orig_id = self.run.id
        initial_count = ResearchRun.objects.count()

        # 1. Attempt 0 fails with 503
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Service Unavailable", status_code=503)

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                mock_retry.side_effect = Retry("Rescheduled")
                with self.assertRaises(Retry):
                    execute_research_run(str(orig_id))

        self.assertEqual(ResearchRun.objects.count(), initial_count)
        self.run.refresh_from_db()
        self.assertEqual(self.run.id, orig_id)

        # 2. Attempt 1 succeeds
        mock_pipeline.run.side_effect = None
        mock_pipeline.run.return_value = ResearchResult(
            objective=self.run.objective,
            final_answer="Answer after retry.",
            evidence=[],
            sources=[],
            queries=[self.run.objective],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=800.0,
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run.request, "retries", 1):
                res = execute_research_run(str(orig_id))
                self.assertEqual(res["status"], "completed")

        self.assertEqual(ResearchRun.objects.count(), initial_count)
        self.run.refresh_from_db()
        self.assertEqual(self.run.id, orig_id)
        self.assertEqual(self.run.status, ResearchRun.STATUS_COMPLETED)

    def test_non_transient_authentication_error_is_not_retried(self) -> None:
        """Non-transient errors (e.g. HTTP 401 or ProviderConfigurationError) fail immediately without retrying."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = GenerationError(
            "NVIDIA model generation failed: Error code: 401 - {'error': {'message': 'Invalid API Key'}}"
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                res = execute_research_run(str(self.run.id))
                # Must NOT schedule a retry
                mock_retry.assert_not_called()
                self.assertEqual(res["status"], "failed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_FAILED)
        self.assertIn("Invalid API Key", self.run.error_message)

    def test_cancellation_prevents_retry_execution(self) -> None:
        """If user cancels ResearchRun while retry is pending, subsequent task execution aborts immediately."""
        # Mark run as cancelled in DB while retry is pending
        self.run.status = ResearchRun.STATUS_CANCELLED
        self.run.error_message = "Cancelled by user."
        self.run.completed_at = timezone.now()
        self.run.save()

        mock_pipeline = MagicMock()

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run.request, "retries", 1):
                res = execute_research_run(str(self.run.id))
                self.assertEqual(res["status"], "cancelled")
                # Pipeline must NOT be called
                mock_pipeline.run.assert_not_called()

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_CANCELLED)

    def test_no_duplicate_research_run_records_are_created(self) -> None:
        """Multi-attempt execution preserves exact single database record invariant."""
        initial_count = ResearchRun.objects.count()

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Overloaded", status_code=503)

        # Attempt 0
        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry", side_effect=Retry("Retrying")):
                with self.assertRaises(Retry):
                    execute_research_run(str(self.run.id))

        self.assertEqual(ResearchRun.objects.count(), initial_count)
        self.assertEqual(ResearchRun.objects.filter(id=self.run.id).count(), 1)

        # Attempt 1
        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry", side_effect=Retry("Retrying")):
                with patch.object(execute_research_run.request, "retries", 1):
                    with self.assertRaises(Retry):
                        execute_research_run(str(self.run.id))

        self.assertEqual(ResearchRun.objects.count(), initial_count)
        self.assertEqual(ResearchRun.objects.filter(id=self.run.id).count(), 1)

    def test_existing_successful_execution_behavior_remains_unchanged(self) -> None:
        """Clean execution with no provider errors finishes on attempt 0 without retrying."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = ResearchResult(
            objective=self.run.objective,
            final_answer="Direct success answer.",
            evidence=[],
            sources=[],
            queries=[self.run.objective],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=450.0,
        )

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
            with patch.object(execute_research_run, "retry") as mock_retry:
                res = execute_research_run(str(self.run.id))
                mock_retry.assert_not_called()
                self.assertEqual(res["status"], "completed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_COMPLETED)
        self.assertEqual(self.run.duration_ms, 450.0)
        self.assertEqual(self.run.error_message, "")

    def test_configuration_controls_max_retries(self) -> None:
        """Configuring AI_GATEWAY.MAX_RETRIES=1 limits retries to exactly 1 attempt."""
        from django.conf import settings
        from django.test import override_settings

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Service Unavailable", status_code=503)

        new_gw = dict(settings.AI_GATEWAY)
        new_gw["MAX_RETRIES"] = 1

        with override_settings(AI_GATEWAY=new_gw):
            with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
                # Attempt 0: Should schedule retry (current_retries=0 < max_retries=1)
                with patch.object(execute_research_run, "retry", side_effect=Retry("Retry 1")) as mock_retry:
                    with self.assertRaises(Retry):
                        execute_research_run(str(self.run.id))
                    mock_retry.assert_called_once()
                    self.assertEqual(mock_retry.call_args[1]["max_retries"], 1)

                # Attempt 1: Should exhaust retries and fail run (current_retries=1 == max_retries=1)
                with patch.object(execute_research_run, "retry") as mock_retry:
                    with patch.object(execute_research_run.request, "retries", 1):
                        res = execute_research_run(str(self.run.id))
                        mock_retry.assert_not_called()
                        self.assertEqual(res["status"], "failed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_FAILED)

    def test_configuration_controls_backoff_base_and_factor(self) -> None:
        """Configuring AI_GATEWAY RETRY_BACKOFF_BASE and RETRY_BACKOFF_FACTOR controls countdown timing."""
        from django.conf import settings
        from django.test import override_settings

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Service Unavailable", status_code=503)

        new_gw = dict(settings.AI_GATEWAY)
        new_gw["RETRY_BACKOFF_BASE"] = 8.0
        new_gw["RETRY_BACKOFF_FACTOR"] = 2.0

        with override_settings(AI_GATEWAY=new_gw):
            with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
                # Attempt 0: 8 * (2 ** 0) = 8s
                with patch.object(execute_research_run, "retry", side_effect=Retry("Retry 1")) as mock_retry:
                    with self.assertRaises(Retry):
                        execute_research_run(str(self.run.id))
                    self.assertEqual(mock_retry.call_args[1]["countdown"], 8)

                # Attempt 1: 8 * (2 ** 1) = 16s
                with patch.object(execute_research_run, "retry", side_effect=Retry("Retry 2")) as mock_retry:
                    with patch.object(execute_research_run.request, "retries", 1):
                        with self.assertRaises(Retry):
                            execute_research_run(str(self.run.id))
                    self.assertEqual(mock_retry.call_args[1]["countdown"], 16)

    def test_configuration_max_retries_zero_disables_retries(self) -> None:
        """Setting AI_GATEWAY.MAX_RETRIES=0 causes immediate failure with zero retries scheduled."""
        from django.conf import settings
        from django.test import override_settings

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Service Unavailable", status_code=503)

        new_gw = dict(settings.AI_GATEWAY)
        new_gw["MAX_RETRIES"] = 0

        with override_settings(AI_GATEWAY=new_gw):
            with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
                with patch.object(execute_research_run, "retry") as mock_retry:
                    res = execute_research_run(str(self.run.id))
                    mock_retry.assert_not_called()
                    self.assertEqual(res["status"], "failed")

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ResearchRun.STATUS_FAILED)

    def test_task_max_retries_attribute_dynamically_synchronized(self) -> None:
        """execute_research_run task instance synchronizes self.max_retries to match active configuration."""
        from django.conf import settings
        from django.test import override_settings

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = TransientModelProviderError("503 Service Unavailable", status_code=503)

        new_gw = dict(settings.AI_GATEWAY)
        new_gw["MAX_RETRIES"] = 4

        with override_settings(AI_GATEWAY=new_gw):
            with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline):
                with patch.object(execute_research_run, "retry", side_effect=Retry("Rescheduled")):
                    with self.assertRaises(Retry):
                        execute_research_run(str(self.run.id))

                # Verify task instance self.max_retries was dynamically updated
                self.assertEqual(execute_research_run.max_retries, 4)

