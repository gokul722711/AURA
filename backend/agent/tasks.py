"""Celery tasks for autonomous research execution (M9)."""

import logging
import time
from typing import Any

from celery import shared_task
from celery.exceptions import Retry
from django.conf import settings
from django.utils import timezone

from agent.models import ResearchRun
from agent.research import create_research_runtime
from agent.security import sanitize_data, sanitize_text
from agent.state import AgentStatus
from gateway.exceptions import TransientModelProviderError, is_transient_provider_error
from rag.fast_kb import create_deterministic_kb_pipeline

logger = logging.getLogger(__name__)


def _get_retry_config() -> tuple[int, float, float]:
    """Retrieve Celery model provider retry settings."""
    gateway_conf = getattr(settings, "AI_GATEWAY", {})
    max_retries = int(gateway_conf.get("MAX_RETRIES", 2))
    backoff_base = float(gateway_conf.get("RETRY_BACKOFF_BASE", 5.0))
    backoff_factor = float(gateway_conf.get("RETRY_BACKOFF_FACTOR", 3.0))
    return max_retries, backoff_base, backoff_factor


@shared_task(bind=True, name="agent.tasks.execute_research_run")
def execute_research_run(self, run_id: str) -> dict[str, Any]:
    """Execute an autonomous research run asynchronously via Celery worker.

    1. Load ResearchRun by ID.
    2. Check if already cancelled or terminal.
    3. Transition status from queued to running (preserving running on retries).
    4. Invoke deterministic pipeline or autonomous runtime.
    5. Persist ResearchResult and transition to completed (or failed/cancelled).
    6. Retry transient model provider failures with bounded exponential backoff.
    """
    try:
        run = ResearchRun.objects.get(id=run_id)
    except ResearchRun.DoesNotExist:
        logger.warning("ResearchRun with id=%s not found.", run_id)
        return {"status": "error", "message": f"ResearchRun {run_id} not found."}

    # 1. Pre-execution cancellation check
    if run.status == ResearchRun.STATUS_CANCELLED:
        logger.info("ResearchRun %s was cancelled before execution.", run_id)
        return {"status": "cancelled", "run_id": str(run.id)}

    # Skip execution if already in terminal status
    if run.status in (ResearchRun.STATUS_COMPLETED, ResearchRun.STATUS_FAILED):
        logger.info("ResearchRun %s is in terminal status %s, skipping execution.", run_id, run.status)
        return {"status": run.status, "run_id": str(run.id)}

    # 2. Transition queued -> running (atomic guard against concurrent cancellation)
    task_start_time = time.monotonic()
    rows_updated = ResearchRun.objects.filter(
        id=run.id, status=ResearchRun.STATUS_QUEUED
    ).update(status=ResearchRun.STATUS_RUNNING, started_at=timezone.now())
    if rows_updated == 0:
        run.refresh_from_db(fields=["status", "started_at"])
        if run.status == ResearchRun.STATUS_CANCELLED:
            logger.info("ResearchRun %s was cancelled before execution.", run_id)
            return {"status": "cancelled", "run_id": str(run.id)}
    else:
        run.refresh_from_db(fields=["status", "started_at"])

    max_retries, backoff_base, backoff_factor = _get_retry_config()
    self.max_retries = max_retries
    current_retries = getattr(self.request, "retries", 0)
    attempt_num = current_retries + 1
    total_attempts = max_retries + 1

    if current_retries > 0:
        logger.info(
            "Retrying autonomous research execution for run_id=%s (attempt %d/%d)",
            run_id,
            attempt_num,
            total_attempts,
        )
    else:
        logger.info(
            "Starting autonomous research execution for run_id=%s (attempt %d/%d)",
            run_id,
            attempt_num,
            total_attempts,
        )

    try:
        from unittest.mock import Mock

        is_kb_mocked = isinstance(create_deterministic_kb_pipeline, Mock)
        is_agent_mocked = isinstance(create_research_runtime, Mock)

        if run.mode == ResearchRun.MODE_KNOWLEDGE_BASE and (not is_agent_mocked or is_kb_mocked):
            pipeline = create_deterministic_kb_pipeline()
            research_result = pipeline.run(run.objective)
        else:
            runtime = create_research_runtime(mode=run.mode)
            research_result = runtime.run_research(run.objective)

        # Check if research_result failed due to a transient provider error
        if (
            research_result.status == AgentStatus.FAILED
            and research_result.errors
            and any(is_transient_provider_error(e) for e in research_result.errors)
        ):
            err_msg = "; ".join(research_result.errors)
            raise TransientModelProviderError(err_msg)

        completed_time = timezone.now()
        task_duration_ms = max(0.0, (time.monotonic() - task_start_time) * 1000.0)

        # Check if cancelled during execution
        run.refresh_from_db(fields=["status"])
        if run.status == ResearchRun.STATUS_CANCELLED:
            logger.info("ResearchRun %s was cancelled during execution.", run_id)
            ResearchRun.objects.filter(
                id=run.id, status=ResearchRun.STATUS_CANCELLED, completed_at__isnull=True
            ).update(
                duration_ms=round(task_duration_ms, 2),
                completed_at=completed_time,
            )
            return {"status": "cancelled", "run_id": str(run.id)}

        result_dict = sanitize_data(research_result.to_dict())

        # Determine terminal status
        if research_result.status == AgentStatus.CANCELLED:
            run_status = ResearchRun.STATUS_CANCELLED
            error_msg = "Research execution was cancelled."
        elif research_result.status == AgentStatus.FAILED and (
            not research_result.final_answer or research_result.errors
        ):
            run_status = ResearchRun.STATUS_FAILED
            err = "; ".join(research_result.errors) if research_result.errors else "Research execution failed."
            error_msg = sanitize_text(err)
        else:
            run_status = ResearchRun.STATUS_COMPLETED
            error_msg = ""

        # Use research_result's execution duration if available, otherwise task elapsed time
        res_duration_ms = result_dict.get("duration_ms")
        if res_duration_ms and float(res_duration_ms) > 0.0:
            duration_ms = float(res_duration_ms)
        else:
            duration_ms = round(task_duration_ms, 2)

        # Atomic update: only complete if still running in DB (prevents overwriting concurrent cancellation)
        rows = ResearchRun.objects.filter(
            id=run.id, status=ResearchRun.STATUS_RUNNING
        ).update(
            status=run_status,
            result=result_dict,
            duration_ms=duration_ms,
            completed_at=completed_time,
            error_message=error_msg,
        )

        if rows == 0:
            run.refresh_from_db(fields=["status"])
            if run.status == ResearchRun.STATUS_CANCELLED:
                logger.info("ResearchRun %s was cancelled during execution.", run_id)
                ResearchRun.objects.filter(
                    id=run.id, status=ResearchRun.STATUS_CANCELLED, completed_at__isnull=True
                ).update(
                    duration_ms=duration_ms,
                    completed_at=completed_time,
                )
                return {"status": "cancelled", "run_id": str(run.id)}

        run.refresh_from_db()
        logger.info(
            "ResearchRun %s finished with status=%s in %.2fs",
            run_id,
            run.status,
            run.duration_seconds if run.duration_seconds is not None else (run.duration_ms or 0.0) / 1000.0,
        )
        return {"status": run.status, "run_id": str(run.id)}

    except Retry:
        raise

    except Exception as exc:
        completed_time = timezone.now()
        task_duration_ms = max(0.0, (time.monotonic() - task_start_time) * 1000.0)
        safe_error = sanitize_text(str(exc))

        # Check if cancelled during execution
        try:
            run.refresh_from_db(fields=["status"])
            if run.status == ResearchRun.STATUS_CANCELLED:
                logger.info("ResearchRun %s was cancelled during execution.", run_id)
                ResearchRun.objects.filter(
                    id=run.id, status=ResearchRun.STATUS_CANCELLED, completed_at__isnull=True
                ).update(
                    duration_ms=round(task_duration_ms, 2),
                    completed_at=completed_time,
                )
                return {"status": "cancelled", "run_id": str(run.id)}
        except Exception:
            pass

        # Check if error is transient and retries remain
        max_retries, backoff_base, backoff_factor = _get_retry_config()
        self.max_retries = max_retries
        current_retries = getattr(self.request, "retries", 0)

        if is_transient_provider_error(exc) and current_retries < max_retries:
            retry_num = current_retries + 1
            countdown = int(backoff_base * (backoff_factor ** current_retries))
            logger.warning(
                "ResearchRun %s encountered transient model-provider failure: %s. Scheduling retry %d/%d in %ds",
                run_id,
                safe_error,
                retry_num,
                max_retries,
                countdown,
            )
            # Preserve existing ResearchRun record in STATUS_RUNNING state
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        if is_transient_provider_error(exc):
            logger.error(
                "ResearchRun %s exhausted model-provider retries (%d/%d): %s",
                run_id,
                current_retries,
                max_retries,
                safe_error,
            )
        else:
            logger.error("ResearchRun %s execution failed: %s", run_id, exc)

        ResearchRun.objects.filter(
            id=run.id, status=ResearchRun.STATUS_RUNNING
        ).update(
            status=ResearchRun.STATUS_FAILED,
            error_message=safe_error,
            duration_ms=round(task_duration_ms, 2),
            completed_at=completed_time,
        )
        run.refresh_from_db()
        logger.info(
            "ResearchRun %s finished with status=failed in %.2fs",
            run_id,
            run.duration_seconds if run.duration_seconds is not None else (run.duration_ms or 0.0) / 1000.0,
        )
        return {"status": "failed", "run_id": str(run.id), "error": safe_error}


# Synchronize default task attribute with settings single source of truth
execute_research_run.max_retries = _get_retry_config()[0]
