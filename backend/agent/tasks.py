"""Celery tasks for autonomous research execution (M9)."""

import logging
from typing import Any

from celery import shared_task
from django.utils import timezone

from agent.models import ResearchRun
from agent.research import create_research_runtime
from agent.security import sanitize_data, sanitize_text
from agent.state import AgentStatus

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="agent.tasks.execute_research_run")
def execute_research_run(self, run_id: str) -> dict[str, Any]:
    """Execute an autonomous research run asynchronously via Celery worker.

    1. Load ResearchRun by ID.
    2. Check if already cancelled.
    3. Transition status from queued to running.
    4. Invoke ResearchRuntime.run_research().
    5. Persist ResearchResult and transition to completed (or failed/cancelled).
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

    # 2. Transition queued -> running (atomic guard against concurrent cancellation)
    rows_updated = ResearchRun.objects.filter(
        id=run.id, status=ResearchRun.STATUS_QUEUED
    ).update(status=ResearchRun.STATUS_RUNNING, started_at=timezone.now())
    if rows_updated == 0:
        run.refresh_from_db(fields=["status"])
        if run.status == ResearchRun.STATUS_CANCELLED:
            logger.info("ResearchRun %s was cancelled before execution.", run_id)
            return {"status": "cancelled", "run_id": str(run.id)}
    run.refresh_from_db(fields=["status", "started_at"])

    logger.info("Starting autonomous research execution for run_id=%s", run_id)

    try:
        runtime = create_research_runtime()
        research_result = runtime.run_research(run.objective)

        # Check if cancelled during execution
        run.refresh_from_db(fields=["status"])
        if run.status == ResearchRun.STATUS_CANCELLED:
            logger.info("ResearchRun %s was cancelled during execution.", run_id)
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

        completed_time = timezone.now()
        duration_ms = result_dict.get("duration_ms") or 0.0

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
                return {"status": "cancelled", "run_id": str(run.id)}

        run.refresh_from_db()
        logger.info(
            "ResearchRun %s finished with status=%s in %.2fms",
            run_id,
            run.status,
            run.duration_ms,
        )
        return {"status": run.status, "run_id": str(run.id)}

    except Exception as exc:
        logger.error("ResearchRun %s execution failed: %s", run_id, exc)
        try:
            run.refresh_from_db(fields=["status"])
            if run.status == ResearchRun.STATUS_CANCELLED:
                logger.info("ResearchRun %s was cancelled during execution.", run_id)
                return {"status": "cancelled", "run_id": str(run.id)}
        except Exception:
            pass
        safe_error = sanitize_text(str(exc))
        completed_time = timezone.now()
        ResearchRun.objects.filter(
            id=run.id, status=ResearchRun.STATUS_RUNNING
        ).update(
            status=ResearchRun.STATUS_FAILED,
            error_message=safe_error,
            completed_at=completed_time,
        )
        run.refresh_from_db()
        return {"status": "failed", "run_id": str(run.id), "error": safe_error}
