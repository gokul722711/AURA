"""API views for AURA agent and asynchronous research (M9)."""

import logging
import uuid
from typing import Any

from django.utils import timezone
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agent.models import ResearchRun
from agent.security import sanitize_data, sanitize_text
from agent.tasks import execute_research_run

logger = logging.getLogger(__name__)


def _parse_run_uuid(run_id: str) -> uuid.UUID | None:
    """Parse string run_id into a valid UUID, returning None if invalid."""
    try:
        return uuid.UUID(str(run_id))
    except (ValueError, AttributeError):
        return None


class ResearchView(APIView):
    """Queue a new autonomous research run.

    Endpoint: POST /api/research/
    Request:  {"objective": "..."}
    Response: 202 Accepted {"run_id": "...", "status": "queued", "objective": "..."}
    """

    authentication_classes = []
    permission_classes = []

    def dispatch_task(self, run_id: str) -> Any:
        """Enqueue the Celery task for the research run."""
        return execute_research_run.delay(run_id)

    def post(self, request: Request) -> Response:
        """Validate research objective, persist ResearchRun, and queue task."""
        data = request.data
        if not isinstance(data, dict):
            return Response(
                {"error": "Invalid request body; JSON object expected."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if "objective" not in data:
            return Response(
                {"error": "Missing required field: 'objective'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        objective = data["objective"]
        if not isinstance(objective, str):
            return Response(
                {"error": "Field 'objective' must be a string."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        trimmed_objective = objective.strip()
        if not trimmed_objective:
            return Response(
                {"error": "Field 'objective' cannot be empty or whitespace-only."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate mode (default to knowledge_base if omitted)
        mode = data.get("mode", ResearchRun.MODE_KNOWLEDGE_BASE)
        if not isinstance(mode, str):
            return Response(
                {"error": "Field 'mode' must be a string."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        mode = mode.strip().lower()
        valid_modes = [c[0] for c in ResearchRun.MODE_CHOICES]
        if mode not in valid_modes:
            return Response(
                {
                    "error": f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}.",
                    "valid_modes": valid_modes,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1. Create persistent ResearchRun
        run = ResearchRun.objects.create(
            objective=trimmed_objective,
            mode=mode,
            status=ResearchRun.STATUS_QUEUED,
        )

        # 2. Enqueue Celery task
        try:
            self.dispatch_task(str(run.id))
        except Exception as exc:
            logger.warning(
                "Failed to queue research task for run_id=%s: %s",
                run.id,
                sanitize_text(str(exc)),
            )
            run.status = ResearchRun.STATUS_FAILED
            run.error_message = "Failed to queue research task. Please verify Celery/Redis is running."
            run.completed_at = timezone.now()
            run.save(update_fields=["status", "error_message", "completed_at"])
            return Response(
                {
                    "error": "Failed to queue research task.",
                    "detail": "Could not dispatch task to broker (Redis).",
                    "run_id": str(run.id),
                    "status": "failed",
                    "mode": run.mode,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        response_data = {
            "run_id": str(run.id),
            "status": run.status,
            "mode": run.mode,
            "objective": run.objective,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        return Response(sanitize_data(response_data), status=status.HTTP_202_ACCEPTED)


class ResearchDetailView(APIView):
    """Retrieve the current status and result of a research run.

    Endpoint: GET /api/research/<run_id>/
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request: Request, run_id: str) -> Response:
        """Return the current persisted status or completed M6 result."""
        parsed_id = _parse_run_uuid(run_id)
        if parsed_id is None:
            return Response(
                {"error": "Research run not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            run = ResearchRun.objects.get(id=parsed_id)
        except ResearchRun.DoesNotExist:
            return Response(
                {"error": "Research run not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(run.to_detail_dict(), status=status.HTTP_200_OK)


class ResearchHistoryView(APIView):
    """List recent research runs ordered newest first.

    Endpoint: GET /api/research/runs/
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request: Request) -> Response:
        """Return lightweight metadata for historical research runs."""
        runs = ResearchRun.objects.order_by("-created_at")[:100]
        data = [run.to_summary_dict() for run in runs]
        return Response(data, status=status.HTTP_200_OK)


class ResearchCancelView(APIView):
    """Cancel a queued or running research run.

    Endpoint: POST /api/research/<run_id>/cancel/
    """

    authentication_classes = []
    permission_classes = []

    def post(self, request: Request, run_id: str) -> Response:
        """Mark the research run as cancelled."""
        parsed_id = _parse_run_uuid(run_id)
        if parsed_id is None:
            return Response(
                {"error": "Research run not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            run = ResearchRun.objects.get(id=parsed_id)
        except ResearchRun.DoesNotExist:
            return Response(
                {"error": "Research run not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        ResearchRun.objects.filter(
            id=run.id,
            status__in=[ResearchRun.STATUS_QUEUED, ResearchRun.STATUS_RUNNING],
        ).update(
            status=ResearchRun.STATUS_CANCELLED,
            completed_at=timezone.now(),
            error_message="Cancelled by user.",
        )
        run.refresh_from_db()

        return Response(run.to_detail_dict(), status=status.HTTP_200_OK)
