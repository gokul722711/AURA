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

        # Validate and resolve ModelProfile (M15)
        profile = None
        requested_profile_id = data.get("model_profile_id")
        if requested_profile_id:
            profile_uuid = _parse_run_uuid(requested_profile_id)
            if profile_uuid is None:
                return Response(
                    {"error": f"Invalid model_profile_id format: '{requested_profile_id}'."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            from gateway.models import ModelProfile

            profile = ModelProfile.objects.filter(id=profile_uuid).first()
            if profile is None:
                return Response(
                    {"error": f"Model profile '{requested_profile_id}' not found."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            from gateway.models import get_active_model_profile

            profile = get_active_model_profile()

        if profile is None:
            return Response(
                {
                    "error": "No research model configured. Configure a model profile before starting research.",
                    "code": "NO_MODEL_CONFIGURED",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1. Create persistent ResearchRun
        run = ResearchRun.objects.create(
            objective=trimmed_objective,
            mode=mode,
            status=ResearchRun.STATUS_QUEUED,
            model_profile=profile,
            model_name=profile.model,
            provider_name=profile.provider,
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
            "model_profile_id": str(profile.id),
            "model_name": profile.model,
            "provider_name": profile.provider,
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

    def delete(self, request: Request, run_id: str) -> Response:
        """Delete an individual research run."""
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

        run.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


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

    def delete(self, request: Request) -> Response:
        """Delete multiple or all historical research runs."""
        run_ids = request.data.get("run_ids") if isinstance(request.data, dict) else None
        if run_ids is not None:
            if not isinstance(run_ids, list):
                return Response(
                    {"error": "Field 'run_ids' must be a list of UUID strings."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            valid_ids = []
            for rid in run_ids:
                pid = _parse_run_uuid(str(rid))
                if pid is not None:
                    valid_ids.append(pid)
            deleted_count, _ = ResearchRun.objects.filter(id__in=valid_ids).delete()
            return Response({"deleted_count": deleted_count}, status=status.HTTP_200_OK)

        deleted_count, _ = ResearchRun.objects.all().delete()
        return Response({"deleted_count": deleted_count}, status=status.HTTP_200_OK)


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

        now = timezone.now()
        duration_ms = None
        if run.started_at:
            duration_ms = max(0.0, round((now - run.started_at).total_seconds() * 1000.0, 2))
        elif run.created_at:
            duration_ms = max(0.0, round((now - run.created_at).total_seconds() * 1000.0, 2))

        ResearchRun.objects.filter(
            id=run.id,
            status__in=[ResearchRun.STATUS_QUEUED, ResearchRun.STATUS_RUNNING],
        ).update(
            status=ResearchRun.STATUS_CANCELLED,
            completed_at=now,
            duration_ms=duration_ms,
            error_message="Cancelled by user.",
        )
        run.refresh_from_db()

        return Response(run.to_detail_dict(), status=status.HTTP_200_OK)
