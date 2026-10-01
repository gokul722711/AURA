"""API views for AURA agent and research."""

import logging
from typing import Any

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agent.research import ResearchRuntime, create_research_runtime
from agent.security import sanitize_data, sanitize_text

logger = logging.getLogger(__name__)


class ResearchView(APIView):
    """Execute autonomous research for a user-provided objective.

    Endpoint: POST /api/research/
    Request: {"objective": "..."}
    Response: ResearchResult serialized to JSON
    """

    authentication_classes = []
    permission_classes = []

    def get_runtime(self) -> ResearchRuntime:
        """Create or return the configured ResearchRuntime instance."""
        return create_research_runtime()

    def post(self, request: Request) -> Response:
        """Execute autonomous research synchronously."""
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

        try:
            runtime = self.get_runtime()
            research_result = runtime.run_research(trimmed_objective)
        except Exception as exc:
            logger.warning("Research execution failed: %s", sanitize_text(str(exc)))
            safe_detail = sanitize_text(str(exc))
            return Response(
                {
                    "error": "Research execution failed.",
                    "detail": safe_detail,
                    "status": "failed",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        result_dict = sanitize_data(research_result.to_dict())
        return Response(result_dict, status=status.HTTP_200_OK)
