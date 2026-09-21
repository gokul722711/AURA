"""Health check views."""

from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthCheckView(APIView):
    """Return a simple JSON response indicating the backend is running."""

    authentication_classes = []
    permission_classes = []

    def get(self, request: Request) -> Response:
        return Response(
            {
                "status": "ok",
                "service": "aura-backend",
            }
        )
