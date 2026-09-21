"""Tests for the health check endpoint."""

from django.test import TestCase
from rest_framework.test import APIClient


class HealthCheckTests(TestCase):
    """Tests for GET /api/health/."""

    def setUp(self):
        self.client = APIClient()

    def test_health_check_returns_200(self):
        """The health endpoint should return HTTP 200."""
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)

    def test_health_check_returns_expected_json(self):
        """The health endpoint should return the expected JSON payload."""
        response = self.client.get("/api/health/")
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "aura-backend")

    def test_health_check_content_type(self):
        """The health endpoint should return application/json."""
        response = self.client.get("/api/health/")
        self.assertEqual(response["Content-Type"], "application/json")
