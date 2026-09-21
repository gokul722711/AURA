"""Health check URL configuration."""

from django.urls import path

from health.views import HealthCheckView

urlpatterns = [
    path("health/", HealthCheckView.as_view(), name="health-check"),
]
