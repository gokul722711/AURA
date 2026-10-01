"""URL configuration for agent and research endpoints."""

from django.urls import path

from agent.views import ResearchView

urlpatterns = [
    path("research/", ResearchView.as_view(), name="research"),
]
