"""URL configuration for agent and asynchronous research endpoints (M9)."""

from django.urls import path

from agent.views import (
    ResearchCancelView,
    ResearchDetailView,
    ResearchHistoryView,
    ResearchView,
)

urlpatterns = [
    path("research/", ResearchView.as_view(), name="research-list-create"),
    path("research/runs/", ResearchHistoryView.as_view(), name="research-history"),
    path("research/<str:run_id>/", ResearchDetailView.as_view(), name="research-detail"),
    path("research/<str:run_id>/cancel/", ResearchCancelView.as_view(), name="research-cancel"),
]
