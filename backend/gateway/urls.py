"""URL configuration for AURA Model Gateway endpoints (M15)."""

from django.urls import path

from gateway.views import (
    ModelProfileActivateView,
    ModelProfileDetailView,
    ModelProfileListCreateView,
    ModelProfileTestConnectionView,
)

urlpatterns = [
    path("models/", ModelProfileListCreateView.as_view(), name="model-profile-list-create"),
    path("models/test/", ModelProfileTestConnectionView.as_view(), name="model-profile-test-draft"),
    path("models/<str:profile_id>/", ModelProfileDetailView.as_view(), name="model-profile-detail"),
    path("models/<str:profile_id>/activate/", ModelProfileActivateView.as_view(), name="model-profile-activate"),
    path("models/<str:profile_id>/test/", ModelProfileTestConnectionView.as_view(), name="model-profile-test"),
]
