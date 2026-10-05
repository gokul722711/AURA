"""
URL configuration for AURA project.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("health.urls")),
    path("api/", include("gateway.urls")),
    path("api/", include("agent.urls")),
    path("api/", include("rag.urls")),
]
