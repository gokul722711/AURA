"""URL configuration for RAG document endpoints."""

from django.urls import path

from rag.views import DocumentDetailView, DocumentListCreateView

urlpatterns = [
    path("documents/", DocumentListCreateView.as_view(), name="document-list-create"),
    path("documents/<str:document_id>/", DocumentDetailView.as_view(), name="document-detail"),
]
