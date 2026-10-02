"""Django models for AURA autonomous research run persistence (M9)."""

import uuid
from typing import Any

from django.db import models

from agent.security import sanitize_data


class ResearchRun(models.Model):
    """Persistent record of an autonomous research run."""

    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    MODE_MODEL_KNOWLEDGE = "model_knowledge"
    MODE_KNOWLEDGE_BASE = "knowledge_base"
    MODE_WEB = "web"
    MODE_WEB_KNOWLEDGE_BASE = "web_knowledge_base"

    MODE_CHOICES = [
        (MODE_MODEL_KNOWLEDGE, "Model Knowledge"),
        (MODE_KNOWLEDGE_BASE, "Knowledge Base"),
        (MODE_WEB, "Web"),
        (MODE_WEB_KNOWLEDGE_BASE, "Web + Knowledge Base"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    objective = models.TextField(help_text="User-provided research objective.")
    mode = models.CharField(
        max_length=32,
        choices=MODE_CHOICES,
        default=MODE_KNOWLEDGE_BASE,
        db_index=True,
        help_text="Information source mode for this research run.",
    )
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_QUEUED,
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.FloatField(
        null=True,
        blank=True,
        help_text="Total execution duration in milliseconds.",
    )
    result = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured result dictionary from ResearchResult.to_dict().",
    )
    error_message = models.TextField(
        blank=True,
        default="",
        help_text="Sanitized error description if status is failed or cancelled.",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["mode", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"ResearchRun({self.id}, mode={self.mode}, status={self.status})"

    def to_summary_dict(self) -> dict[str, Any]:
        """Return lightweight metadata for research history listing."""
        is_grounded = False
        has_evidence = False
        citation_count = 0

        if isinstance(self.result, dict):
            is_grounded = bool(self.result.get("is_grounded", False))
            has_evidence = bool(self.result.get("has_evidence", False))
            citations = self.result.get("citations", [])
            citation_count = len(citations) if isinstance(citations, list) else 0

        data: dict[str, Any] = {
            "run_id": str(self.id),
            "objective": self.objective,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "is_grounded": is_grounded,
            "has_evidence": has_evidence,
            "citation_count": citation_count,
        }
        return sanitize_data(data)

    def to_detail_dict(self) -> dict[str, Any]:
        """Return full status or completed research result."""
        data: dict[str, Any] = {
            "run_id": str(self.id),
            "objective": self.objective,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }

        if self.status == self.STATUS_COMPLETED and isinstance(self.result, dict):
            data.update(
                {
                    "final_answer": self.result.get("final_answer", ""),
                    "evidence": self.result.get("evidence", []),
                    "sources": self.result.get("sources", []),
                    "citations": self.result.get("citations", []),
                    "queries": self.result.get("queries", []),
                    "iteration_count": self.result.get("iteration_count", 0),
                    "is_grounded": self.result.get("is_grounded", False),
                    "has_evidence": self.result.get("has_evidence", False),
                    "errors": self.result.get("errors", []),
                    "metadata": self.result.get("metadata", {}),
                }
            )
        elif self.status == self.STATUS_FAILED:
            msg = self.error_message or "Research execution failed."
            data["error"] = msg
            data["errors"] = [msg]
        elif self.status == self.STATUS_CANCELLED:
            msg = self.error_message or "Research was cancelled."
            data["error"] = msg
            data["errors"] = [msg]

        return sanitize_data(data)
