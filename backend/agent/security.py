"""Security utilities for sanitizing sensitive credentials in responses and logs."""

import os
import re
from typing import Any

from django.conf import settings

# Patterns for sensitive tokens (NVIDIA, OpenAI, Bearer tokens, URLs with passwords)
_SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"://([^:]+):([^@]+)@", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Sanitize sensitive credentials, API keys, and tokens from a string."""
    if not text:
        return text

    sanitized = text

    # Strip configured gateway API key if present
    try:
        gw_conf = getattr(settings, "AI_GATEWAY", {})
        configured_key = gw_conf.get("API_KEY")
        if configured_key and isinstance(configured_key, str) and len(configured_key.strip()) > 3:
            sanitized = sanitized.replace(configured_key.strip(), "[REDACTED]")
    except Exception:
        pass

    # Strip environment API key if present
    env_key = os.environ.get("AI_API_KEY")
    if env_key and len(env_key.strip()) > 3:
        sanitized = sanitized.replace(env_key.strip(), "[REDACTED]")

    # Apply regex patterns for known token formats
    sanitized = re.sub(r"nvapi-[A-Za-z0-9_\-\.]{4,}", "[REDACTED]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"sk-[A-Za-z0-9_\-\.]{4,}", "[REDACTED]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{4,}", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"://([^:]+):([^@]+)@", r"://\1:[REDACTED]@", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"(?i)(key|secret|token|password|auth)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{4,})['\"]?",
        r"\1=[REDACTED]",
        sanitized,
    )

    return sanitized


def sanitize_data(data: Any) -> Any:
    """Recursively sanitize dictionaries, lists, and strings."""
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data
