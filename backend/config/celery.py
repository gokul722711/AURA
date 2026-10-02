"""Celery application configuration for AURA."""

import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("aura")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
