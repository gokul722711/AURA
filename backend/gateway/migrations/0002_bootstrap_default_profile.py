"""Data migration to bootstrap initial ModelProfile from environment configuration (M15)."""

from django.conf import settings
from django.db import migrations


def bootstrap_default_profile(apps, schema_editor):
    """Seed initial ModelProfile from existing environment configuration if none exists."""
    ModelProfile = apps.get_model("gateway", "ModelProfile")
    if ModelProfile.objects.exists():
        return

    gateway_conf = getattr(settings, "AI_GATEWAY", {})
    provider = (gateway_conf.get("PROVIDER") or "mock").strip().lower()
    model = (gateway_conf.get("MODEL") or "mock-model").strip()
    endpoint = (gateway_conf.get("ENDPOINT") or "").strip()
    api_key = (gateway_conf.get("API_KEY") or "").strip()
    timeout = float(gateway_conf.get("TIMEOUT", 30.0))

    name = "Default Profile"
    if provider == "nvidia":
        name = "NVIDIA Nemotron"
    elif provider == "ollama":
        name = "Local Ollama"
    elif provider == "mock":
        name = "Default Mock"
    elif provider == "openai_compatible":
        name = "Local Model"

    ModelProfile.objects.create(
        name=name,
        provider=provider,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout=timeout,
        is_active=True,
    )


def remove_default_profile(apps, schema_editor):
    """No-op for reversing bootstrap."""
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("gateway", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(bootstrap_default_profile, remove_default_profile),
    ]
