"""Django AppConfig for the Model Gateway application."""

from django.apps import AppConfig


class GatewayAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gateway"
    verbose_name = "Model Gateway"
