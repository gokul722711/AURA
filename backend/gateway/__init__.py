"""AURA Model Gateway package.

Provides provider-agnostic access to language models for AURA.
"""

from gateway.base import LLMProvider
from gateway.config import GatewayConfig
from gateway.exceptions import (
    GatewayError,
    GenerationError,
    InvalidRequestError,
    NoModelConfiguredError,
    ProviderConfigurationError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from gateway.gateway import ModelGateway, get_gateway
from gateway.providers.mock import MockLLMProvider
from gateway.registry import get_provider_class, register_provider
from gateway.types import (
    ALL_CAPABILITIES,
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    GenerationResponse,
    Message,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
    UsageInfo,
)

__all__ = [
    # Gateway & Factory
    "ModelGateway",
    "get_gateway",
    # Base Provider
    "LLMProvider",
    # Providers
    "MockLLMProvider",
    # Registry
    "register_provider",
    "get_provider_class",
    # Config
    "GatewayConfig",
    # Types & Contracts
    "Message",
    "UsageInfo",
    "GenerationRequest",
    "GenerationResponse",
    "StreamChunk",
    "StructuredOutputRequest",
    "StructuredOutputResponse",
    "ProviderMetadata",
    # Capabilities
    "CAPABILITY_GENERATION",
    "CAPABILITY_STREAMING",
    "CAPABILITY_STRUCTURED_OUTPUT",
    "ALL_CAPABILITIES",
    # Exceptions
    "GatewayError",
    "ProviderUnavailableError",
    "UnsupportedCapabilityError",
    "InvalidRequestError",
    "ProviderConfigurationError",
    "NoModelConfiguredError",
    "GenerationError",
]
