"""Data representations and contracts for AURA Model Gateway."""

from dataclasses import dataclass, field
from typing import Any

from gateway.exceptions import InvalidRequestError

CAPABILITY_GENERATION = "generation"
CAPABILITY_STREAMING = "streaming"
CAPABILITY_STRUCTURED_OUTPUT = "structured_output"

ALL_CAPABILITIES = (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
)


@dataclass(frozen=True)
class Message:
    """Represents a conversation message in a provider-agnostic manner."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _normalize_messages(messages: Any) -> list[Message]:
    """Normalize and validate a list of messages."""
    if not isinstance(messages, (list, tuple)):
        raise InvalidRequestError("Messages must be provided as a list or tuple.")
    if len(messages) == 0:
        raise InvalidRequestError("Messages list must not be empty.")

    normalized: list[Message] = []
    for idx, item in enumerate(messages):
        if isinstance(item, Message):
            normalized.append(item)
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if not role or not isinstance(role, str):
                raise InvalidRequestError(f"Message at index {idx} has invalid or missing 'role'.")
            if content is None or not isinstance(content, str):
                raise InvalidRequestError(f"Message at index {idx} has invalid or missing 'content'.")
            normalized.append(Message(role=role, content=content))
        else:
            raise InvalidRequestError(
                f"Message at index {idx} must be a Message instance or dict, got {type(item).__name__}."
            )
    return normalized


@dataclass
class UsageInfo:
    """Observability metadata on token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationRequest:
    """Request representation for model text generation."""

    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = _normalize_messages(self.messages)
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise InvalidRequestError("Temperature must be between 0.0 and 2.0.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise InvalidRequestError("max_tokens must be greater than 0.")


@dataclass
class GenerationResponse:
    """Response representation for model text generation."""

    text: str
    provider: str
    model: str
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A single chunk emitted during streaming generation."""

    text: str
    index: int = 0
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredOutputRequest:
    """Request representation for structured model output."""

    messages: list[Message]
    schema: dict[str, Any] | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = _normalize_messages(self.messages)
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise InvalidRequestError("Temperature must be between 0.0 and 2.0.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise InvalidRequestError("max_tokens must be greater than 0.")


@dataclass
class StructuredOutputResponse:
    """Response representation for structured model output."""

    data: Any
    raw_text: str
    provider: str
    model: str
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderMetadata:
    """Metadata describing provider identity, model, and capabilities."""

    provider: str
    model: str
    capabilities: tuple[str, ...] = ALL_CAPABILITIES

    def supports(self, capability: str) -> bool:
        """Check whether the provider supports a given capability."""
        return capability in self.capabilities
