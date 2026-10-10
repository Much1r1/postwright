from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    char_count: int = 0
    max_length: int = 280
    thread_parts: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class PublishResult:
    success: bool
    platform_post_id: str | None = None
    error_message: str | None = None
    dry_run: bool = True
    thread_post_ids: list[str] = field(default_factory=list)


class PlatformAdapter(ABC):

    @abstractmethod
    def validate(self, text_or_draft: Any) -> ValidationResult:
        """Validate text or candidate draft against platform limits and return ValidationResult."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into platform-supported parts (e.g., thread tweets)."""

    @abstractmethod
    def publish(self, post_text: str, *, dry_run: bool = True) -> PublishResult:
        """Publish post (or dry-run publish) to platform."""
