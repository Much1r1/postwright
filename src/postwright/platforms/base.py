from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    char_count: int = 0
    max_length: int = 280
    thread_parts: list[str] = field(default_factory=list)


@dataclass
class PublishResult:
    success: bool
    platform_post_id: str | None = None
    error_message: str | None = None
    dry_run: bool = True
    thread_post_ids: list[str] = field(default_factory=list)


class PlatformAdapter(ABC):

    @abstractmethod
    def validate(self, text: str) -> ValidationResult:
        """Validate text against platform limits and return ValidationResult."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into platform-supported parts (e.g., thread tweets)."""

    @abstractmethod
    def publish(self, post_text: str, *, dry_run: bool = True) -> PublishResult:
        """Publish post (or dry-run publish) to platform."""
