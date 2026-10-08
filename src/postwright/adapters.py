import re
from typing import Any, Protocol

URL_REGEX = re.compile(r"https?://[^\s]+")


def get_x_weighted_length(text: str) -> int:
    """Calculate weighted length for X/Twitter posts where URLs count as 23 chars."""
    # Replace any URL with 23 placeholder characters
    substituted = URL_REGEX.sub("x" * 23, text)
    return len(substituted)


class PlatformAdapter(Protocol):
    def validate(self, text_or_draft: Any) -> bool:
        ...

    def split_thread(self, text: str) -> list[str]:
        ...


class BasePlatformAdapter:
    def validate(self, text_or_draft: Any) -> bool:
        raise NotImplementedError

    def split_thread(self, text: str) -> list[str]:
        # Basic sentence boundary thread splitter
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        parts: list[str] = []
        current = ""
        for s in sentences:
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= 280:
                current += " " + s
            else:
                parts.append(current)
                current = s
        if current:
            parts.append(current)
        return parts


class XAdapter(BasePlatformAdapter):
    MAX_WEIGHTED_LENGTH = 280

    def validate(self, text_or_draft: Any) -> bool:
        if hasattr(text_or_draft, "content"):
            content = text_or_draft.content
            is_thread = getattr(text_or_draft, "is_thread", False)
            thread_parts = getattr(text_or_draft, "thread_parts", [])
            if is_thread and thread_parts:
                return all(get_x_weighted_length(p) <= self.MAX_WEIGHTED_LENGTH for p in thread_parts)
            return get_x_weighted_length(content) <= self.MAX_WEIGHTED_LENGTH
        elif isinstance(text_or_draft, dict):
            content = text_or_draft.get("content", "")
            is_thread = text_or_draft.get("is_thread", False)
            thread_parts = text_or_draft.get("thread_parts", [])
            if is_thread and thread_parts:
                return all(get_x_weighted_length(p) <= self.MAX_WEIGHTED_LENGTH for p in thread_parts)
            return get_x_weighted_length(content) <= self.MAX_WEIGHTED_LENGTH
        elif isinstance(text_or_draft, str):
            return get_x_weighted_length(text_or_draft) <= self.MAX_WEIGHTED_LENGTH
        return False


class LinkedInAdapter(BasePlatformAdapter):
    MAX_LENGTH = 3000

    def validate(self, text_or_draft: Any) -> bool:
        if hasattr(text_or_draft, "content"):
            content = text_or_draft.content
            return len(content) <= self.MAX_LENGTH
        elif isinstance(text_or_draft, dict):
            content = text_or_draft.get("content", "")
            return len(content) <= self.MAX_LENGTH
        elif isinstance(text_or_draft, str):
            return len(text_or_draft) <= self.MAX_LENGTH
        return False


def get_adapter(platform: str) -> BasePlatformAdapter:
    p = platform.lower().strip()
    if p in ("x", "twitter"):
        return XAdapter()
    elif p == "linkedin":
        return LinkedInAdapter()
    return BasePlatformAdapter()
