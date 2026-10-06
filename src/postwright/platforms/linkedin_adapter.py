from postwright.platforms.base import PlatformAdapter, PublishResult, ValidationResult

LINKEDIN_MAX_LENGTH = 3000


class LinkedInAdapter(PlatformAdapter):
    """Platform adapter for LinkedIn.

    Validation supported (up to 3000 chars, no threading).
    Publishing raises NotImplementedError per specifications.
    """

    def validate(self, text: str) -> ValidationResult:
        """Validate LinkedIn post length (max 3000 characters)."""
        text = text.strip()
        char_count = len(text)
        errors: list[str] = []

        if char_count > LINKEDIN_MAX_LENGTH:
            errors.append(
                f"LinkedIn post exceeds character limit ({char_count}/{LINKEDIN_MAX_LENGTH} characters)."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            char_count=char_count,
            max_length=LINKEDIN_MAX_LENGTH,
            thread_parts=[text],
        )

    def split(self, text: str) -> list[str]:
        """LinkedIn does not support threads; returns single item list."""
        return [text.strip()]

    def publish(self, post_text: str, *, dry_run: bool = True) -> PublishResult:
        """Publishing is not implemented for LinkedIn."""
        raise NotImplementedError(
            "LinkedIn publishing is not supported yet. Postwright currently supports posting dry-runs and automated publishing for X/Twitter only."
        )
