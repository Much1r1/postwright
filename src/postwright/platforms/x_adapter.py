import os
import re

from postwright.platforms.base import PlatformAdapter, PublishResult, ValidationResult
from postwright.platforms.http_client import XHttpClient

# X API URL fixed length rule:
# X counts any http:// or https:// URL as exactly 23 characters regardless of length.
# Ref: https://developer.x.com/en/docs/counting-characters
X_URL_WEIGHTED_LENGTH = 23
X_TWEET_MAX_LENGTH = 280

URL_REGEX = re.compile(r"https?://\S+")


def count_x_weighted_length(text: str) -> int:
    """Calculate character length using X's weighted URL counting rule.

    Rule: Every URL starting with http:// or https:// counts as 23 characters.
    Non-URL text is counted by character length.
    """
    if not text:
        return 0

    total_len = 0
    last_end = 0

    for match in URL_REGEX.finditer(text):
        start, end = match.span()
        # Non-URL text before this match
        total_len += start - last_end
        # URL counts as fixed 23 weighted length
        total_len += X_URL_WEIGHTED_LENGTH
        last_end = end

    # Remaining non-URL text after last match
    total_len += len(text) - last_end
    return total_len


def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentences preserving sentence boundary punctuation."""
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    sentences: list[str] = []

    for block in raw_blocks:
        parts = re.split(r"(?<=[.!?])\s+", block)
        for p in parts:
            p_str = p.strip()
            if p_str:
                sentences.append(p_str)

    return sentences if sentences else [text.strip()]


class XAdapter(PlatformAdapter):
    """Platform adapter for X (Twitter) with weighted URL length counting,

    sentence-boundary thread splitting, and API publishing.
    """

    def __init__(self, http_client: XHttpClient | None = None) -> None:
        self.http_client = http_client or XHttpClient()

    def validate(self, text: str) -> ValidationResult:
        """Validate text against X limits (280 weighted characters per tweet/thread part)."""
        text = text.strip()
        weighted_len = count_x_weighted_length(text)

        if weighted_len <= X_TWEET_MAX_LENGTH:
            return ValidationResult(
                is_valid=True,
                errors=[],
                char_count=weighted_len,
                max_length=X_TWEET_MAX_LENGTH,
                thread_parts=[text],
            )

        thread_parts = self.split(text)
        errors: list[str] = []
        for idx, part in enumerate(thread_parts, start=1):
            p_len = count_x_weighted_length(part)
            if p_len > X_TWEET_MAX_LENGTH:
                errors.append(
                    f"Thread part {idx} exceeds limit ({p_len}/{X_TWEET_MAX_LENGTH} weighted chars)"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            char_count=weighted_len,
            max_length=X_TWEET_MAX_LENGTH,
            thread_parts=thread_parts,
        )

    def split(self, text: str) -> list[str]:
        """Split long text into a numbered thread (1/N ...) at sentence boundaries.

        Never splits mid-word.
        """
        text = text.strip()
        weighted_len = count_x_weighted_length(text)
        if weighted_len <= X_TWEET_MAX_LENGTH:
            return [text]

        sentences = split_text_into_sentences(text)

        parts_raw: list[list[str]] = []
        curr_sentences: list[str] = []
        curr_len = 0

        for s in sentences:
            s_len = count_x_weighted_length(s)
            if curr_sentences and (curr_len + 1 + s_len > X_TWEET_MAX_LENGTH - 8):
                parts_raw.append(curr_sentences)
                curr_sentences = [s]
                curr_len = s_len
            else:
                if curr_sentences:
                    curr_len += 1 + s_len
                else:
                    curr_len = s_len
                curr_sentences.append(s)

        if curr_sentences:
            parts_raw.append(curr_sentences)

        total_parts = len(parts_raw)

        final_parts: list[str] = []
        for i, s_list in enumerate(parts_raw, start=1):
            prefix = f"{i}/{total_parts} "
            part_body = " ".join(s_list)
            full_part = f"{prefix}{part_body}"

            if count_x_weighted_length(full_part) > X_TWEET_MAX_LENGTH:
                words = part_body.split()
                sub_parts: list[str] = []
                curr_words: list[str] = []
                for w in words:
                    test_str = f"{prefix}" + " ".join(curr_words + [w])
                    if count_x_weighted_length(test_str) > X_TWEET_MAX_LENGTH:
                        if curr_words:
                            sub_parts.append(" ".join(curr_words))
                            curr_words = [w]
                        else:
                            curr_words = [w]
                    else:
                        curr_words.append(w)
                if curr_words:
                    sub_parts.append(" ".join(curr_words))
                for sp in sub_parts:
                    final_parts.append(f"{prefix}{sp}")
            else:
                final_parts.append(full_part)

        return final_parts

    def publish(self, post_text: str, *, dry_run: bool = True) -> PublishResult:
        """Publish post or dry-run post to X."""
        val = self.validate(post_text)
        if not val.is_valid:
            return PublishResult(
                success=False,
                error_message="; ".join(val.errors),
                dry_run=dry_run,
            )

        thread_parts = val.thread_parts

        if dry_run:
            return PublishResult(
                success=True,
                platform_post_id="dry-run-x-id",
                error_message=None,
                dry_run=True,
                thread_post_ids=[f"dry-run-x-id-{i+1}" for i in range(len(thread_parts))],
            )

        bearer_token = os.getenv("X_BEARER_TOKEN") or os.getenv("X_API_KEY")
        if not bearer_token:
            return PublishResult(
                success=False,
                error_message="Missing X credentials (X_BEARER_TOKEN or X_API_KEY env var required).",
                dry_run=False,
            )

        published_ids: list[str] = []
        last_id: str | None = None

        try:
            for part in thread_parts:
                res = self.http_client.post_tweet(
                    text=part,
                    bearer_token=bearer_token,
                    in_reply_to_tweet_id=last_id,
                )
                tweet_id = res.get("data", {}).get("id")
                if not tweet_id:
                    raise RuntimeError(f"Unexpected response structure from X API: {res}")
                published_ids.append(str(tweet_id))
                last_id = str(tweet_id)

            return PublishResult(
                success=True,
                platform_post_id=published_ids[0],
                error_message=None,
                dry_run=False,
                thread_post_ids=published_ids,
            )
        except (RuntimeError, ValueError) as e:
            return PublishResult(
                success=False,
                platform_post_id=published_ids[0] if published_ids else None,
                error_message=str(e),
                dry_run=False,
                thread_post_ids=published_ids,
            )
