import logging
import os
from datetime import UTC, datetime

from postwright.config import get_settings
from postwright.platforms.base import PublishResult
from postwright.platforms.registry import get_adapter
from postwright.store import QueuedPostRecord, QueueStore

logger = logging.getLogger("postwright.publisher")

MAX_RETRIES = 3


class PublishingRefusedError(Exception):
    """Raised when a publishing attempt violates safety requirements."""


def is_live_enabled() -> bool:
    """Check if live publishing is explicitly enabled via POSTWRIGHT_LIVE env var or settings."""
    env_live = os.getenv("POSTWRIGHT_LIVE", "").strip().lower()
    if env_live in ("1", "true", "yes", "on"):
        return True
    return get_settings().postwright_live


def publish_post(
    post: QueuedPostRecord,
    store: QueueStore | None = None,
    *,
    force_live: bool = False,
) -> PublishResult:
    """Publish a single queued post following all safety rules.

    Safety rules enforced:
    1. Dry-run is default. Real publish requires BOTH `POSTWRIGHT_LIVE=1` AND row status in (`queued`, `failed` with retry_count < 3).
    2. Idempotency: Posts with status `published` or existing platform_post_id return cached success without re-publishing.
    3. Failure path: Records failure message and increments retry_count. Stays `failed` permanently if retry_count >= 3.
    """
    if store is None:
        store = QueueStore()

    # Refuse if post is already published (Idempotency)
    if post.status == "published":
        return PublishResult(
            success=True,
            platform_post_id=post.platform_post_id or "already-published",
            error_message=None,
            dry_run=False,
            thread_post_ids=[],
        )

    # Refuse if status is not queued or failed
    if post.status not in ("queued", "failed", "approved"):
        raise PublishingRefusedError(
            f"Refusing to publish post ID {post.id}: status is '{post.status}', expected 'queued' or 'failed'."
        )

    # Check retry count cap
    if post.retry_count >= MAX_RETRIES:
        msg = f"Refusing to publish post ID {post.id}: maximum retries ({MAX_RETRIES}) reached."
        store.mark_failed(post.id, msg, increment_retry=False)
        raise PublishingRefusedError(msg)

    live_mode = is_live_enabled() or force_live

    # Refuse live publish if POSTWRIGHT_LIVE is not set
    if not live_mode:
        # In dry-run mode, simulate dry-run via adapter
        adapter = get_adapter(post.platform)
        res = adapter.publish(post.final_text, dry_run=True)
        return res

    # Live Mode Execution
    adapter = get_adapter(post.platform)
    res = adapter.publish(post.final_text, dry_run=False)

    if res.success and res.platform_post_id:
        store.mark_published(post.id, res.platform_post_id)
    else:
        err_msg = res.error_message or "Unknown platform publishing failure."
        store.mark_failed(post.id, err_msg, increment_retry=True)

    return res


def publish_due_posts(
    store: QueueStore | None = None,
    now: datetime | None = None,
) -> list[PublishResult]:
    """Find and publish queued posts whose scheduled slot_utc has passed."""
    if store is None:
        store = QueueStore()

    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    queued_posts = store.get_queued_posts(status="queued")
    failed_posts = store.get_queued_posts(status="failed")

    # Retry eligible failed posts if retry_count < 3
    retryable_failed = [p for p in failed_posts if p.retry_count < MAX_RETRIES]
    candidates = queued_posts + retryable_failed

    due_results: list[PublishResult] = []

    for post in candidates:
        dt_utc = datetime.fromisoformat(post.slot_utc)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC)
        else:
            dt_utc = dt_utc.astimezone(UTC)

        if dt_utc <= now:
            try:
                res = publish_post(post, store=store)
                due_results.append(res)
            except PublishingRefusedError as e:
                logger.warning("Publishing refused for post %s: %s", post.id, e)

    return due_results
