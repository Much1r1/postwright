from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from postwright.platforms.http_client import XHttpClient, mask_sensitive_data
from postwright.platforms.linkedin_adapter import LinkedInAdapter
from postwright.platforms.registry import get_adapter
from postwright.platforms.x_adapter import XAdapter, count_x_weighted_length
from postwright.publisher import PublishingRefusedError, publish_due_posts, publish_post
from postwright.store import QueueStore


def test_count_x_weighted_length():
    # URL fixed length = 23 chars
    url = "https://example.com/a/very/long/url/path/that/exceeds/23/characters"
    text = f"Check this out: {url} cool!"
    # "Check this out: " = 16 chars, url = 23, " cool!" = 6 -> total = 45
    assert count_x_weighted_length(text) == 45


def test_x_adapter_validation_and_thread_split():
    adapter = XAdapter()

    # Short post
    short_text = "Building postwright today."
    val_short = adapter.validate(short_text)
    assert val_short.is_valid is True
    assert val_short.thread_parts == [short_text]

    # Long text requiring thread split at sentence boundaries
    long_text = (
        "We built a brand new feature in Postwright today that supports platform adapters. "
        "It uses weighted URL counting so links take exactly 23 characters regardless of length. "
        "When a post exceeds two hundred and eighty characters, it splits the text at sentence boundaries into a numbered thread without ever breaking words mid-string. "
        "This ensures maximum clarity and readability for readers on social media platforms."
    )
    val_long = adapter.validate(long_text)
    assert val_long.is_valid is True
    assert len(val_long.thread_parts) > 1
    for i, part in enumerate(val_long.thread_parts, start=1):
        assert part.startswith(f"{i}/")
        assert count_x_weighted_length(part) <= 280


def test_linkedin_adapter():
    adapter = LinkedInAdapter()
    val = adapter.validate("Hello LinkedIn! " * 10)
    assert val.is_valid is True

    # Over 3000 chars
    long_li = "A" * 3001
    val_over = adapter.validate(long_li)
    assert val_over.is_valid is False
    assert "exceeds character limit" in val_over.errors[0]

    with pytest.raises(NotImplementedError) as exc_info:
        adapter.publish("Test post", dry_run=False)
        assert "LinkedIn publishing is not supported yet" in str(exc_info.value)


def test_adapter_registry():
    assert isinstance(get_adapter("x"), XAdapter)
    assert isinstance(get_adapter("twitter"), XAdapter)
    assert isinstance(get_adapter("linkedin"), LinkedInAdapter)
    with pytest.raises(ValueError):
        get_adapter("unknown")


def test_dry_run_never_calls_http(monkeypatch):
    mock_http = MagicMock(spec=XHttpClient)
    adapter = XAdapter(http_client=mock_http)

    res = adapter.publish("Test post", dry_run=True)
    assert res.success is True
    assert res.dry_run is True
    assert mock_http.post_tweet.call_count == 0


def test_live_publish_requires_env_flag(tmp_path, monkeypatch):
    db_path = tmp_path / "test_store.db"
    store = QueueStore(db_path=db_path)
    rec = store.enqueue(
        thread_id="t1",
        draft_id="d1",
        platform="x",
        final_text="Test live post",
        slot_utc="2025-01-01T10:00:00+00:00",
        slot_local="2025-01-01T13:00:00+03:00",
    )

    # Without POSTWRIGHT_LIVE=1, publish_post defaults to dry-run
    monkeypatch.delenv("POSTWRIGHT_LIVE", raising=False)
    res = publish_post(rec, store=store)
    assert res.dry_run is True

    # With POSTWRIGHT_LIVE=1, live mode executes
    monkeypatch.setenv("POSTWRIGHT_LIVE", "1")
    monkeypatch.setenv("X_BEARER_TOKEN", "mock_token")

    mock_http = MagicMock()
    mock_http.post_tweet.return_value = {"data": {"id": "12345"}}
    monkeypatch.setattr("postwright.platforms.x_adapter.XHttpClient", lambda: mock_http)

    # Re-fetch or create adapter with mock
    adapter = XAdapter(http_client=mock_http)
    monkeypatch.setattr("postwright.publisher.get_adapter", lambda p: adapter)

    res_live = publish_post(rec, store=store)
    assert res_live.success is True
    assert res_live.dry_run is False
    assert res_live.platform_post_id == "12345"
    assert mock_http.post_tweet.call_count == 1

    # Verify status in store updated to published
    updated = store.get_post(rec.id)
    assert updated.status == "published"
    assert updated.platform_post_id == "12345"


def test_idempotency_and_crash_recovery(tmp_path, monkeypatch):
    db_path = tmp_path / "test_store.db"
    store = QueueStore(db_path=db_path)
    rec = store.enqueue(
        thread_id="t1",
        draft_id="d1",
        platform="x",
        final_text="Test idempotency post",
        slot_utc="2025-01-01T10:00:00+00:00",
        slot_local="2025-01-01T13:00:00+03:00",
    )

    monkeypatch.setenv("POSTWRIGHT_LIVE", "1")
    monkeypatch.setenv("X_BEARER_TOKEN", "mock_token")

    mock_http = MagicMock()
    mock_http.post_tweet.return_value = {"data": {"id": "tweet-999"}}
    adapter = XAdapter(http_client=mock_http)
    monkeypatch.setattr("postwright.publisher.get_adapter", lambda p: adapter)

    # 1. First publish succeeds
    res1 = publish_post(rec, store=store)
    assert res1.platform_post_id == "tweet-999"
    assert mock_http.post_tweet.call_count == 1

    # 2. Re-running publish on the published post never calls HTTP client again
    updated_rec = store.get_post(rec.id)
    res2 = publish_post(updated_rec, store=store)
    assert res2.success is True
    assert res2.platform_post_id == "tweet-999"
    assert mock_http.post_tweet.call_count == 1  # Still 1 call!

    # 3. Simulate re-running publish-due
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    due_results = publish_due_posts(store=store, now=now)
    assert len(due_results) == 0  # Already published post is ignored in due list


def test_failure_path_and_retry_cap(tmp_path, monkeypatch):
    db_path = tmp_path / "test_store.db"
    store = QueueStore(db_path=db_path)
    rec = store.enqueue(
        thread_id="t1",
        draft_id="d1",
        platform="x",
        final_text="Failing post",
        slot_utc="2025-01-01T10:00:00+00:00",
        slot_local="2025-01-01T13:00:00+03:00",
    )

    monkeypatch.setenv("POSTWRIGHT_LIVE", "1")
    monkeypatch.setenv("X_BEARER_TOKEN", "mock_token")

    mock_http = MagicMock()
    mock_http.post_tweet.side_effect = RuntimeError("API rate limit exceeded")
    adapter = XAdapter(http_client=mock_http)
    monkeypatch.setattr("postwright.publisher.get_adapter", lambda p: adapter)

    # Retry 1
    publish_post(rec, store=store)
    p1 = store.get_post(rec.id)
    assert p1.status == "failed"
    assert p1.retry_count == 1
    assert "rate limit" in p1.error_message

    # Retry 2
    publish_post(p1, store=store)
    p2 = store.get_post(rec.id)
    assert p2.retry_count == 2

    # Retry 3
    publish_post(p2, store=store)
    p3 = store.get_post(rec.id)
    assert p3.retry_count == 3

    # Attempt 4 (Exceeds max retries 3)
    with pytest.raises(PublishingRefusedError) as exc_info:
        publish_post(p3, store=store)
    assert "maximum retries (3) reached" in str(exc_info.value)


def test_credentials_never_leaked():
    token = "secret_x_api_token_12345"
    raw_error = f"Error with Bearer {token} and secret={token}"
    masked = mask_sensitive_data(raw_error)

    assert token not in masked
    assert "[REDACTED]" in masked
