import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("postwright.platforms.http_client")


def mask_sensitive_data(text: str) -> str:
    """Mask sensitive authorization headers, tokens, or keys in text."""
    if not text:
        return text
    masked = text
    import re

    masked = re.sub(
        r"(Bearer\s+)[A-Za-z0-9%_\-\.]+", r"\1[REDACTED]", masked, flags=re.IGNORECASE
    )
    masked = re.sub(
        r"(oauth_token|oauth_consumer_key|access_token|api_key|secret)=[^& \"]+",
        r"\1=[REDACTED]",
        masked,
        flags=re.IGNORECASE,
    )
    masked = re.sub(
        r"('Authorization':\s*')[^']+'", r"\1[REDACTED]'", masked, flags=re.IGNORECASE
    )
    masked = re.sub(
        r'("Authorization":\s*")[^"]+"', r'\1[REDACTED]"', masked, flags=re.IGNORECASE
    )
    return masked


class XHttpClient:
    """Thin HTTP client for X (Twitter) API.

    All external HTTP calls for X go through this single module so tests can mock it.
    Never logs or raises credentials or unmasked headers.
    """

    def __init__(self, api_url: str = "https://api.twitter.com/2/tweets") -> None:
        self.api_url = api_url

    def post_tweet(
        self,
        text: str,
        bearer_token: str,
        in_reply_to_tweet_id: str | None = None,
    ) -> dict[str, Any]:
        """Post a single tweet to X API v2."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        }

        payload: dict[str, Any] = {"text": text}
        if in_reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": in_reply_to_tweet_id}

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url, data=req_data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_bytes = resp.read()
                data: dict[str, Any] = json.loads(resp_bytes.decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                err_body = ""
            masked_err = mask_sensitive_data(f"HTTP {e.code}: {err_body or e.reason}")
            logger.error("X API error: %s", masked_err)
            raise RuntimeError(f"X API request failed: {masked_err}") from None
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            masked_err = mask_sensitive_data(str(e))
            logger.error("X API connection error: %s", masked_err)
            raise RuntimeError(f"X API connection failed: {masked_err}") from None
