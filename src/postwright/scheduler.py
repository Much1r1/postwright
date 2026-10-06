from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from postwright.state import ScheduledSlot

# Type alias for convenience
Slot = ScheduledSlot

DAY_NAME_TO_INT = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class SlotPolicy(ABC):
    """Abstract interface for slot suggestion policies."""

    @abstractmethod
    def suggest(
        self,
        platform: Literal["x", "linkedin"] | str,
        now: datetime,
        queued_slots: list[ScheduledSlot | datetime] | None = None,
    ) -> ScheduledSlot:
        """Suggest the next optimal scheduled slot for a platform."""


class UniversalPriorPolicy(SlotPolicy):
    """Slot policy using schedule.yaml benchmark priors."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        user_timezone: str | None = None,
    ) -> None:
        if config_path is None:
            config_path = Path("config/schedule.yaml")
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self._load_config(user_timezone)

    def _load_config(self, user_timezone_override: str | None = None) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        self.user_timezone = (
            user_timezone_override
            or data.get("user_timezone")
            or "Africa/Nairobi"
        )
        self.priors: dict[str, list[dict[str, Any]]] = {
            "x": data.get("x", []),
            "linkedin": data.get("linkedin", []),
        }

    def suggest(
        self,
        platform: Literal["x", "linkedin"] | str,
        now: datetime,
        queued_slots: list[ScheduledSlot | datetime] | None = None,
    ) -> ScheduledSlot:
        platform_key = platform.lower()
        templates = self.priors.get(platform_key, [])
        if not templates:
            raise ValueError(f"No schedule configuration found for platform '{platform}'.")

        if now.tzinfo is None:
            now_utc = now.replace(tzinfo=UTC)
        else:
            now_utc = now.astimezone(UTC)

        booked_utcs: set[datetime] = set()
        if queued_slots:
            for s in queued_slots:
                if isinstance(s, ScheduledSlot):
                    dt = s.scheduled_at_utc
                elif isinstance(s, datetime):
                    dt = s
                else:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                else:
                    dt = dt.astimezone(UTC)
                booked_utcs.add(dt)

        # Generate candidates for the next 8 weeks
        candidates: list[tuple[datetime, datetime]] = []

        for template in templates:
            day_str = str(template.get("day_of_week", "")).strip().lower()
            target_weekday = DAY_NAME_TO_INT.get(day_str)
            if target_weekday is None:
                continue

            time_str = str(template.get("time", "12:00")).strip()
            hour, minute = map(int, time_str.split(":"))

            aud_tz_str = template.get("audience_timezone", "UTC")
            try:
                aud_tz = ZoneInfo(aud_tz_str)
            except (ZoneInfoNotFoundError, ValueError):
                aud_tz = ZoneInfo("UTC")

            now_aud = now_utc.astimezone(aud_tz)
            start_date = now_aud.date()

            # Look up to 60 days ahead for matching weekday occurrences
            for day_offset in range(60):
                candidate_date = start_date + timedelta(days=day_offset)
                if candidate_date.weekday() == target_weekday:
                    cand_aud = datetime(
                        candidate_date.year,
                        candidate_date.month,
                        candidate_date.day,
                        hour,
                        minute,
                        tzinfo=aud_tz,
                    )
                    cand_utc = cand_aud.astimezone(UTC)
                    if cand_utc > now_utc:
                        user_tz = ZoneInfo(self.user_timezone)
                        cand_local = cand_utc.astimezone(user_tz)
                        candidates.append((cand_utc, cand_local))

        # Sort candidate slots chronologically
        candidates.sort(key=lambda x: x[0])

        for cand_utc, cand_local in candidates:
            if cand_utc not in booked_utcs:
                return ScheduledSlot(
                    platform=platform_key,  # type: ignore[arg-type]
                    scheduled_at_utc=cand_utc,
                    scheduled_at_local=cand_local,
                    user_timezone=self.user_timezone,
                )

        raise RuntimeError("No available slot found in the search window.")


def check_staleness(
    config_path: str | Path | None = None,
    days_threshold: int = 90,
    now: datetime | None = None,
) -> list[str]:
    """Check if any schedule prior is older than days_threshold."""
    if config_path is None:
        config_path = Path("config/schedule.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if now is None:
        now = datetime.now(UTC)

    warnings = []
    for platform in ["x", "linkedin"]:
        entries = data.get(platform, [])
        for entry in entries:
            retrieved_str = entry.get("retrieved_on")
            if not retrieved_str:
                continue
            try:
                retrieved_dt = datetime.strptime(str(retrieved_str), "%Y-%m-%d").replace(
                    tzinfo=UTC
                )
                age_days = (now - retrieved_dt).days
                if age_days > days_threshold:
                    warnings.append(
                        f"Schedule prior for {platform.upper()} ({entry.get('day_of_week')} {entry.get('time')}) "
                        f"is {age_days} days old (retrieved on {retrieved_str}, threshold: {days_threshold} days)."
                    )
            except ValueError:
                continue

    return warnings
