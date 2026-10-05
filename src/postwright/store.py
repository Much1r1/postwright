import sqlite3
from pathlib import Path
from typing import Literal

DEFAULT_DB_PATH = Path("postwright_store.db")

FormatType = Literal["build_log", "lesson", "opinion", "diagram_prompt"]
FORMAT_CYCLE: list[FormatType] = ["build_log", "lesson", "opinion", "diagram_prompt"]


class AngleHistoryStore:

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS angle_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    format TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_recent_formats(self, limit: int = 10) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT format FROM angle_history ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            return [row["format"] for row in rows]

    def record_format(self, format_name: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO angle_history (format) VALUES (?)", (format_name,)
            )
            conn.commit()

    def pick_next_formats(self, count: int) -> list[FormatType]:
        recent = self.get_recent_formats(limit=10)
        last_format = recent[0] if recent else None

        if last_format in FORMAT_CYCLE:
            start_index = (FORMAT_CYCLE.index(last_format) + 1) % len(FORMAT_CYCLE)
        else:
            start_index = 0

        selected: list[FormatType] = []
        for i in range(count):
            fmt = FORMAT_CYCLE[(start_index + i) % len(FORMAT_CYCLE)]
            selected.append(fmt)
            self.record_format(fmt)

        return selected
