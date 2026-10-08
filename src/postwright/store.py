import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from postwright.config import get_settings

DEFAULT_DB_PATH = Path("postwright_store.db")

FormatType = Literal["build_log", "lesson", "opinion", "diagram_prompt"]


class QueuedPostRecord(BaseModel):
    id: int
    thread_id: str
    draft_id: str
    platform: str
    final_text: str
    slot_utc: str
    slot_local: str
    user_timezone: str
    status: str = "queued"
    created_at: str | None = None


class QueueStore:

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path = get_settings().postwright_store_db
        else:
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
                CREATE TABLE IF NOT EXISTS post_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    draft_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    slot_utc TEXT NOT NULL,
                    slot_local TEXT NOT NULL,
                    user_timezone TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def enqueue(
        self,
        thread_id: str,
        draft_id: str,
        platform: str,
        final_text: str,
        slot_utc: str,
        slot_local: str,
        user_timezone: str = "Africa/Nairobi",
        status: str = "queued",
    ) -> QueuedPostRecord:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO post_queue (
                    thread_id, draft_id, platform, final_text, slot_utc, slot_local, user_timezone, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    draft_id,
                    platform,
                    final_text,
                    slot_utc,
                    slot_local,
                    user_timezone,
                    status,
                ),
            )
            rec_id = cursor.lastrowid
            conn.commit()

            cursor.execute("SELECT * FROM post_queue WHERE id = ?", (rec_id,))
            row = cursor.fetchone()
            return QueuedPostRecord(
                id=row["id"],
                thread_id=row["thread_id"],
                draft_id=row["draft_id"],
                platform=row["platform"],
                final_text=row["final_text"],
                slot_utc=row["slot_utc"],
                slot_local=row["slot_local"],
                user_timezone=row["user_timezone"],
                status=row["status"],
                created_at=str(row["created_at"]),
            )

    def get_queued_posts(self, status: str = "queued") -> list[QueuedPostRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, thread_id, draft_id, platform, final_text, slot_utc, slot_local, user_timezone, status, created_at
                FROM post_queue
                WHERE status = ?
                ORDER BY slot_utc ASC
                """,
                (status,),
            )
            rows = cursor.fetchall()
            return [
                QueuedPostRecord(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    draft_id=row["draft_id"],
                    platform=row["platform"],
                    final_text=row["final_text"],
                    slot_utc=row["slot_utc"],
                    slot_local=row["slot_local"],
                    user_timezone=row["user_timezone"],
                    status=row["status"],
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]

    def cancel_post(self, post_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE post_queue SET status = 'cancelled' WHERE id = ? AND status = 'queued'",
                (post_id,),
            )
            conn.commit()
            return cursor.rowcount > 0


class ApprovalHistoryRecord(BaseModel):
    id: int | None = None
    thread_id: str
    draft_id: str
    original_draft: str
    final_text: str
    unified_diff: str
    score: int | None = None
    cost: float = 0.0
    created_at: str | None = None


class ApprovalHistoryStore:

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path = get_settings().postwright_store_db
        else:
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
                CREATE TABLE IF NOT EXISTS approval_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    draft_id TEXT NOT NULL,
                    original_draft TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    unified_diff TEXT NOT NULL,
                    score INTEGER,
                    cost REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(approval_history)")
            cols = [row[1] for row in cursor.fetchall()]
            if "cost" not in cols:
                conn.execute("ALTER TABLE approval_history ADD COLUMN cost REAL DEFAULT 0.0")
            conn.commit()

    def record_approval(
        self,
        thread_id: str,
        draft_id: str,
        original_draft: str,
        final_text: str,
        unified_diff: str,
        score: int | None = None,
        cost: float = 0.0,
    ) -> ApprovalHistoryRecord:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO approval_history (thread_id, draft_id, original_draft, final_text, unified_diff, score, cost)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, draft_id, original_draft, final_text, unified_diff, score, cost),
            )
            rec_id = cursor.lastrowid
            conn.commit()

            cursor.execute(
                "SELECT id, thread_id, draft_id, original_draft, final_text, unified_diff, score, cost, created_at FROM approval_history WHERE id = ?",
                (rec_id,),
            )
            row = cursor.fetchone()
            return ApprovalHistoryRecord(
                id=row["id"],
                thread_id=row["thread_id"],
                draft_id=row["draft_id"],
                original_draft=row["original_draft"],
                final_text=row["final_text"],
                unified_diff=row["unified_diff"],
                score=row["score"],
                cost=row["cost"] if row["cost"] is not None else 0.0,
                created_at=str(row["created_at"]),
            )

    def get_history(self, limit: int = 50) -> list[ApprovalHistoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, thread_id, draft_id, original_draft, final_text, unified_diff, score, cost, created_at
                FROM approval_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                ApprovalHistoryRecord(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    draft_id=row["draft_id"],
                    original_draft=row["original_draft"],
                    final_text=row["final_text"],
                    unified_diff=row["unified_diff"],
                    score=row["score"],
                    cost=row["cost"] if row["cost"] is not None else 0.0,
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]


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
