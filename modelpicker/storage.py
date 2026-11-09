from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """Single-owner local storage. Each operation uses its own SQLite connection."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / "modelpicker.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS records "
                "(id TEXT PRIMARY KEY, kind TEXT NOT NULL, created TEXT NOT NULL, body TEXT NOT NULL)"
            )
        self.path.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def create(self, kind: str, body: dict[str, Any]) -> dict:
        value = {**body, "id": uuid.uuid4().hex, "created_at": now()}
        with self.lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?)",
                (value["id"], kind, value["created_at"], json.dumps(value, allow_nan=False)),
            )
        return value

    def get(self, kind: str, record_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT body FROM records WHERE id=? AND kind=?", (record_id, kind)).fetchone()
        if not row:
            raise ValueError(f"Unknown {kind} ID: {record_id}")
        return json.loads(row[0])

    def update(self, kind: str, record_id: str, **changes) -> dict:
        with self.lock:
            value = {**self.get(kind, record_id), **changes, "updated_at": now()}
            with self.connect() as conn:
                conn.execute(
                    "UPDATE records SET body=? WHERE id=? AND kind=?",
                    (json.dumps(value, allow_nan=False), record_id, kind),
                )
            return value

    def list(self, kind: str, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY created DESC LIMIT ?", (kind, limit)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, kind: str, record_id: str) -> None:
        self.get(kind, record_id)
        with self.lock, self.connect() as conn:
            conn.execute("DELETE FROM records WHERE id=? AND kind=?", (record_id, kind))
