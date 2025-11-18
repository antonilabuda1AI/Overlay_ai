from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  ts_ms INTEGER NOT NULL,
  bbox TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_ms);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


@dataclass
class SessionDB:
    path: Path
    conn: sqlite3.Connection
    _buf: List[tuple] = field(default_factory=list)
    _last_flush_ms: int = 0
    _stmt: sqlite3.Cursor | None = None

    @classmethod
    def open(cls, path: Path) -> "SessionDB":
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA journal_size_limit=67108864;")
        conn.executescript(SCHEMA)
        db = cls(path=path, conn=conn)
        db._stmt = conn.cursor()
        db._last_flush_ms = int(time.time() * 1000)
        return db

    def append_event(self, ts_ms: int, bbox: List[int], text: str, source: str) -> None:
        self._buf.append((ts_ms, json.dumps(bbox), text, source))
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        from app.config import SETTINGS  # local import to avoid cycles
        now = int(time.time() * 1000)
        if len(self._buf) >= SETTINGS.db_write_batch_rows or (now - self._last_flush_ms) >= SETTINGS.db_write_batch_ms:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        self._stmt.executemany(
            "INSERT INTO events(ts_ms, bbox, text, source) VALUES (?,?,?,?)",
            self._buf,
        )
        self.conn.commit()
        self._buf.clear()
        self._last_flush_ms = int(time.time() * 1000)

    def query(self, session_id: str, from_ts: int | None, to_ts: int | None, limit: int = 200) -> List[Tuple[int, List[int], str]]:
        q = "SELECT ts_ms, bbox, text FROM events WHERE 1=1"
        params: List[int] = []
        if from_ts is not None:
            q += " AND ts_ms >= ?"
            params.append(from_ts)
        if to_ts is not None:
            q += " AND ts_ms <= ?"
            params.append(to_ts)
        q += " ORDER BY ts_ms ASC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(q, tuple(params)).fetchall()
        out = []
        for ts_ms, bbox, text in rows:
            out.append((int(ts_ms), json.loads(bbox), str(text)))
        return out

    def close(self) -> None:
        self._flush()
        self.conn.close()
