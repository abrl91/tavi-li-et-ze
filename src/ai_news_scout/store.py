"""SQLite store for items, briefs, and the response cache."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    url        TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    topic      TEXT NOT NULL,
    score      REAL NOT NULL,
    embedding  BLOB NOT NULL,
    embed_dim  INTEGER NOT NULL,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS briefs (
    date        TEXT PRIMARY KEY,
    markdown    TEXT NOT NULL,
    stats_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    value       BLOB NOT NULL,
    expires_at  INTEGER
);
"""


class Store:
    """Thin wrapper over SQLite. Holds items+embeddings (for novelty), briefs, and cache."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("AI_NEWS_SCOUT_DB", "data/ai_news_scout.db"))
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets FastAPI's threadpool workers reuse one
        # Store instance. WAL allows concurrent reads with one writer, and
        # SQLite serializes writes internally, so this is safe in practice.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        # WAL lets reads run concurrently with one writer. NORMAL trims fsync
        # cost. busy_timeout retries instead of raising SQLITE_BUSY on contention.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def all_embeddings(self) -> tuple[np.ndarray | None, list[str]]:
        rows = self._conn.execute("SELECT url, embedding, embed_dim FROM items").fetchall()
        if not rows:
            return None, []
        dim = rows[0][2]
        n_filtered = sum(1 for row in rows if row[2] != dim)
        if n_filtered:
            # Mixed dims usually mean an embedding-model swap. Filtered rows are
            # invisible to novelty, so everything looks new until backfilled.
            _log.warning(
                "all_embeddings: %d/%d rows filtered for embed_dim != %d "
                "(likely a model swap, consider re-embedding to restore novelty)",
                n_filtered, len(rows), dim,
            )
        mat = np.stack(
            [np.frombuffer(row[1], dtype=np.float32) for row in rows if row[2] == dim]
        )
        urls = [row[0] for row in rows if row[2] == dim]
        return mat, urls

    def save_items(self, items: list, embeddings: np.ndarray, first_seen: str) -> None:
        rows = [
            (
                it.url,
                it.title,
                it.content,
                it.topic,
                float(it.score),
                emb.astype(np.float32).tobytes(),
                int(emb.shape[0]),
                first_seen,
            )
            for it, emb in zip(items, embeddings)
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO items"
            "(url,title,content,topic,score,embedding,embed_dim,first_seen) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def save_brief(self, date: str, markdown: str, stats: dict) -> None:
        # REPLACE: a re-run on the same date overwrites prior stats+markdown for that date.
        self._conn.execute(
            "INSERT OR REPLACE INTO briefs(date,markdown,stats_json,created_at) "
            "VALUES(?,?,?,?)",
            (date, markdown, json.dumps(stats), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def list_briefs(self) -> list[tuple[str, str]]:
        return self._conn.execute(
            "SELECT date, created_at FROM briefs ORDER BY date DESC"
        ).fetchall()

    def get_brief(self, date: str) -> tuple[str, dict] | None:
        row = self._conn.execute(
            "SELECT markdown, stats_json FROM briefs WHERE date=?", (date,)
        ).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1])

    def latest_brief(self) -> tuple[str, str, dict] | None:
        row = self._conn.execute(
            "SELECT date, markdown, stats_json FROM briefs ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return row[0], row[1], json.loads(row[2])

    def cache_get(self, key: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return None
        value, expires_at = row
        if expires_at is not None and expires_at < int(time.time()):
            self._conn.execute("DELETE FROM cache WHERE key=?", (key,))
            self._conn.commit()
            return None
        return value

    def cache_set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        expires_at = int(time.time()) + ttl_seconds if ttl_seconds else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cache(key, value, expires_at) VALUES(?,?,?)",
            (key, value, expires_at),
        )
        self._conn.commit()
