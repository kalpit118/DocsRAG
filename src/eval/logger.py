"""
SQLite logging for query and evaluation results.

Called after generation to record /ask responses, and by scripts/compare_configs.py
to store benchmark metrics. src.api.main reads the latest eval rows for /eval/latest.
"""

import os
import sqlite3
from contextlib import contextmanager

from src.api.schemas import EvalResult

_DB_PATH = os.getenv("SQLITE_DB_PATH", "db/eval.db")

_DDL = """
CREATE TABLE IF NOT EXISTS queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question     TEXT,
    chunking     TEXT,
    use_reranker INTEGER,
    answer       TEXT,
    grounded     INTEGER,
    latency_ms   REAL,
    timestamp    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    config         TEXT,
    hit_at_1       INTEGER,
    hit_at_5       INTEGER,
    hit_at_10      INTEGER,
    mrr            REAL,
    faithfulness   REAL,
    avg_latency_ms REAL,
    timestamp      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    sources_json TEXT,
    grounded     INTEGER,
    latency_ms   REAL,
    timestamp    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, id);
"""


@contextmanager
def _conn(db_path: str = _DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_DDL)
        yield con
        con.commit()
    finally:
        con.close()


def log_query(
    question: str,
    chunking: str,
    use_reranker: bool,
    answer: str,
    grounded: bool,
    latency_ms: float,
    db_path: str = _DB_PATH,
) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO queries (question, chunking, use_reranker, answer, grounded, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, chunking, int(use_reranker), answer, int(grounded), latency_ms),
        )
        return cur.lastrowid


def log_eval(
    config: str,
    hit_at_1: bool,
    hit_at_5: bool,
    hit_at_10: bool,
    mrr: float,
    faithfulness: float,
    avg_latency_ms: float,
    db_path: str = _DB_PATH,
) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO eval_results "
            "(config, hit_at_1, hit_at_5, hit_at_10, mrr, faithfulness, avg_latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (config, int(hit_at_1), int(hit_at_5), int(hit_at_10), mrr, faithfulness, avg_latency_ms),
        )
        return cur.lastrowid


def get_latest_evals(n: int = 10, db_path: str = _DB_PATH) -> list[EvalResult]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM eval_results ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [EvalResult(**dict(r)) for r in rows]


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    grounded: bool | None = None,
    latency_ms: float | None = None,
    db_path: str = _DB_PATH,
) -> int:
    import json
    sources_json = json.dumps(sources) if sources is not None else None
    grounded_int = int(grounded) if grounded is not None else None
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources_json, grounded, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, sources_json, grounded_int, latency_ms),
        )
        return cur.lastrowid


def get_session_messages(session_id: str, limit: int = 30, db_path: str = _DB_PATH) -> list[dict]:
    import json
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("sources_json"):
            try:
                d["sources"] = json.loads(d["sources_json"])
            except Exception:
                d["sources"] = []
        else:
            d["sources"] = []
        d["grounded"] = bool(d["grounded"]) if d["grounded"] is not None else None
        results.append(d)
    return results


def clear_session_messages(session_id: str, db_path: str = _DB_PATH) -> int:
    with _conn(db_path) as con:
        cur = con.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        return cur.rowcount

