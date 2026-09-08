"""Local SQLite persistence for interview sessions.

Stores every session (résumé/JD excerpts, questions, answers, and final
report) so the `ml.py` module can learn the candidate's answer patterns over
time. The whole database lives under `backend/data/`, outside git — it holds
résumé excerpts and spoken answers, which is sensitive local data.
"""

import sqlite3
import time
from pathlib import Path
from typing import Callable, List, Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "star_session.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            role TEXT NOT NULL DEFAULT '',
            resume_excerpt TEXT NOT NULL DEFAULT '',
            jd_excerpt TEXT NOT NULL DEFAULT '',
            questions_source TEXT NOT NULL DEFAULT '',
            report_text TEXT NOT NULL DEFAULT '',
            report_source TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            theme TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '',
            skipped INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER NOT NULL DEFAULT 0,
            cov_situation INTEGER NOT NULL DEFAULT 0,
            cov_task INTEGER NOT NULL DEFAULT 0,
            cov_action INTEGER NOT NULL DEFAULT 0,
            cov_result INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_answers_session ON answers(session_id);
        """
    )
    conn.commit()
    conn.close()


def save_session(
    role: str,
    resume_excerpt: str,
    jd_excerpt: str,
    questions_source: str,
    report_text: str,
    report_source: str,
    transcripts: List[dict],
    coverage_fn: Callable[[str], List[str]],
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO sessions
           (created_at, role, resume_excerpt, jd_excerpt, questions_source, report_text, report_source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), role, resume_excerpt, jd_excerpt, questions_source, report_text, report_source),
    )
    session_id = cur.lastrowid

    for i, t in enumerate(transcripts):
        answer = (t.get("answer") or "").strip()
        skipped = bool(t.get("skipped"))
        covered = coverage_fn(answer) if (answer and not skipped) else []
        conn.execute(
            """INSERT INTO answers
               (session_id, idx, theme, question, answer, skipped, word_count,
                cov_situation, cov_task, cov_action, cov_result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                i,
                t.get("theme", ""),
                t.get("question", ""),
                answer,
                1 if skipped else 0,
                len(answer.split()) if answer else 0,
                1 if "situation" in covered else 0,
                1 if "task" in covered else 0,
                1 if "action" in covered else 0,
                1 if "result" in covered else 0,
            ),
        )
    conn.commit()
    conn.close()
    return session_id


def all_answers() -> List[dict]:
    """Every real answer recorded so far (not skipped, not empty) — used to train the model."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM answers WHERE skipped = 0 AND answer != ''").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def theme_stats() -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT theme,
                  COUNT(*) AS n,
                  AVG(cov_situation) AS situation,
                  AVG(cov_task) AS task,
                  AVG(cov_action) AS action,
                  AVG(cov_result) AS result,
                  AVG(word_count) AS avg_words
           FROM answers
           WHERE skipped = 0 AND answer != ''
           GROUP BY theme"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def session_count() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return n


def answer_count() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM answers WHERE skipped = 0 AND answer != ''").fetchone()[0]
    conn.close()
    return n
