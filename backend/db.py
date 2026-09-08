"""Persistência local em SQLite das sessões de entrevista.

Guarda cada sessão (currículo/JD resumidos, perguntas, respostas e relatório
final) para que o módulo `ml.py` possa aprender, com o tempo, os padrões de
resposta do candidato. Todo o banco fica em `backend/data/`, fora do git —
contém trechos de currículo e respostas faladas, então é dado sensível local.
"""

import sqlite3
import time
from pathlib import Path
from typing import Callable, List, Optional

DB_PATH = Path(__file__).resolve().parent / "data" / "sessao_star.db"


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
            cov_situacao INTEGER NOT NULL DEFAULT 0,
            cov_tarefa INTEGER NOT NULL DEFAULT 0,
            cov_acao INTEGER NOT NULL DEFAULT 0,
            cov_resultado INTEGER NOT NULL DEFAULT 0
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
                cov_situacao, cov_tarefa, cov_acao, cov_resultado)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                i,
                t.get("theme", ""),
                t.get("question", ""),
                answer,
                1 if skipped else 0,
                len(answer.split()) if answer else 0,
                1 if "situação" in covered else 0,
                1 if "tarefa" in covered else 0,
                1 if "ação" in covered else 0,
                1 if "resultado" in covered else 0,
            ),
        )
    conn.commit()
    conn.close()
    return session_id


def all_answers() -> List[dict]:
    """Todas as respostas reais (não puladas, não vazias) já registradas — usadas para treinar o modelo."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM answers WHERE skipped = 0 AND answer != ''").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def theme_stats() -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT theme,
                  COUNT(*) AS n,
                  AVG(cov_situacao) AS situacao,
                  AVG(cov_tarefa) AS tarefa,
                  AVG(cov_acao) AS acao,
                  AVG(cov_resultado) AS resultado,
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
