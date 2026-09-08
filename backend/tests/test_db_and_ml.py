import db
import ml


def configure_test_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "star_session.db")
    monkeypatch.setattr(ml, "MODEL_DIR", tmp_path / "models")
    ml._model_cache.clear()
    ml._model_loaded.clear()
    db.init_db()


def test_session_persistence_is_language_aware(tmp_path, monkeypatch):
    configure_test_storage(tmp_path, monkeypatch)

    session_id = db.save_session(
        role="Sales Manager",
        resume_excerpt="resume",
        jd_excerpt="jd",
        questions_source="local",
        report_text="report",
        report_source="local",
        transcripts=[{
            "theme": "Leadership",
            "question": "Tell me about leadership.",
            "answer": "I led the team and the result was a 20% increase.",
            "skipped": False,
        }],
        coverage_fn=lambda _: ["situation", "action", "result"],
        lang="en",
    )

    assert session_id == 1
    assert db.session_count("en") == 1
    assert db.session_count("pt") == 0
    assert db.answer_count("en") == 1
    assert db.all_answers("en")[0]["lang"] == "en"
    assert db.coverage_summary("en")["result"] == 1.0
    assert db.history(lang="en")[0]["coverage"] == 0.75


def test_existing_database_gets_language_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    conn = db.get_conn()
    conn.executescript(
        """
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, created_at REAL NOT NULL, role TEXT NOT NULL DEFAULT '', resume_excerpt TEXT NOT NULL DEFAULT '', jd_excerpt TEXT NOT NULL DEFAULT '', questions_source TEXT NOT NULL DEFAULT '', report_text TEXT NOT NULL DEFAULT '', report_source TEXT NOT NULL DEFAULT '');
        CREATE TABLE answers (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, idx INTEGER NOT NULL, theme TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '', skipped INTEGER NOT NULL DEFAULT 0, word_count INTEGER NOT NULL DEFAULT 0, cov_situation INTEGER NOT NULL DEFAULT 0, cov_task INTEGER NOT NULL DEFAULT 0, cov_action INTEGER NOT NULL DEFAULT 0, cov_result INTEGER NOT NULL DEFAULT 0);
        """
    )
    conn.commit()
    conn.close()

    db.init_db()

    conn = db.get_conn()
    session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    answer_columns = {row[1] for row in conn.execute("PRAGMA table_info(answers)")}
    conn.close()
    assert "lang" in session_columns
    assert "lang" in answer_columns


def test_ml_trains_separate_artifacts_for_each_language(tmp_path, monkeypatch):
    configure_test_storage(tmp_path, monkeypatch)

    def labels(answer):
        return ["situation", "task", "action", "result"] if "complete" in answer else ["result"]

    for lang in ("pt", "en"):
        for index in range(20):
            db.save_session(
                role="",
                resume_excerpt="",
                jd_excerpt="",
                questions_source="local",
                report_text="",
                report_source="local",
                transcripts=[{
                    "theme": "Leadership",
                    "question": "Q",
                    "answer": f"{lang} {'complete' if index % 2 else 'short'} answer {index}",
                    "skipped": False,
                }],
                coverage_fn=labels,
                lang=lang,
            )

    assert ml.train("pt") is not None
    assert ml.train("en") is not None
    assert ml.model_path("pt").exists()
    assert ml.model_path("en").exists()
    assert ml.model_status("pt")["trained"] is True
    assert ml.model_status("en")["trained"] is True
    assert ml.model_status("pt")["answers_available"] == 20
    assert ml.model_status("en")["answers_available"] == 20
