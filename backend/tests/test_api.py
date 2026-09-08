from fastapi.testclient import TestClient

import db
import main


def test_follow_up_is_adaptive_and_bilingual(monkeypatch):
    monkeypatch.setattr(main, "get_anthropic_client", lambda: None)
    monkeypatch.setattr(main, "coverage_for", lambda answer, lang="pt": main.star_coverage(answer, lang))
    client = TestClient(main.app)

    response = client.post("/api/follow-up", json={
        "question": "Tell me about a challenge.",
        "answer": "It was difficult and I solved it.",
        "theme": "Problem solving",
        "lang": "en",
    })
    payload = response.json()
    assert response.status_code == 200
    assert payload["should_follow_up"] is True
    assert payload["source"] == "local"
    assert "What" in payload["question"]

    complete = client.post("/api/follow-up", json={
        "question": "Tell me about a challenge.",
        "answer": "When the system failed, my task was to restore it. I organized the team, fixed the issue, and the result was a 20% reduction in downtime.",
        "lang": "en",
    })
    assert complete.json()["should_follow_up"] is False


def test_tutor_returns_claude_feedback_when_client_is_available(monkeypatch):
    class Block:
        type = "text"
        text = "You covered the situation well. Add a measurable result next time."

    class Messages:
        def create(self, **_kwargs):
            return type("Message", (), {"content": [Block()]})()

    client = type("Client", (), {"messages": Messages()})()
    monkeypatch.setattr(main, "get_anthropic_client", lambda: client)
    response = TestClient(main.app).post("/api/tutor-feedback", json={
        "question": "Tell me about a challenge.",
        "answer": "I coordinated the team and solved the issue.",
        "lang": "en",
    })
    assert response.status_code == 200
    assert response.json() == {
        "feedback": "You covered the situation well. Add a measurable result next time.",
        "source": "claude",
    }


def test_question_fallback_uses_requested_language(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_anthropic_client", lambda: None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "star_session.db")
    db.init_db()
    client = TestClient(main.app)
    response = client.post("/api/questions", json={
        "resume": "Sales analyst with team leadership experience.",
        "role": "Sales Manager",
        "jd": "Lead a regional sales team.",
        "lang": "en",
    })
    payload = response.json()
    assert response.status_code == 200
    assert payload["source"] == "local"
    assert payload["questions"][0]["theme"] == "Leadership"
    assert payload["questions"][0]["question"].startswith("Tell me")


def test_insights_expose_visual_history_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "star_session.db")
    db.init_db()
    client = TestClient(main.app)
    response = client.get("/api/insights?lang=en")
    payload = response.json()
    assert response.status_code == 200
    assert payload["sessions_total"] == 0
    assert payload["answers_total"] == 0
    assert payload["coverage"]["answers"] == 0
    assert payload["timeline"] == []
    assert payload["model"]["language"] == "en"
