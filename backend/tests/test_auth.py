from fastapi.testclient import TestClient

import db
import main


def test_authentication_persists_and_protects_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "star_session.db")
    db.init_db()
    client = TestClient(main.app)

    assert client.get("/").status_code == 200
    assert "A entrevista acontece" in client.get("/").text
    assert client.get("/login").status_code == 200
    assert "Acesso seguro" in client.get("/login").text or "ACESSO SEGURO" in client.get("/login").text
    assert client.get("/app", follow_redirects=False).status_code == 303

    register = client.post("/api/auth/register", json={
        "name": "Julio Pessan",
        "email": "julio@example.com",
        "password": "secure-pass-123",
    })
    assert register.status_code == 200
    assert register.json()["user"]["email"] == "julio@example.com"

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["name"] == "Julio Pessan"
    assert client.get("/app").status_code == 200

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/insights?lang=en").status_code == 401

    login = client.post("/api/auth/login", json={
        "email": "julio@example.com",
        "password": "secure-pass-123",
    })
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["user"]["email"] == "julio@example.com"


def test_password_is_not_stored_as_plain_text(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "star_session.db")
    db.init_db()
    client = TestClient(main.app)
    response = client.post("/api/auth/register", json={
        "name": "Candidate",
        "email": "hash@example.com",
        "password": "secure-pass-123",
    })
    assert response.status_code == 200

    conn = db.get_conn()
    password_hash = conn.execute("SELECT password_hash FROM users WHERE email = ?", ("hash@example.com",)).fetchone()[0]
    conn.close()
    assert password_hash.startswith("scrypt$")
    assert password_hash != "secure-pass-123"
