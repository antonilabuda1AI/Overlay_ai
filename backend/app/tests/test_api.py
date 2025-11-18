from fastapi.testclient import TestClient

from app.main import app


def test_health_and_session_flow(tmp_path, monkeypatch):
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"]

    r = client.post("/session/start")
    sid = r.json()["session_id"]
    assert sid

    # add a note
    r = client.post("/note", json={"session_id": sid, "text": "hello world"})
    assert r.status_code == 200

    # query timeline
    r = client.get("/timeline", params={"session_id": sid})
    items = r.json()
    assert len(items) >= 1

    # ask (uses local stub embeddings/chat if no API key)
    r = client.post("/ask", json={"session_id": sid, "question": "what is said?"})
    assert r.status_code == 200
    j = r.json()
    assert "answer" in j and "contexts" in j

    # export
    r = client.post("/export", json={"session_id": sid, "format": "json"})
    assert r.status_code == 200
    assert r.json()["path"].endswith(".json")

    r = client.post("/session/stop", params={"session_id": sid})
    assert r.status_code == 200

