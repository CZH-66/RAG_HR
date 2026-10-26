# tests/test_api.py
# -*- coding: utf-8 -*-
import json


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_health(api_env):
    client, _ = api_env
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_returns_answer_and_sources(api_env):
    client, _ = api_env
    resp = client.post("/query", json={"question": "working hours", "lang": "zh"})
    assert resp.status_code == 200
    data = resp.json()
    assert "fake answer" in data["answer"]
    assert data["sources"][0]["doc_id"] == 1


def test_query_log_contains_review_fields(api_env):
    client, log_file = api_env
    client.post("/query", json={"question": "q1", "lang": "en"})
    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["endpoint"] == "/query"
    assert record["lang"] == "en"
    assert record["rewritten_question"] is None
    assert record["prompt_version"] == "test"
    assert record["model"]
    assert record["sources"][0]["score"] == 0.8


def test_chat_stream_events(api_env):
    client, _ = api_env
    with client.stream("POST", "/chat", json={"message": "hi", "lang": "zh"}) as resp:
        assert resp.status_code == 200
        events = _parse_sse("".join(resp.iter_text()))
    assert events[0]["session_id"]
    assert any("delta" in event for event in events)
    assert events[-1]["done"] is True
    assert events[-1]["sources"]


def test_chat_log_records_rewritten_question(api_env):
    client, log_file = api_env
    resp = client.post("/chat", json={"message": "那需要审批吗", "session_id": "s1", "lang": "zh"})
    assert resp.status_code == 200
    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["rewritten_question"] == "rewritten: 那需要审批吗"
    assert record["answer"] == "你好，我是 HR 助手"


def test_reset_clears_sessions_of_both_languages(api_env):
    client, _ = api_env
    client.post("/chat", json={"message": "a", "session_id": "s2", "lang": "zh"})
    client.post("/chat", json={"message": "b", "session_id": "s2", "lang": "en"})
    assert client.get("/health").json()["sessions"] == 2

    resp = client.post("/reset", json={"session_id": "s2"})
    assert resp.json()["removed"] == 2
    assert client.get("/health").json()["sessions"] == 0