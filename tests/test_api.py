import json
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent.parent / "sample-docs" / "leave-policy.md"


def upload(client, name="leave-policy.md", data=None, **kw):
    data = SAMPLE.read_bytes() if data is None else data
    return client.post("/v1/documents", files={"file": (name, data, "text/markdown")}, **kw)


def test_health_endpoints_need_no_key(client):
    client.headers.pop("X-API-Key")
    assert client.get("/health/live").json() == {"status": "up"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_every_v1_route_requires_a_valid_key(client):
    client.headers.pop("X-API-Key")
    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.get("/v1/documents", headers={"X-API-Key": "old-key"}).status_code == 200
    )  # a rotated-out key still listed


def test_upload_list_reupload_and_delete(client):
    first = upload(client)
    assert first.status_code == 201 and first.json()["status"] == "created" and first.json()["chunk_count"] == 5
    assert upload(client).json()["status"] == "unchanged"

    listed = client.get("/v1/documents").json()
    assert [d["name"] for d in listed] == ["leave-policy.md"]

    doc_id = first.json()["id"]
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 204
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 404
    assert client.get("/v1/documents").json() == []


def test_bad_uploads_get_the_right_status_codes(client, settings):
    assert upload(client, "virus.exe", b"MZ").status_code == 415
    assert upload(client, "empty.txt", b"  ").status_code == 422
    big = b"x" * (settings.max_upload_bytes + 1)
    assert upload(client, "big.txt", big).status_code == 413


def test_ask_answers_with_citations_and_refuses_what_is_not_covered(client):
    upload(client)
    ok = client.post("/v1/ask", json={"question": "How many days is sick leave paid?"}).json()
    assert ok["refused"] is False and "10 days" in ok["answer"] and ok["citations"][0]["n"] == 1

    refused = client.post("/v1/ask", json={"question": "zebra giraffe safari"}).json()
    assert refused["refused"] is True and refused["answer"] is None


def test_ask_validates_its_input(client):
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/ask", json={}).status_code == 422
    assert client.post("/v1/ask", json={"question": "x" * 1001}).status_code == 422


def test_search_shows_the_scores_behind_an_answer(client):
    upload(client)
    body = client.post("/v1/search", json={"query": "sick leave certificate", "k": 2}).json()
    assert len(body["hits"]) == 2 and body["best_distance"] is not None
    assert body["hits"][0]["section"] == "Leave Policy > Sick leave"
    assert body["hits"][0]["vector_rank"] and body["hits"][0]["keyword_rank"]


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_streaming_endpoint_sends_server_sent_events(client):
    upload(client)
    response = client.post("/v1/ask/stream", json={"question": "How many days is sick leave paid?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[-1][0] == "done" and events[-1][1]["citations"]
    assert sum(1 for n, _ in events if n == "token") > 3


def test_every_response_carries_a_request_id_and_keeps_the_callers_id(client):
    assert client.get("/health/live").headers["X-Request-ID"]
    assert client.get("/health/live", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"
