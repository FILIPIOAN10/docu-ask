"""A tiny HTTP server that speaks the parts of the OpenAI REST API this project uses (/v1/embeddings and
/v1/chat/completions, with and without streaming). The REAL langchain-openai and openai client classes talk to it over
HTTP, so the tests prove our provider wiring, request shape and response parsing without a key, a network or a cost."""

import base64
import json
import socket
import struct
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from docuask.fake import FakeChatModel, FakeEmbeddings

app = FastAPI()
requests_seen: list[dict] = []  # what the client really sent, for the tests to inspect
_extractor = FakeChatModel()


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    requests_seen.append({"path": "/v1/embeddings", "body": body})
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    vectors = FakeEmbeddings().embed_documents(texts)
    if body.get("encoding_format") == "base64":
        data = [base64.b64encode(struct.pack(f"{len(v)}f", *v)).decode() for v in vectors]
    else:
        data = vectors
    return {
        "object": "list",
        "model": body["model"],
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(data)],
        "usage": {"prompt_tokens": 5 * len(texts), "total_tokens": 5 * len(texts)},
    }


def _completion(body: dict, content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content, "refusal": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    requests_seen.append({"path": "/v1/chat/completions", "body": body})
    if (
        request.headers.get("x-fail-first")
        and len([r for r in requests_seen if r["path"].endswith("completions")]) == 1
    ):
        return JSONResponse({"error": {"message": "temporary", "type": "server_error"}}, status_code=500)
    user_text = body["messages"][-1]["content"]
    covered, answer, ids = _extractor._extract(user_text)

    if body.get("response_format", {}).get("type") == "json_schema":
        payload = {"covered": covered, "answer": answer, "source_ids": ids}
        return _completion(body, json.dumps(payload))

    text = answer if covered else "The documents do not cover this."
    if not body.get("stream"):
        return _completion(body, text)

    def sse():
        base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        for i, token in enumerate(text.split(" ")):
            delta = {"content": token + " "} | ({"role": "assistant"} if i == 0 else {})
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


class FakeOpenAIServer:
    """Runs the app above on a free local port in a background thread."""

    def __enter__(self) -> "FakeOpenAIServer":
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        requests_seen.clear()
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"
