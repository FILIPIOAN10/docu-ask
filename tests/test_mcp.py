"""The MCP door. In-memory first (a real MCP client talking to the server object, no network), then over real HTTP with
the bearer-token check, then the wiring in main.py."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError

from docuask.config import get_settings
from docuask.mcp_server import ApiKeyVerifier, create_mcp_server

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
}
MCP_ACCEPT = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mcp(rag, loaded):
    return create_mcp_server(lambda: rag, lambda: loaded)


@pytest.mark.anyio
async def test_the_client_discovers_two_read_only_tools_with_their_schemas(mcp):
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}

    assert set(tools) == {"ask_documents", "search_documents"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False
        assert tool.description and tool.output_schema  # the model chooses by the description; clients parse by schema
    assert tools["ask_documents"].input_schema["properties"]["question"]["maxLength"] == 1000
    assert tools["search_documents"].input_schema["properties"]["k"]["maximum"] == 20


@pytest.mark.anyio
async def test_ask_answers_with_citations_like_the_rest_api(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("ask_documents", {"question": "How many days is sick leave paid?"})

    answer = result.structured_content
    assert answer["refused"] is False and "10 days" in answer["answer"]
    assert answer["citations"][0]["document"] == "leave-policy.md"
    assert json.loads(result.content[0].text) == answer  # clients that only read text get the same JSON


@pytest.mark.anyio
async def test_ask_refuses_what_the_documents_do_not_cover(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("ask_documents", {"question": "zebra giraffe safari"})

    assert result.structured_content["refused"] is True and result.structured_content["answer"] is None


@pytest.mark.anyio
async def test_search_returns_passages_and_scores_without_calling_the_model(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("search_documents", {"query": "sick leave certificate", "k": 2})

    hits = result.structured_content["hits"]
    assert len(hits) == 2 and hits[0]["section"] == "Leave Policy > Sick leave"
    assert result.structured_content["best_distance"] is not None


@pytest.mark.anyio
async def test_bad_arguments_are_rejected_before_the_service_runs(mcp):
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="at least 3 characters"):
            await client.call_tool("ask_documents", {"question": "hi"})
        with pytest.raises(ToolError):
            await client.call_tool("search_documents", {"query": "leave", "k": 500})


@pytest.mark.anyio
async def test_an_unexpected_error_does_not_leak_its_details(loaded):
    def broken():
        raise RuntimeError("password=hunter2 in a connection string")

    async with Client(create_mcp_server(broken, lambda: loaded)) as client:
        result = await client.call_tool("ask_documents", {"question": "anything at all"}, raise_on_error=False)

    assert result.is_error and "hunter2" not in result.content[0].text


@pytest.mark.anyio
async def test_the_documents_resource_lists_what_can_be_asked_about(mcp):
    async with Client(mcp) as client:
        contents = await client.read_resource("docuask://documents")

    names = [d["name"] for d in json.loads(contents[0].text)]
    assert "leave-policy.md" in names and names == sorted(names)


# --- over HTTP ---------------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mcp_url(rag, loaded, settings):
    """A real uvicorn server in a thread: the client below goes through TCP, HTTP and the bearer check."""
    server_object = create_mcp_server(lambda: rag, lambda: loaded, auth=ApiKeyVerifier(lambda: settings.keys))
    app = server_object.http_app(path="/mcp")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.anyio
async def test_over_http_a_valid_key_gets_an_answer(mcp_url):
    async with Client(mcp_url, auth="old-key") as client:  # any listed key works: rotation without downtime
        result = await client.call_tool("ask_documents", {"question": "How many days is sick leave paid?"})

    assert result.structured_content["refused"] is False


@pytest.mark.anyio
async def test_over_http_no_key_or_a_wrong_key_is_401(mcp_url):
    import httpx

    async with httpx.AsyncClient() as http:
        missing = await http.post(mcp_url, json=INITIALIZE, headers=MCP_ACCEPT)
        wrong = await http.post(mcp_url, json=INITIALIZE, headers={**MCP_ACCEPT, "Authorization": "Bearer nope"})

    assert missing.status_code == 401 and wrong.status_code == 401
    assert missing.headers["www-authenticate"].startswith("Bearer")


def test_main_mounts_mcp_behind_the_same_keys_as_v1(monkeypatch):
    """The production wiring: /mcp/ inside the DocuAsk app, keys from API_KEYS. initialize needs no database."""
    monkeypatch.setenv("API_KEYS", "mcp-key")
    get_settings.cache_clear()
    try:
        from docuask.main import app

        with TestClient(app) as client:
            missing = client.post("/mcp/", json=INITIALIZE, headers=MCP_ACCEPT)
            ok = client.post("/mcp/", json=INITIALIZE, headers={**MCP_ACCEPT, "Authorization": "Bearer mcp-key"})
    finally:
        get_settings.cache_clear()

    assert missing.status_code == 401
    assert ok.status_code == 200 and '"serverInfo"' in ok.text and "DocuAsk" in ok.text
