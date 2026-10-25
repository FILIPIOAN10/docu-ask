"""Application entry point:  uvicorn docuask.main:app"""

import contextvars
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from docuask.api import get_rag_service, health, protected
from docuask.config import get_settings
from docuask.db import get_session_factory
from docuask.mcp_server import ApiKeyVerifier, create_mcp_server
from docuask.models import EMBEDDING_DIM
from docuask.providers import get_chat_model, get_embeddings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Stamps every log line with the id of the request being served, so one request can be followed through the logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RequestIdFilter())
log = logging.getLogger("docuask")
# The MCP door, over HTTP. Same keys as /v1, read per request (rotation works the same way), sent as a bearer token.
mcp_http = create_mcp_server(
    get_rag_service, get_session_factory, auth=ApiKeyVerifier(lambda: get_settings().keys)
).http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Fail fast: a wrong setting should stop the pod at start, not break the first user's request."""
    settings = get_settings()
    if settings.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(f"EMBEDDING_DIM={settings.embedding_dim} but the database column is vector({EMBEDDING_DIM})")
    get_embeddings()  # builds the clients, so a missing API key raises here
    get_chat_model()
    log.info("started provider=%s", settings.llm_provider)
    async with mcp_http.lifespan(app):
        yield


app = FastAPI(title="DocuAsk", version="0.1.0", lifespan=lifespan)
app.include_router(protected)
app.include_router(health)
# Mounted under /mcp, so the endpoint is /mcp/ - WITH the trailing slash (/mcp answers 307 to it).
app.mount("/mcp", mcp_http)
STATIC = Path(__file__).parent / "static"
# The page only talks to this same origin and renders every piece of text with textContent, never innerHTML.
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
        "img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """A small page with buttons for the API: upload, list, delete, ask, show passages."""
    return FileResponse(STATIC / "index.html", headers=PAGE_HEADERS)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    if not request.url.path.startswith("/health"):
        log.info(
            "%s %s -> %d in %d ms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """The client gets a generic message; the details (and any secret inside them) stay in our logs."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "Internal error"}, status_code=500)
