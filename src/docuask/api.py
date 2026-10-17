"""HTTP layer: thin on purpose. Validation, auth and status codes here; the work lives in ingest.py / rag.py."""

import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.embeddings import Embeddings
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from docuask.chunking import EmptyDocument, UnsupportedDocument
from docuask.config import Settings, get_settings
from docuask.db import get_session, get_session_factory
from docuask.ingest import IngestConflict, ingest
from docuask.models import Document
from docuask.providers import get_chat_model, get_embeddings
from docuask.rag import RagService
from docuask.retrieval import search
from docuask.schemas import AskIn, AskOut, DocumentOut, HitOut, SearchIn, SearchOut, UploadOut
from docuask.security import require_api_key

protected = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
health = APIRouter(prefix="/health", tags=["health"])


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_settings(), get_session_factory(), get_embeddings(), get_chat_model())


Sessions = Annotated[sessionmaker[Session], Depends(get_session_factory)]
Emb = Annotated[Embeddings, Depends(get_embeddings)]
Cfg = Annotated[Settings, Depends(get_settings)]
Rag = Annotated[RagService, Depends(get_rag_service)]


@protected.post("/documents", response_model=UploadOut, status_code=status.HTTP_201_CREATED, tags=["documents"])
def upload_document(file: UploadFile, sessions: Sessions, embeddings: Emb, settings: Cfg) -> UploadOut:
    if not file.filename:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "A file name is required")
    # Read one byte more than the limit: if we get it, the file is too big, and we never hold more than the limit.
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Maximum size is {settings.max_upload_bytes} bytes")
    try:
        result = ingest(sessions, embeddings, settings, file.filename, data)
    except UnsupportedDocument as e:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)) from e
    except EmptyDocument as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    except IngestConflict as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    return UploadOut(**DocumentOut.model_validate(result.document).model_dump(), status=result.status)


@protected.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(session: Annotated[Session, Depends(get_session)]) -> list[Document]:
    return list(session.scalars(select(Document).order_by(Document.name)))


@protected.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(document_id: UUID, session: Annotated[Session, Depends(get_session)]) -> None:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document")
    session.delete(doc)  # the chunks go with it: ON DELETE CASCADE
    session.commit()


@protected.post("/search", response_model=SearchOut, tags=["rag"])
def search_passages(
    body: SearchIn, session: Annotated[Session, Depends(get_session)], embeddings: Emb, settings: Cfg
) -> SearchOut:
    """Debug view of retrieval alone: which passages, with which scores. Invaluable when an answer looks wrong."""
    found = search(session, embeddings, body.query, mode=body.mode, k=body.k, candidate_k=settings.candidate_k)
    return SearchOut(best_distance=found.best_distance, hits=[HitOut.model_validate(h) for h in found.hits])


@protected.post("/ask", response_model=AskOut, tags=["rag"])
def ask(body: AskIn, rag: Rag) -> AskOut:
    return rag.ask(body.question)


@protected.post("/ask/stream", tags=["rag"])
def ask_stream(body: AskIn, rag: Rag) -> StreamingResponse:
    found = rag.retrieve(body.question)  # database work finishes here, before the stream starts

    def events() -> Iterator[str]:
        for name, payload in rag.stream(body.question, found):
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@health.get("/live")
def live() -> dict[str, str]:
    """Liveness: the process is up. Must not touch the database, or a DB outage would restart healthy pods."""
    return {"status": "up"}


@health.get("/ready")
def ready(session: Annotated[Session, Depends(get_session)]) -> dict[str, str]:
    """Readiness: can this instance serve traffic? Needs the database, the pgvector extension AND the migrated schema,
    so a pod that starts before the migration job finished stays out of the load balancer until the tables exist."""
    try:
        ok = session.execute(
            text(
                "SELECT to_regclass('public.chunks') IS NOT NULL "
                "AND EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
        ).scalar_one()
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database not ready") from e
    if not ok:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "schema not migrated")
    return {"status": "ready"}
