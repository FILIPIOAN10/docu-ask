"""Ingestion: bytes in, searchable chunks out - and running it twice on the same file changes nothing.

sha256 same as stored?  -> "unchanged" (no embedding call, no cost)
else  chunk -> embed (slow, network, NO database transaction open) -> one short transaction that replaces the chunks
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from docuask.chunking import chunk_sections, load
from docuask.config import Settings
from docuask.models import EMBEDDING_DIM, Chunk, Document

log = logging.getLogger(__name__)


class IngestConflict(RuntimeError):
    """Another request is ingesting a document with the same name right now."""


@dataclass(frozen=True)
class IngestResult:
    document: Document
    status: Literal["created", "updated", "unchanged"]


def ingest(
    sessions: sessionmaker[Session], embeddings: Embeddings, settings: Settings, name: str, data: bytes
) -> IngestResult:
    digest = hashlib.sha256(data).hexdigest()
    with sessions() as session:
        existing = session.scalar(select(Document).where(Document.name == name))
        if existing is not None and existing.sha256 == digest:
            return IngestResult(existing, "unchanged")

    drafts = chunk_sections(load(name, data), settings.chunk_size, settings.chunk_overlap)
    vectors = embeddings.embed_documents([d.embed_text for d in drafts])
    if len(vectors) != len(drafts) or any(len(v) != EMBEDDING_DIM for v in vectors):
        raise RuntimeError(f"The embedding model must return {EMBEDDING_DIM} numbers per text")

    try:
        with sessions.begin() as session:
            doc = session.scalar(select(Document).where(Document.name == name).with_for_update())
            status: Literal["created", "updated"] = "updated"
            if doc is None:
                doc = Document(name=name, sha256=digest)
                session.add(doc)
                session.flush()
                status = "created"
            else:
                session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            doc.sha256 = digest
            doc.chunk_count = len(drafts)
            doc.updated_at = datetime.now(UTC)
            session.add_all(
                Chunk(document_id=doc.id, ord=d.ord, content=d.content, section=d.section, page=d.page, embedding=v)
                for d, v in zip(drafts, vectors, strict=True)
            )
    except IntegrityError as e:  # two uploads of a brand-new name raced; the unique constraint picked the loser
        raise IngestConflict(f"'{name}' is being ingested by another request, retry in a moment") from e
    log.info("ingested name=%s status=%s chunks=%d", name, status, len(drafts))
    return IngestResult(doc, status)
