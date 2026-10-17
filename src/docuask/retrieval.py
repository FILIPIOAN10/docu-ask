"""Finding the passages: vector search, keyword search, and the fusion of both.

vector   meaning:   "How many days off do I get?" finds the "annual leave" section even with no shared word
keyword  exact:     "GDPR", "IBAN", "SEC-4411" - names, codes and numbers that embeddings blur
hybrid   both, fused with Reciprocal Rank Fusion: a passage ranked high by EITHER list floats to the top,
         and one ranked high by both is first. RRF uses ranks, not scores, so no calibration between the two scales.
"""

import re
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from docuask.models import Chunk
from docuask.models import Document as DocumentRow

Mode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60  # the constant from the original RRF paper; it flattens the advantage of rank 1 over rank 2


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: UUID
    document: str
    section: str | None
    page: int | None
    content: str
    distance: float | None = None  # cosine distance, if the vector search found it (0 = identical)
    vector_rank: int | None = None
    keyword_rank: int | None = None
    score: float = 0.0


@dataclass(frozen=True)
class Found:
    hits: list[Hit]
    #: nearest cosine distance among the vector candidates, whether or not it made the final cut; None if not computed
    best_distance: float | None


_COLUMNS = (Chunk.id, Chunk.document_id, DocumentRow.name, Chunk.section, Chunk.page, Chunk.content)


def _hit(row: Any, **extra: Any) -> Hit:
    return Hit(row[0], row[1], row[2], row[3], row[4], row[5], **extra)


def vector_candidates(session: Session, query_vector: list[float], limit: int) -> list[Hit]:
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(*_COLUMNS, distance)
        .join(DocumentRow, DocumentRow.id == Chunk.document_id)
        .order_by(distance, Chunk.id)
        .limit(limit)
    )
    return [_hit(r, distance=float(r.distance), vector_rank=i) for i, r in enumerate(session.execute(stmt), start=1)]


def keyword_candidates(session: Session, text: str, limit: int) -> list[Hit]:
    # A question is not a boolean query: "what is the sick leave policy" must match chunks that have SOME of the
    # words, ranked by how many. We build `sick | leave | policy` ourselves from \w+ tokens only, so no user input
    # reaches the tsquery syntax as an operator. Stop words ("what", "is") are dropped by the 'english' config.
    terms = list(dict.fromkeys(re.findall(r"\w+", text.lower())))
    if not terms:
        return []
    query = func.to_tsquery("english", " | ".join(terms))
    rank = func.ts_rank_cd(Chunk.tsv, query).label("rank")
    stmt = (
        select(*_COLUMNS, rank)
        .join(DocumentRow, DocumentRow.id == Chunk.document_id)
        .where(Chunk.tsv.op("@@")(query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [_hit(r, keyword_rank=i) for i, r in enumerate(session.execute(stmt), start=1)]


def fuse(vector_hits: list[Hit], keyword_hits: list[Hit], k: int) -> list[Hit]:
    merged: dict[int, Hit] = {}
    for hit in [*vector_hits, *keyword_hits]:
        previous = merged.get(hit.chunk_id)
        merged[hit.chunk_id] = (
            hit
            if previous is None
            else replace(
                previous,
                distance=previous.distance if previous.distance is not None else hit.distance,
                vector_rank=previous.vector_rank or hit.vector_rank,
                keyword_rank=previous.keyword_rank or hit.keyword_rank,
            )
        )
    scored = [
        replace(
            h,
            score=sum(1.0 / (RRF_K + r) for r in (h.vector_rank, h.keyword_rank) if r is not None),
        )
        for h in merged.values()
    ]
    return sorted(scored, key=lambda h: (-h.score, h.chunk_id))[:k]


def search(
    session: Session, embeddings: Embeddings, question: str, *, mode: Mode = "hybrid", k: int = 4, candidate_k: int = 20
) -> Found:
    vector_hits: list[Hit] = []
    keyword_hits: list[Hit] = []
    if mode in ("vector", "hybrid"):
        vector_hits = vector_candidates(session, embeddings.embed_query(question), candidate_k)
    if mode in ("keyword", "hybrid"):
        keyword_hits = keyword_candidates(session, question, candidate_k)
    best = min((h.distance for h in vector_hits if h.distance is not None), default=None)
    if mode == "vector":
        hits = vector_hits[:k]
    elif mode == "keyword":
        hits = keyword_hits[:k]
    else:
        hits = fuse(vector_hits, keyword_hits, k)
    return Found(hits, best)


class HybridRetriever(BaseRetriever):
    """The same search as a standard LangChain retriever, so it can go into any chain, agent or evaluation tool
    that expects `retriever.invoke(question) -> list[Document]`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sessions: Any
    embeddings: Any
    mode: Mode = "hybrid"
    k: int = 4
    candidate_k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        sessions: sessionmaker[Session] = self.sessions
        with sessions() as session:
            found = search(session, self.embeddings, query, mode=self.mode, k=self.k, candidate_k=self.candidate_k)
        return [
            Document(
                page_content=h.content,
                metadata={
                    "chunk_id": h.chunk_id,
                    "document": h.document,
                    "section": h.section,
                    "page": h.page,
                    "distance": h.distance,
                    "score": h.score,
                },
            )
            for h in found.hits
        ]
