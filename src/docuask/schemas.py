"""The API contract: what comes in and what goes out. Database rows never leave the service layer."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class UploadOut(DocumentOut):
    status: Literal["created", "updated", "unchanged"]


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    k: int = Field(default=4, ge=1, le=20)


class HitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: int
    document: str
    section: str | None
    page: int | None
    content: str
    distance: float | None
    score: float
    vector_rank: int | None
    keyword_rank: int | None


class SearchOut(BaseModel):
    best_distance: float | None
    hits: list[HitOut]


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class CitationOut(BaseModel):
    n: int
    document: str
    section: str | None
    page: int | None
    snippet: str


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AskOut(BaseModel):
    answer: str | None
    refused: bool
    #: why there is no answer: no_passages | too_far | not_covered | no_citation
    reason: str | None
    citations: list[CitationOut]
    usage: UsageOut
    best_distance: float | None
