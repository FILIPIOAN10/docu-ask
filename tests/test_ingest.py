import threading

import pytest
from sqlalchemy import func, select

from docuask.ingest import IngestConflict, ingest
from docuask.models import Chunk, Document

V1 = b"# Leave\n\n## Sick leave\n\nTen days are paid.\n\n## Carry-over\n\nFive days carry over.\n"
V2 = b"# Leave\n\n## Sick leave\n\nTwelve days are paid.\n"


def count(sessions, model):
    with sessions() as s:
        return s.scalar(select(func.count()).select_from(model))


def test_first_upload_creates_the_document_and_its_chunks(sessions, embeddings, settings):
    result = ingest(sessions, embeddings, settings, "leave.md", V1)
    assert result.status == "created"
    assert result.document.chunk_count == 2
    assert count(sessions, Chunk) == 2


def test_the_same_bytes_again_change_nothing_and_cost_no_embedding_call(sessions, embeddings, settings):
    ingest(sessions, embeddings, settings, "leave.md", V1)

    class Exploding:
        def embed_documents(self, texts):
            raise AssertionError("an unchanged file must not be embedded again")

    again = ingest(sessions, Exploding(), settings, "leave.md", V1)
    assert again.status == "unchanged"
    assert count(sessions, Chunk) == 2


def test_a_changed_file_replaces_its_chunks_instead_of_adding_to_them(sessions, embeddings, settings):
    first = ingest(sessions, embeddings, settings, "leave.md", V1)
    second = ingest(sessions, embeddings, settings, "leave.md", V2)
    assert second.status == "updated" and second.document.id == first.document.id
    assert count(sessions, Chunk) == 1
    with sessions() as s:
        assert "Twelve" in s.scalar(select(Chunk.content))


def test_deleting_a_document_deletes_its_chunks(sessions, embeddings, settings):
    ingest(sessions, embeddings, settings, "leave.md", V1)
    with sessions.begin() as s:
        s.delete(s.scalar(select(Document)))
    assert count(sessions, Chunk) == 0


def test_a_model_that_returns_the_wrong_width_is_caught_before_anything_is_written(sessions, settings):
    class WrongWidth:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    with pytest.raises(RuntimeError, match="1536"):
        ingest(sessions, WrongWidth(), settings, "leave.md", V1)
    assert count(sessions, Chunk) == 0


def test_two_uploads_of_a_new_name_at_the_same_time_leave_one_consistent_document(sessions, embeddings, settings):
    barrier = threading.Barrier(2)
    outcomes = []

    class Synced:
        """Makes both threads finish embedding before either one writes, so they really do race."""

        def embed_documents(self, texts):
            vectors = embeddings.embed_documents(texts)
            barrier.wait(timeout=10)
            return vectors

    def worker():
        try:
            outcomes.append(ingest(sessions, Synced(), settings, "race.md", V1).status)
        except IngestConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sorted(outcomes) in (["conflict", "created"], ["created", "updated"], ["created", "unchanged"])
    assert count(sessions, Document) == 1
    assert count(sessions, Chunk) == 2  # not 4: no duplicates
