import pytest

from docuask.retrieval import Hit, HybridRetriever, fuse, keyword_candidates, search


def make_hit(i, **kw):
    return Hit(chunk_id=i, document_id=None, document="d.md", section=None, page=None, content="c", **kw)


def test_fuse_puts_a_chunk_found_by_both_searches_first():
    vec = [make_hit(1, distance=0.1, vector_rank=1), make_hit(2, distance=0.2, vector_rank=2)]
    kw = [make_hit(3, keyword_rank=1), make_hit(2, keyword_rank=2)]
    fused = fuse(vec, kw, k=3)
    assert [h.chunk_id for h in fused] == [2, 1, 3]
    assert fused[0].vector_rank == 2 and fused[0].keyword_rank == 2 and fused[0].distance == 0.2


def test_fuse_respects_k_and_is_stable_on_ties():
    vec = [make_hit(i, vector_rank=i) for i in range(1, 6)]
    assert [h.chunk_id for h in fuse(vec, [], k=2)] == [1, 2]


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_every_mode_finds_the_sick_leave_section(loaded, embeddings, mode):
    with loaded() as session:
        found = search(session, embeddings, "How many days is sick leave paid?", mode=mode, k=3)
    assert found.hits[0].document == "leave-policy.md"
    assert "Sick leave" in found.hits[0].section


def test_keyword_search_finds_an_exact_code(loaded):
    with loaded() as session:
        hits = keyword_candidates(session, "SEC", 5)
    assert hits and hits[0].document == "information-security.md"


@pytest.mark.parametrize("nasty", ["'; DROP TABLE chunks; --", "a & b | !c :* (d)", '"quoted" <b>', "!!!", "   "])
def test_user_text_can_never_break_the_keyword_query(loaded, embeddings, nasty):
    with loaded() as session:
        search(session, embeddings, nasty, mode="hybrid")  # must not raise
        assert keyword_candidates(session, "annual leave", 3)  # and the table is still there


def test_best_distance_is_the_nearest_vector_candidate(loaded, embeddings):
    with loaded() as session:
        near = search(session, embeddings, "sick leave certificate", mode="hybrid").best_distance
        far = search(session, embeddings, "zebra giraffe safari", mode="hybrid").best_distance
        assert near < far
        assert search(session, embeddings, "sick leave", mode="keyword").best_distance is None


def test_the_langchain_retriever_returns_documents_with_metadata(loaded, embeddings):
    retriever = HybridRetriever(sessions=loaded, embeddings=embeddings, k=2)
    docs = retriever.invoke("How many days of annual leave do I get?")
    assert len(docs) == 2
    assert docs[0].metadata["document"] == "leave-policy.md"
    assert "25 working days" in docs[0].page_content
