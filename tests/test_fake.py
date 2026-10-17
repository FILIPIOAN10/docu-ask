import math

from langchain_core.prompts import ChatPromptTemplate

from docuask.fake import FakeChatModel, FakeEmbeddings
from docuask.rag import Draft


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_embeddings_have_the_requested_width_and_unit_length():
    vec = FakeEmbeddings(dim=64).embed_query("annual leave policy")
    assert len(vec) == 64
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-9)


def test_embeddings_are_deterministic():
    emb = FakeEmbeddings()
    assert emb.embed_query("sick leave") == emb.embed_query("sick leave")


def test_texts_that_share_words_are_closer_than_unrelated_ones():
    emb = FakeEmbeddings()
    q, near, far = (
        emb.embed_query(t) for t in ("sick leave days", "sick leave is paid for ten days", "office desk chair")
    )
    assert cosine(q, near) > cosine(q, far)


def test_stopwords_alone_embed_to_a_zero_vector_without_crashing():
    assert set(FakeEmbeddings(dim=8).embed_query("what is the")) == {0.0}


PROMPT = ChatPromptTemplate.from_messages([("human", "<context>\n{ctx}\n</context>\n<question>\n{q}\n</question>")])
CTX = "[1] leave.md > Sick leave\nSick leave is paid for 10 days per year. A certificate is needed.\n\n[2] other.md\nDesks are blue."


def test_fake_model_copies_the_best_passage_and_cites_it():
    structured = FakeChatModel().with_structured_output(Draft, include_raw=True)
    out = (PROMPT | structured).invoke({"ctx": CTX, "q": "How many days is sick leave paid?"})
    assert out["parsed"].covered is True
    assert out["parsed"].source_ids == [1]
    assert "10 days" in out["parsed"].answer


def test_fake_model_says_not_covered_when_nothing_matches():
    out = (PROMPT | FakeChatModel().with_structured_output(Draft, include_raw=True)).invoke(
        {"ctx": CTX, "q": "zebra giraffe safari"}
    )
    assert out["parsed"].covered is False and out["parsed"].source_ids == []
