"""The OpenAI and Azure code paths, exercised with the real client libraries against a local fake server."""

import pytest

from docuask.config import Settings
from docuask.fake import FakeChatModel, FakeEmbeddings
from docuask.providers import ProviderConfigError, make_chat_model, make_embeddings
from docuask.rag import build_chain, build_stream_chain
from docuask.retrieval import Found, Hit
from tests import fake_openai
from tests.fake_openai import FakeOpenAIServer


@pytest.fixture(scope="module")
def server():
    with FakeOpenAIServer() as s:
        yield s


@pytest.fixture
def openai_settings(server):
    return Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sk-test",
        openai_base_url=server.base_url,
        chat_model="test-chat-model",
        embedding_model="test-embed-model",
        llm_max_retries=2,
    )


def test_fake_provider_needs_nothing():
    s = Settings(_env_file=None, llm_provider="fake")
    assert isinstance(make_embeddings(s), FakeEmbeddings) and isinstance(make_chat_model(s), FakeChatModel)


def test_openai_embeddings_send_text_and_parse_the_vectors(openai_settings, server):
    vectors = make_embeddings(openai_settings).embed_documents(["sick leave", "annual leave"])
    assert len(vectors) == 2 and len(vectors[0]) == 1536
    sent = [r for r in fake_openai.requests_seen if r["path"] == "/v1/embeddings"][-1]["body"]
    assert sent["model"] == "test-embed-model"
    assert sent["input"] == ["sick leave", "annual leave"]  # plain text, not pre-tokenised ids


def test_openai_chat_runs_the_whole_chain_with_structured_output_and_token_usage(openai_settings):
    chain = build_chain(make_chat_model(openai_settings))
    passages = "[1] leave.md > Sick leave\nSick leave is paid for 10 days per year. A certificate is needed."
    hit = Hit(1, None, "leave.md", "Sick leave", None, passages.split("\n", 1)[1])
    out = chain.invoke(
        {"question": "How many days is sick leave paid?", "hits": [hit], "found": Found([hit], 0.2)},
    )
    assert out.refused is False and out.citations[0].n == 1 and "10 days" in out.answer
    assert (out.usage.input_tokens, out.usage.output_tokens) == (120, 30)  # read from the response's usage block

    sent = [r for r in fake_openai.requests_seen if r["path"] == "/v1/chat/completions"][-1]["body"]
    assert sent["model"] == "test-chat-model"
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in sent  # some models reject it, so we never send it
    assert "<question>" in sent["messages"][-1]["content"]


def test_openai_streaming_yields_tokens(openai_settings):
    hit = Hit(1, None, "leave.md", "Sick leave", None, "Sick leave is paid for 10 days per year.")
    tokens = list(
        build_stream_chain(make_chat_model(openai_settings)).stream(
            {"question": "How many days is sick leave paid?", "hits": [hit]}
        )
    )
    assert len(tokens) > 3 and "10 days" in "".join(tokens)


def test_a_missing_key_fails_at_construction_not_on_the_first_request():
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        make_chat_model(Settings(_env_file=None, llm_provider="openai"))
    with pytest.raises(ProviderConfigError, match="AZURE_OPENAI_ENDPOINT"):
        make_embeddings(
            Settings(_env_file=None, llm_provider="azure", azure_embedding_deployment="emb", azure_openai_api_key="k")
        )


def test_azure_clients_are_built_from_deployment_names_not_model_names():
    s = Settings(
        _env_file=None,
        llm_provider="azure",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="k",
        azure_chat_deployment="my-chat",
        azure_embedding_deployment="my-embed",
    )
    chat, emb = make_chat_model(s), make_embeddings(s)
    assert chat.deployment_name == "my-chat"
    assert emb.deployment == "my-embed"
