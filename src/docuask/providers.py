"""The two model clients, built from settings in ONE place. Every other module receives them, so switching provider
(fake -> OpenAI -> Azure OpenAI) is a configuration change, not a code change."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from docuask.config import Settings, get_settings
from docuask.fake import FakeChatModel, FakeEmbeddings


class ProviderConfigError(RuntimeError):
    """A setting the chosen provider needs is missing. Raised at startup so the pod fails fast, not on the first call."""


def _need(value, name: str):
    if not value:
        raise ProviderConfigError(f"{name} is required when LLM_PROVIDER is set to this provider")
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def make_embeddings(s: Settings) -> Embeddings:
    if s.llm_provider == "fake":
        return FakeEmbeddings(dim=s.embedding_dim)
    if s.llm_provider == "openai":
        return OpenAIEmbeddings(
            check_embedding_ctx_length=False,  # send plain strings: no tiktoken download, works with any compatible server
            model=s.embedding_model,
            api_key=_need(s.openai_api_key, "OPENAI_API_KEY"),
            base_url=s.openai_base_url,
            timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
    return AzureOpenAIEmbeddings(
        check_embedding_ctx_length=False,
        azure_deployment=_need(s.azure_embedding_deployment, "AZURE_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=_need(s.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT"),
        api_key=_need(s.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
        api_version=s.azure_openai_api_version,
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )


def make_chat_model(s: Settings) -> BaseChatModel:
    # No `temperature` on purpose: some current models reject it. Stability comes from a strict output schema and a
    # prompt that forbids outside knowledge, not from a sampling knob.
    if s.llm_provider == "fake":
        return FakeChatModel()
    if s.llm_provider == "openai":
        return ChatOpenAI(
            model=s.chat_model,
            api_key=_need(s.openai_api_key, "OPENAI_API_KEY"),
            base_url=s.openai_base_url,
            timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
    return AzureChatOpenAI(
        azure_deployment=_need(s.azure_chat_deployment, "AZURE_CHAT_DEPLOYMENT"),
        azure_endpoint=_need(s.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT"),
        api_key=_need(s.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
        api_version=s.azure_openai_api_version,
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    return make_embeddings(get_settings())


@lru_cache
def get_chat_model() -> BaseChatModel:
    return make_chat_model(get_settings())
