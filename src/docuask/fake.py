"""Offline stand-ins for the two model calls, so tests, CI and the first demo need no key and no network.

FakeEmbeddings  - hashed bag of words: texts that share words get vectors with a high cosine similarity. It is a lexical
                  model, so it cannot match paraphrases ("holiday" vs "leave"); a real model can. That difference is
                  exactly what python -m docuask.evaluate lets you measure.
FakeChatModel   - extractive: it finds the passage that shares most words with the question and copies its first
                  sentences, citing it. Deterministic, so a test can assert the exact answer.
"""

import hashlib
import math
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "can",
        "could",
        "i",
        "you",
        "we",
        "they",
        "it",
        "my",
        "our",
        "your",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "why",
        "that",
        "this",
        "these",
        "those",
        "with",
        "at",
        "by",
        "from",
        "as",
        "if",
        "not",
        "no",
        "there",
        "about",
        "any",
    ]
)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def _bucket(word: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
    n = int.from_bytes(digest, "big")
    return n % dim, 1.0 if (n >> 63) & 1 else -1.0  # the sign spreads collisions around zero


class FakeEmbeddings(Embeddings):
    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in words(text):
            i, sign = _bucket(w, self.dim)
            vec[i] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_PASSAGE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.S | re.M)
_QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)


class FakeChatModel(BaseChatModel):
    """Reads the numbered passages and the question out of the last message, exactly as a real model would."""

    #: the passage must share at least this share of the question's words, or the fake says "not covered"
    min_overlap: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "fake-extractive"

    def _extract(self, text: str) -> tuple[bool, str, list[int]]:
        question = (_QUESTION.search(text) or [None, ""])[1]
        asked = set(words(question))
        best_n, best_text, best_score = 0, "", 0
        for m in _PASSAGE.finditer(text.split("<question>")[0]):
            # the label line ("file > section") is part of what a real model reads too
            score = len(asked & set(words(m.group(0))))
            if score > best_score:  # ties keep the earlier passage = the one retrieval ranked higher
                best_n, best_text, best_score = int(m.group(1)), m.group(2).strip(), score
        if best_score == 0 or best_score < self.min_overlap * len(asked):
            return False, "", []
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(best_text.split()))
        # the two sentences that share most words with the question, kept in their original order
        ranked = sorted(range(len(sentences)), key=lambda i: (-len(asked & set(words(sentences[i]))), i))[:2]
        return True, " ".join(sentences[i] for i in sorted(ranked)) + f" [{best_n}]", [best_n]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        covered, answer, _ = self._extract(str(messages[-1].content))
        text = answer if covered else "The documents do not cover this."
        message = AIMessage(content=text, usage_metadata=self._usage(messages, text))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        covered, answer, _ = self._extract(str(messages[-1].content))
        for token in (answer if covered else "The documents do not cover this.").split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token + " "))

    @staticmethod
    def _usage(messages: list[BaseMessage], output: str) -> dict[str, int]:
        n_in = sum(len(str(m.content)) for m in messages) // 4
        n_out = len(output) // 4
        return {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Runnable:
        """Same contract as a real model: a runnable that returns {"raw", "parsed", "parsing_error"}."""

        def run(prompt_value: Any) -> dict[str, Any]:
            messages = prompt_value.to_messages()
            covered, answer, ids = self._extract(str(messages[-1].content))
            parsed = schema(covered=covered, answer=answer, source_ids=ids)
            raw = AIMessage(content=parsed.model_dump_json(), usage_metadata=self._usage(messages, answer))
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(run)
