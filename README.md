# DocuAsk

[![CI](https://github.com/FILIPIOAN10/docu-ask/actions/workflows/ci.yml/badge.svg)](https://github.com/FILIPIOAN10/docu-ask/actions/workflows/ci.yml)

A small, carefully engineered **RAG service**: upload company documents, ask questions in natural language, get an answer
**with citations** - or an explicit refusal when the documents do not cover the question.

```
upload (.md/.txt/.pdf)                       ask
        |                                      |
   chunk by structure                    embed the question
   embed (one call per file)             vector search  +  keyword search (Postgres FTS)
        |                                      \              /
   Postgres + pgvector  <------------------    fuse with Reciprocal Rank Fusion
   (HNSW index, GIN index)                           |
                                          gate: nearest passage too far?  -> refuse
                                                     |
                                          LLM (strict JSON: covered, answer, source_ids)
                                                     |
                                          citations validated -> answer + sources
```

Stack: Python 3.12, FastAPI, LangChain, SQLAlchemy + Alembic, PostgreSQL 16 with pgvector, `uv`, Docker, GitHub Actions.

## Run it (no API key needed)

```bash
docker compose up -d --build            # Postgres + pgvector, migrations, API on :8000
for f in sample-docs/*.md; do curl -s -H "X-API-Key: dev-key" -F "file=@$f" localhost:8000/v1/documents; echo; done
curl -s -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
     -d '{"question":"How many days is sick leave paid?"}' localhost:8000/v1/ask
```

On Windows PowerShell use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`) and send the JSON body on stdin,
because PowerShell 5.1 strips the quotes inside `-d '{...}'`:
`'{"question":"..."}' | curl.exe -s -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d "@-" localhost:8000/v1/ask`

`LLM_PROVIDER=fake` (the default) uses an offline lexical embedder and an extractive "model", so everything runs without
a key. Set `LLM_PROVIDER=openai` or `azure` (see `.env.example`) for real models; there are no code changes.

## Develop

```bash
uv sync
docker run -d --name docuask-db -e POSTGRES_USER=docuask -e POSTGRES_PASSWORD=docuask -e POSTGRES_DB=docuask \
  -p 5434:5432 pgvector/pgvector:pg16
uv run alembic upgrade head
uv run python -m docuask.cli ingest sample-docs
uv run uvicorn docuask.main:app --reload
uv run pytest --cov=docuask             # needs Docker: the tests start a real Postgres with Testcontainers
uv run python -m docuask.evaluate       # retrieval quality on the golden set
```

## API

| Method | Path | What |
|---|---|---|
| POST | `/v1/documents` | upload a file; idempotent by content hash (`created` / `updated` / `unchanged`) |
| GET / DELETE | `/v1/documents[/{id}]` | list / delete (chunks cascade) |
| POST | `/v1/search` | retrieval only, with scores - the first thing to look at when an answer is wrong |
| POST | `/v1/ask` | grounded answer + citations, or `refused` with a `reason` |
| POST | `/v1/ask/stream` | the same as server-sent events (`token` ... `done`) |
| GET | `/health/live`, `/health/ready` | liveness (process) / readiness (database + pgvector + migrated schema) |

Auth: `X-API-Key` header, several keys allowed (rotation), constant-time comparison.

## Design decisions

- **Hybrid retrieval.** Embeddings find meaning, Postgres full-text finds exact codes and numbers; Reciprocal Rank Fusion
  merges the two ranked lists without calibrating their scores. It is a hypothesis to measure, not a free win: see the
  evaluation below.
- **Three independent "I don't know" filters:** a distance gate before the model (free), the model's own `covered` flag,
  and a citation check (an answer that cites nothing real is discarded).
- **Embedding outside the transaction.** The slow network call never holds a database lock or a pooled connection.
- **Idempotent ingestion** by SHA-256, so re-uploading a file costs nothing; a changed file replaces its chunks in one
  short transaction. Two concurrent uploads of the same new name leave one consistent document.
- **Provider seam.** One factory (`providers.py`) builds the model clients; fake / OpenAI / Azure OpenAI is a setting.
- **Fail fast.** Missing keys or a wrong embedding width stop the process at startup, not on the first request.
- **Migrations as a separate job** (`docker compose`: `migrate` runs once and exits; the API starts only after it succeeded).
- **Tests use the real database** (Testcontainers + pgvector) and the real OpenAI client classes against a local fake
  HTTP server, so wiring bugs show up without a key or a bill.

## Evaluation

28 questions written by me on the 5 sample documents: 22 answerable (paraphrased, e.g. "vacation days" for "annual
leave") and 6 the documents do not cover. `python -m docuask.evaluate` reports, per search mode, **hit@1** (the right
section is the first result), **hit@3** and **MRR**.

| search | fake embeddings: hit@1 / MRR | text-embedding-3-small: hit@1 / MRR |
|---|---|---|
| vector | 77 % / 0.88 | **100 % / 1.00** |
| keyword | 86 % / 0.93 | 86 % / 0.93 |
| hybrid | 91 % / 0.95 | 95 % / 0.98 |

hit@3 is 100 % everywhere. With the offline lexical embedder, hybrid is the best mode; **with a real embedding model,
vector search alone is best on this set**. The one question hybrid loses (hit@1 95 %) is a rank tie in the fusion that is
broken by chunk id; breaking ties by the vector rank would be the next thing to try and measure.

**Refusal gate** (nearest vector distance, `text-embedding-3-small`): answerable questions fall between 0.25 and 0.68,
unanswerable ones between 0.67 and 0.92. They overlap by 0.012, so no threshold separates them perfectly.
`MAX_DISTANCE=0.72` lets all 22 answerable questions through and refuses 5 of the 6 unanswerable ones before any model
call; the model's `covered` flag and the citation check handle the rest. The default `0.95` is deliberately loose
for the fake embedder, whose distances do not separate the questions at all.

The golden set is small and written by me, on documents I know: use it as a regression signal, not as proof of quality.
CI fails the build if hybrid hit@3 drops below 90 %.

## Quality gates

`ruff` (lint + format), `pytest` (60 tests, about 90 % coverage, floor 80 %), the retrieval quality gate above, and a
Docker image job (build, start, health check), all in GitHub Actions.

## What was verified, and what was not

Verified: the full test suite on a real Postgres; CI green on GitHub Actions; the stack through `docker compose`
(non-root image, about 295 MB); ingestion, a 28-question evaluation and several questions with `text-embedding-3-small`
and `gpt-4o-mini`.

Not verified: Azure OpenAI against a real endpoint (only the client construction and a protocol-level contract test);
latency or behaviour under load; real-world scanned or complex PDFs (the PDF test generates its own file); other
languages; image vulnerability scanning.

## Known limits

- Full-text search uses the `english` configuration; other languages need their own configuration (or the vector leg only).
- Scanned PDFs have no text layer (no OCR).
- Streaming cannot un-send tokens: the final `done` event carries the citation verdict and the client must honour it, and it does not report token usage.
- The distance threshold was calibrated on the bundled documents and 28 questions; recalibrate for yours.
- No per-tenant isolation yet: one set of documents, shared by every API key. A single static API key list instead of signed tokens.
- No tracing or metrics, only request-id logs.