"""Shared fixtures. The database is REAL Postgres + pgvector: started in Docker by Testcontainers (the Python twin of the
Java library of the same name), or - in CI, where a service container already runs - taken from TEST_DATABASE_URL.
Migrations are applied exactly as in production, so a test also proves the migration works."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from docuask.api import get_rag_service
from docuask.config import Settings, get_settings
from docuask.db import get_session, get_session_factory, make_engine
from docuask.fake import FakeChatModel, FakeEmbeddings
from docuask.ingest import ingest
from docuask.main import app
from docuask.providers import get_chat_model, get_embeddings
from docuask.rag import RagService

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample-docs"


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("pgvector/pgvector:pg16", driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def engine(database_url):
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")
    engine = make_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def sessions(engine):
    """A session factory over an EMPTY database: every test starts from nothing."""
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        session.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))
        session.commit()
    return factory


@pytest.fixture
def settings(database_url):
    return Settings(_env_file=None, database_url=database_url, llm_provider="fake", api_keys="test-key,old-key")


@pytest.fixture
def embeddings(settings):
    return FakeEmbeddings(dim=settings.embedding_dim)


@pytest.fixture
def rag(settings, sessions, embeddings):
    return RagService(settings, sessions, embeddings, FakeChatModel())


@pytest.fixture
def loaded(sessions, embeddings, settings):
    """The five sample documents, ingested. Returns the session factory."""
    for path in sorted(SAMPLE_DOCS.glob("*.md")):
        ingest(sessions, embeddings, settings, path.name, path.read_bytes())
    return sessions


@pytest.fixture
def client(settings, sessions, embeddings, rag):
    def override_session():
        with sessions() as session:
            yield session

    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_session: override_session,
            get_session_factory: lambda: sessions,
            get_embeddings: lambda: embeddings,
            get_chat_model: lambda: FakeChatModel(),
            get_rag_service: lambda: rag,
        }
    )
    with TestClient(app) as c:
        c.headers["X-API-Key"] = "test-key"
        yield c
    app.dependency_overrides.clear()
