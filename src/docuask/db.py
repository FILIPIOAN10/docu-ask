"""Engine and session factory. Sessions are short: open, query, close - never held across a model call."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from docuask.config import get_settings


def make_engine(url: str) -> Engine:
    # pool_pre_ping: a connection that the server closed (restart, idle timeout) is replaced instead of failing a request.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed (connection returned to the pool) when it ends."""
    with get_session_factory()() as session:
        yield session
