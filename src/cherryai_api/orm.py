"""Async SQLAlchemy engine and session factory.

SQLAlchemy is the go-forward data layer: new tables (users, access tokens)
are declarative models here, while legacy tables (sessions, messages,
wiki_entries, feedback_entries) are still accessed through the raw asyncpg
pool in db.py and will be rewritten opportunistically.
"""

from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from cherryai_api.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy-managed tables."""


def sqlalchemy_url() -> str:
    """Return the database URL with the explicit asyncpg driver marker.

    Also rewrites a libpq-style ``sslmode`` query param (as managed Postgres
    providers like DigitalOcean supply) to ``ssl``: SQLAlchemy's asyncpg
    dialect forwards unrecognized query params straight through as kwargs to
    ``asyncpg.connect()``, which has no ``sslmode`` parameter, only ``ssl``.
    """
    url = get_settings().database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    if "sslmode" in query:
        query["ssl"] = query.pop("sslmode")
    return urlunsplit(parts._replace(query=urlencode(query)))


engine = create_async_engine(sqlalchemy_url())
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one AsyncSession per request."""
    async with async_session_maker() as session:
        yield session
