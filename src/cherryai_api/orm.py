"""Async SQLAlchemy engine and session factory.

SQLAlchemy is the go-forward data layer: new tables (users, access tokens)
are declarative models here, while legacy tables (sessions, messages,
wiki_entries, feedback_entries) are still accessed through the raw asyncpg
pool in db.py and will be rewritten opportunistically.
"""

from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Session as _SyncSession

from cherryai_api.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy-managed tables."""


def sqlalchemy_url() -> str:
    """Return the database URL with the explicit asyncpg driver marker.

    Also rewrites a libpq-style ``sslmode`` query param (as managed Postgres
    providers like DigitalOcean supply) to ``ssl``: SQLAlchemy's asyncpg
    dialect forwards unrecognized query params straight through as kwargs to
    ``asyncpg.connect()``, which has no ``sslmode`` parameter, only ``ssl``.

    Uses the superuser ``database_url`` — this is the migration connection.
    Runtime queries use ``app_sqlalchemy_url()`` instead.
    """
    url = get_settings().database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    if "sslmode" in query:
        query["ssl"] = query.pop("sslmode")
    return urlunsplit(parts._replace(query=urlencode(query)))


def app_sqlalchemy_url() -> str:
    """Return the runtime (non-superuser) SQLAlchemy URL.

    Falls back to ``sqlalchemy_url()`` when ``app_database_url`` is unset,
    so existing deployments without the app role continue to work.
    """
    settings = get_settings()
    url = settings.app_database_url or settings.database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    if "sslmode" in query:
        query["ssl"] = query.pop("sslmode")
    return urlunsplit(parts._replace(query=urlencode(query)))


engine = create_async_engine(app_sqlalchemy_url())
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one AsyncSession per request."""
    async with async_session_maker() as session:
        yield session


@event.listens_for(_SyncSession, "after_begin")
def _set_rls_gucs(session, transaction, connection) -> None:
    """Stamp RLS GUCs on every ORM transaction from the request ContextVars.

    Reads ``validated_family_var``, NOT ``active_family_var`` — the latter is
    the raw, caller-supplied header/cookie value, and ``app.family_id`` must
    never carry a family id that ``authz.get_capability`` hasn't checked
    against the caller's memberships. A request whose ORM queries never go
    through ``get_capability`` gets an empty ``app.family_id`` here, which
    RLS treats as "no family" rather than trusting unvalidated input.

    Import inside the handler to avoid an orm↔family_context import cycle at
    module load.
    """
    from cherryai_api.family_context import current_user_var, validated_family_var

    user_id = current_user_var.get()
    family_id = validated_family_var.get()
    connection.execute(
        text("SELECT set_config('app.user_id', :u, true), set_config('app.family_id', :f, true)"),
        {"u": str(user_id) if user_id else "", "f": str(family_id) if family_id else ""},
    )
