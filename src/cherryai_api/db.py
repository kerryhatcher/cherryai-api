"""Chat session and message persistence in Postgres via an asyncpg pool."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import asyncpg
from pydantic import BaseModel

from cherryai_api.settings import get_settings

_SESSION_TITLE_MAX = 60

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_created_idx
    ON messages (session_id, created_at);
"""


class Session(BaseModel):
    """A chat session row."""

    id: uuid.UUID
    title: str
    created_at: datetime


class Message(BaseModel):
    """A single chat message row."""

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    created_at: datetime


class Database:
    """Thin asyncpg-backed data access layer for sessions and messages."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.connect() must be called before use")
        return self._pool

    async def connect(self) -> None:
        """Open the connection pool and create tables if needed."""
        # Local imports keep db.py free of the routers' FastAPI deps.
        from cherryai_api.feedback import CREATE_FEEDBACK_TABLE
        from cherryai_api.wiki import CREATE_WIKI_TABLE

        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLES)
            await conn.execute(CREATE_WIKI_TABLE)
            await conn.execute(CREATE_FEEDBACK_TABLE)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> bool:
        """Return True if a trivial query succeeds."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT 1") == 1

    async def create_session(self, title: str) -> Session:
        title = (title or "New chat").strip()[:_SESSION_TITLE_MAX] or "New chat"
        row = await self.pool.fetchrow(
            "INSERT INTO sessions (id, title) VALUES ($1, $2) RETURNING id, title, created_at",
            uuid.uuid4(),
            title,
        )
        return Session(**dict(row))

    async def list_sessions(self) -> list[Session]:
        rows = await self.pool.fetch(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        )
        return [Session(**dict(row)) for row in rows]

    async def get_session(self, session_id: uuid.UUID) -> Session | None:
        row = await self.pool.fetchrow(
            "SELECT id, title, created_at FROM sessions WHERE id = $1", session_id
        )
        return Session(**dict(row)) if row else None

    async def list_messages(self, session_id: uuid.UUID) -> list[Message]:
        rows = await self.pool.fetch(
            "SELECT id, session_id, role, content, created_at FROM messages "
            "WHERE session_id = $1 ORDER BY created_at",
            session_id,
        )
        return [Message(**dict(row)) for row in rows]

    async def add_message(self, session_id: uuid.UUID, role: str, content: str) -> Message:
        row = await self.pool.fetchrow(
            "INSERT INTO messages (id, session_id, role, content) "
            "VALUES ($1, $2, $3, $4) "
            "RETURNING id, session_id, role, content, created_at",
            uuid.uuid4(),
            session_id,
            role,
            content,
        )
        return Message(**dict(row))

    async def is_session_empty(self, session_id: uuid.UUID) -> bool:
        count = await self.pool.fetchval(
            "SELECT count(*) FROM messages WHERE session_id = $1", session_id
        )
        return count == 0

    async def set_title(self, session_id: uuid.UUID, title: str) -> None:
        title = (title or "New chat").strip()[:_SESSION_TITLE_MAX] or "New chat"
        await self.pool.execute("UPDATE sessions SET title = $2 WHERE id = $1", session_id, title)


def make_session_title(first_user_message: str) -> str:
    """Derive a session title from the first user message, truncated."""
    collapsed = " ".join(first_user_message.split())
    return collapsed[:_SESSION_TITLE_MAX] or "New chat"


def build_database() -> Database:
    """Construct the Database using the asyncpg-compatible DSN from settings."""
    return Database(get_settings().asyncpg_dsn)


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:  # pragma: no cover
    return dict(row)
