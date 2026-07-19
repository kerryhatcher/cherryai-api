"""Shared fixtures for tests that need the dev Postgres.

The pool fixture connects with the same settings the app uses (dev Postgres
from docker-compose). Every test row is created with a ``Ztest`` title so its
slug starts with ``ztest-`` (wiki) or its title starts with ``Ztest ```
(feedback); the fixture wipes only those rows, never touching real demo
content.
"""

from __future__ import annotations

import os

import asyncpg
import pytest

# Tests must never export telemetry, even with local Logfire credentials
# present. Set before any cherryai_api import can trigger logfire.configure().
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")

from cherryai_api.db import build_database  # noqa: E402

_TEST_SLUG_PREFIX = "ztest-"
_TEST_TITLE_PREFIX = "Ztest "


async def _clean_test_rows(pool) -> None:
    await pool.execute("DELETE FROM wiki_entries WHERE slug LIKE $1", f"{_TEST_SLUG_PREFIX}%")
    await pool.execute("DELETE FROM feedback_entries WHERE title LIKE $1", f"{_TEST_TITLE_PREFIX}%")
    await pool.execute("DELETE FROM sessions WHERE title LIKE 'Ztest%'")
    try:
        await pool.execute("DELETE FROM \"user\" WHERE email LIKE 'ztest-%'")
    except asyncpg.UndefinedTableError:
        pass


@pytest.fixture(autouse=True)
async def _dispose_sqlalchemy_engine():
    """Drop pooled asyncpg connections after each test.

    pytest-asyncio gives every test function its own event loop, but
    ``cherryai_api.orm.engine`` is a process-wide singleton created at import
    time. asyncpg connections are bound to the loop that created them, so a
    connection pooled during one test raises "another operation is in
    progress" (or a cross-loop error) when reused from the next test's loop.
    Disposing after each test forces the pool to open fresh connections on
    whichever loop asks next.
    """
    yield
    from cherryai_api.orm import engine

    await engine.dispose()


@pytest.fixture
async def pool():
    """Yield a connected asyncpg pool, cleaning up test rows around each test."""
    db = build_database()
    await db.connect()
    try:
        await _clean_test_rows(db.pool)
        yield db.pool
        await _clean_test_rows(db.pool)
    finally:
        await db.close()


@pytest.fixture
async def auth_app(pool):
    """A minimal FastAPI app with only auth/users routers — no lifespan.

    Auth routes use SQLAlchemy DI exclusively, so the heavy chat lifespan
    (Cognee import, agent build) is unnecessary for these tests. `pool` is
    depended on for its cleanup side effect.
    """
    from fastapi import FastAPI

    from cherryai_api.admin import router as admin_router
    from cherryai_api.auth import auth_backend, fastapi_users_app
    from cherryai_api.users import UserCreate, UserRead, UserUpdate

    app = FastAPI()
    app.include_router(fastapi_users_app.get_auth_router(auth_backend), prefix="/auth")
    app.include_router(fastapi_users_app.get_register_router(UserRead, UserCreate), prefix="/auth")
    app.include_router(fastapi_users_app.get_users_router(UserRead, UserUpdate), prefix="/users")
    app.include_router(admin_router)
    return app


@pytest.fixture
async def client(auth_app):
    import httpx

    transport = httpx.ASGITransport(app=auth_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def make_user(pool):
    """Insert a user directly (bypassing HTTP) and return its row dict."""
    import uuid as _uuid

    from fastapi_users.password import PasswordHelper

    async def _make(
        email: str,
        password: str = "pw-ztest-123",
        *,
        role: str = "chat",
        is_verified: bool = True,
        is_active: bool = True,
    ) -> dict:
        uid = _uuid.uuid4()
        hashed = PasswordHelper().hash(password)
        await pool.execute(
            'INSERT INTO "user" (id, email, hashed_password, is_active, '
            "is_superuser, is_verified, role, display_name, memory_dataset) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, '', $8)",
            uid,
            email,
            hashed,
            is_active,
            role == "admin",
            is_verified,
            role,
            f"user-{uid}",
        )
        return {"id": uid, "email": email, "password": password}

    return _make
