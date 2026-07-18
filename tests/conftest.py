"""Shared fixtures for tests that need the dev Postgres.

The pool fixture connects with the same settings the app uses (dev Postgres
from docker-compose). Every test row is created with a ``Ztest`` title so its
slug starts with ``ztest-``; the fixture wipes only those rows, never touching
real demo pages.
"""

from __future__ import annotations

import pytest

from cherryai_api.db import build_database

_TEST_SLUG_PREFIX = "ztest-"


@pytest.fixture
async def pool():
    """Yield a connected asyncpg pool, cleaning up test rows around each test."""
    db = build_database()
    await db.connect()
    try:
        await db.pool.execute(
            "DELETE FROM wiki_entries WHERE slug LIKE $1", f"{_TEST_SLUG_PREFIX}%"
        )
        yield db.pool
        await db.pool.execute(
            "DELETE FROM wiki_entries WHERE slug LIKE $1", f"{_TEST_SLUG_PREFIX}%"
        )
    finally:
        await db.close()
