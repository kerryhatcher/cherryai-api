"""Shared fixtures for tests that need the dev Postgres.

The pool fixture connects with the same settings the app uses (dev Postgres
from docker-compose). Every test row is created with a ``Ztest`` title so its
slug starts with ``ztest-`` (wiki) or its title starts with ``Ztest ```
(feedback); the fixture wipes only those rows, never touching real demo
content.
"""

from __future__ import annotations

import pytest

from cherryai_api.db import build_database

_TEST_SLUG_PREFIX = "ztest-"
_TEST_TITLE_PREFIX = "Ztest "


async def _clean_test_rows(pool) -> None:
    await pool.execute(
        "DELETE FROM wiki_entries WHERE slug LIKE $1", f"{_TEST_SLUG_PREFIX}%"
    )
    await pool.execute(
        "DELETE FROM feedback_entries WHERE title LIKE $1", f"{_TEST_TITLE_PREFIX}%"
    )


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
