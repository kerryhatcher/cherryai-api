"""Tests for database helpers, including live-Postgres session scoping."""

import pytest

from cherryai_api.db import make_session_title


def test_make_session_title_collapses_whitespace() -> None:
    assert make_session_title("  hello   world  ") == "hello world"


def test_make_session_title_truncates_long_input() -> None:
    title = make_session_title("x" * 200)
    assert len(title) == 60


def test_make_session_title_defaults_when_empty() -> None:
    assert make_session_title("   ") == "New chat"


@pytest.mark.asyncio
async def test_sessions_are_scoped_per_user(pool, make_user):
    from cherryai_api.db import build_database

    alice = await make_user("ztest-alice@example.com")
    bob = await make_user("ztest-bob@example.com")
    db = build_database()
    await db.connect()
    try:
        mine = await db.create_session("Ztest alice chat", alice["id"])
        assert await db.get_session(mine.id, alice["id"]) is not None
        assert await db.get_session(mine.id, bob["id"]) is None
        assert mine.id not in [s.id for s in await db.list_sessions(bob["id"])]
    finally:
        await db.close()
