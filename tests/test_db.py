"""Tests for pure database helpers that need no live Postgres."""

from cherryai_api.db import make_session_title


def test_make_session_title_collapses_whitespace() -> None:
    assert make_session_title("  hello   world  ") == "hello world"


def test_make_session_title_truncates_long_input() -> None:
    title = make_session_title("x" * 200)
    assert len(title) == 60


def test_make_session_title_defaults_when_empty() -> None:
    assert make_session_title("   ") == "New chat"
