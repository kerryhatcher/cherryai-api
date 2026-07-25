"""CLI `errors list` / `errors show` commands.

Seed rows are inserted directly through the ORM (`async_session_maker`),
each call wrapped in its own `asyncio.run()` and followed by
`engine.dispose()` — the same mechanics `cli.py`'s `_run_with_session`
uses, and required so each `asyncio.run()` (seed insert, then the CLI
invocation itself) gets a fresh, undisposed-of set of connections rather
than one bound to a different, already-finished event loop. Tests stay
synchronous (not `async def`) so `CliRunner().invoke()` can call
`asyncio.run()` internally without hitting "asyncio.run() cannot be
called from a running event loop" — mirrors `test_users_list_runs` in
test_cli_users.py, not the subprocess pattern used by the one async CLI
test (which exists only to test cross-process atomicity).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from cherryai_api.cli import app
from cherryai_api.frontend_errors import FrontendError
from cherryai_api.orm import async_session_maker, engine


def _insert_error(**overrides) -> uuid.UUID:
    """Insert one FrontendError row via the ORM and return its id."""
    fields = {
        "id": uuid.uuid4(),
        "user_id": None,
        "message": "Ztest boom",
        "source": "app.js",
        "lineno": 42,
        "colno": 7,
        "stack": None,
        "url": "https://example.test/page",
        "user_agent": "pytest-agent",
        "client_ip": "127.0.0.1",
        "client_timestamp": "2026-07-24T00:00:00Z",
        "context": "unhandledrejection",
    }
    fields.update(overrides)

    async def _do():
        async with async_session_maker() as session:
            session.add(FrontendError(**fields))
            await session.commit()
        await engine.dispose()

    asyncio.run(_do())
    return fields["id"]


def _insert_error_with_user(email: str, **overrides) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert a reporting User plus a FrontendError referencing it."""
    from fastapi_users.password import PasswordHelper

    from cherryai_api.users import User

    user_id = uuid.uuid4()
    fields = {
        "id": uuid.uuid4(),
        "message": "Ztest with a reporter",
        "context": "window.onerror",
    }
    fields.update(overrides)

    async def _do():
        async with async_session_maker() as session:
            # Two commits, not one: the FK to `user` (added by migration 0005)
            # isn't declared on the mapped column (see FrontendError's
            # docstring), so SQLAlchemy has no dependency info to order a
            # single flush correctly — the user row must exist first.
            session.add(
                User(
                    id=user_id,
                    email=email,
                    hashed_password=PasswordHelper().hash("pw-ztest-123"),
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    role="chat",
                    display_name="",
                    memory_dataset=f"user-{user_id}",
                )
            )
            await session.commit()
            session.add(FrontendError(user_id=user_id, **fields))
            await session.commit()
        await engine.dispose()

    asyncio.run(_do())
    return fields["id"], user_id


def test_errors_list_runs(pool):
    """Bare invocation succeeds even with no matching rows (or a populated table)."""
    result = CliRunner().invoke(app, ["errors", "list"])
    assert result.exit_code == 0


def test_errors_list_shows_seeded_row(pool):
    error_id = _insert_error(message="Ztest boom goes the dynamite", context="window.onerror")

    result = CliRunner().invoke(app, ["errors", "list"])

    assert result.exit_code == 0
    # Rows are keyed by a short id prefix, not the full uuid (see `_fit`).
    assert str(error_id)[:8] in result.stdout
    assert "(anonymous)" in result.stdout
    assert "window.onerror" in result.stdout
    assert "Ztest boom goes the dynamite" in result.stdout


def test_errors_list_truncates_long_message(pool):
    long_message = "Ztest " + ("x" * 200)
    _insert_error(message=long_message)

    result = CliRunner().invoke(app, ["errors", "list"])

    assert result.exit_code == 0
    assert long_message not in result.stdout
    # Exactly the preview budget, ellipsis included — asserting on a bare
    # "…" would pass on any other row's truncation.
    assert "Ztest " + ("x" * 33) + "…" in result.stdout


def test_errors_list_rows_fit_in_120_columns(pool):
    _insert_error(message="Ztest " + ("x" * 200), context="Z" * 80)
    _insert_error_with_user("ztest-cli-errors-wide@example.com", message="Ztest wide row")

    result = CliRunner().invoke(app, ["errors", "list"])

    assert result.exit_code == 0
    assert result.stdout.strip()
    for line in result.stdout.splitlines():
        assert len(line) <= 120, line


def test_errors_list_respects_limit(pool):
    _insert_error(message="Ztest one")
    _insert_error(message="Ztest two")
    _insert_error(message="Ztest three")

    result = CliRunner().invoke(app, ["errors", "list", "--limit", "2"])

    assert result.exit_code == 0
    assert len(result.stdout.strip().splitlines()) == 2


def test_errors_list_shows_reporting_user_email(pool):
    _insert_error_with_user("ztest-cli-errors@example.com")

    result = CliRunner().invoke(app, ["errors", "list"])

    assert result.exit_code == 0
    assert "ztest-cli-errors@example.com" in result.stdout


def test_errors_show_full_detail(pool):
    error_id = _insert_error(
        message="Ztest full detail",
        stack="Error: Ztest full detail\n    at foo (app.js:1:1)",
    )

    result = CliRunner().invoke(app, ["errors", "show", str(error_id)])

    assert result.exit_code == 0
    assert "Ztest full detail" in result.stdout
    assert "at foo (app.js:1:1)" in result.stdout
    # Every stored column is reachable from `show` — client_ts is the one
    # the frontend sends and the server does not re-derive.
    assert "2026-07-24T00:00:00Z" in result.stdout
    assert "app.js:42:7" in result.stdout
    assert "pytest-agent" in result.stdout


def test_errors_show_accepts_id_prefix(pool):
    error_id = _insert_error(message="Ztest by prefix")

    result = CliRunner().invoke(app, ["errors", "show", str(error_id)[:8]])

    assert result.exit_code == 0
    assert str(error_id) in result.stdout


def test_errors_show_rejects_ambiguous_prefix(pool):
    shared = "aaaaaaaa-0000-0000-0000-00000000000"
    _insert_error(id=uuid.UUID(f"{shared}1"), message="Ztest ambiguous one")
    _insert_error(id=uuid.UUID(f"{shared}2"), message="Ztest ambiguous two")

    result = CliRunner().invoke(app, ["errors", "show", "aaaaaaaa"])

    assert result.exit_code == 1
    assert "ambiguous" in result.stderr


def test_errors_show_indents_multiline_message(pool):
    error_id = _insert_error(message="Ztest first line\nsecond line")

    result = CliRunner().invoke(app, ["errors", "show", str(error_id)])

    assert result.exit_code == 0
    # Continuation lines line up under the value column, not at column 0.
    assert "message:    Ztest first line" in result.stdout
    assert "\n            second line" in result.stdout


def test_errors_show_renders_null_user_id(pool):
    error_id = _insert_error(message="Ztest orphaned row")

    result = CliRunner().invoke(app, ["errors", "show", str(error_id)])

    assert result.exit_code == 0
    assert "user:       (anonymous) (-)" in result.stdout


def test_errors_show_unknown_id(pool):
    result = CliRunner().invoke(app, ["errors", "show", str(uuid.uuid4())])

    assert result.exit_code == 1


def test_errors_show_invalid_id(pool):
    result = CliRunner().invoke(app, ["errors", "show", "not-a-uuid"])

    assert result.exit_code == 1


def _count_older_than(days: int) -> int:
    """Rows the prune is about to delete, including any not inserted by this test.

    `prune_frontend_errors` deletes by age alone — it has no `Ztest` message
    filter — so a database holding real reports older than the window makes
    any hardcoded expected count wrong. Row-count assertions are stated
    relative to this baseline instead.
    """
    from sqlalchemy import func, select

    async def _do():
        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            async with async_session_maker() as session:
                return await session.scalar(
                    select(func.count())
                    .select_from(FrontendError)
                    .where(FrontendError.created_at < cutoff)
                )
        finally:
            await engine.dispose()

    return asyncio.run(_do())


def test_errors_prune_deletes_old_rows_and_reports_count(pool):
    baseline = _count_older_than(14)
    old_id = _insert_error(
        message="Ztest prune old", created_at=datetime.now(UTC) - timedelta(days=30)
    )
    kept_id = _insert_error(
        message="Ztest prune keep", created_at=datetime.now(UTC) - timedelta(days=1)
    )

    result = CliRunner().invoke(app, ["errors", "prune", "--days", "14"])

    assert result.exit_code == 0
    assert f"Deleted {baseline + 1} frontend error report(s) older than 14 day(s)." in result.stdout

    show_old = CliRunner().invoke(app, ["errors", "show", str(old_id)])
    assert show_old.exit_code == 1
    show_kept = CliRunner().invoke(app, ["errors", "show", str(kept_id)])
    assert show_kept.exit_code == 0


def test_errors_prune_defaults_to_14_days(pool):
    """Bare invocation (no `--days`) uses the shared retention constant."""
    old_id = _insert_error(
        message="Ztest prune default", created_at=datetime.now(UTC) - timedelta(days=15)
    )

    result = CliRunner().invoke(app, ["errors", "prune"])

    assert result.exit_code == 0
    assert "older than 14 day(s)" in result.stdout
    show_old = CliRunner().invoke(app, ["errors", "show", str(old_id)])
    assert show_old.exit_code == 1


def test_errors_prune_with_no_matching_rows(pool):
    """A second prune with the same window is a no-op — nothing is left to delete."""
    CliRunner().invoke(app, ["errors", "prune", "--days", "14"])

    result = CliRunner().invoke(app, ["errors", "prune", "--days", "14"])

    assert result.exit_code == 0
    assert "Deleted 0 frontend error report(s)" in result.stdout
