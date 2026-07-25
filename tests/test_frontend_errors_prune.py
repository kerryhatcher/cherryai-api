"""prune_frontend_errors: deletes `frontend_errors` rows older than the retention window.

Direct ORM tests (no HTTP layer, no lifespan) — the retention window is
exercised via explicit `created_at` timestamps rather than by waiting or
manipulating the clock. Rows are given a `Ztest` message prefix so the
`pool` fixture's cleanup catches them regardless of how old their
`created_at` is set to.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from cherryai_api.frontend_errors import (
    FRONTEND_ERROR_RETENTION_DAYS,
    FrontendError,
    prune_frontend_errors,
)
from cherryai_api.orm import async_session_maker


async def _insert(created_at: datetime, message: str) -> uuid.UUID:
    error_id = uuid.uuid4()
    async with async_session_maker() as session:
        session.add(FrontendError(id=error_id, message=message, created_at=created_at))
        await session.commit()
    return error_id


async def _still_exists(error_id: uuid.UUID) -> bool:
    async with async_session_maker() as session:
        return await session.get(FrontendError, error_id) is not None


async def _count_older_than(days: int) -> int:
    """Rows the prune is about to delete, including any not inserted by this test.

    `prune_frontend_errors` deletes by age alone — it has no `Ztest` message
    filter — so a database holding real reports older than the window makes
    any hardcoded expected count wrong. Row-count assertions are stated
    relative to this baseline instead.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with async_session_maker() as session:
        return await session.scalar(
            select(func.count()).select_from(FrontendError).where(FrontendError.created_at < cutoff)
        )


@pytest.mark.asyncio
async def test_prune_deletes_rows_older_than_window(pool):
    old_id = await _insert(datetime.now(UTC) - timedelta(days=15), "Ztest ancient error")

    async with async_session_maker() as session:
        deleted = await prune_frontend_errors(session, retention_days=14)

    assert deleted >= 1
    assert not await _still_exists(old_id)


@pytest.mark.asyncio
async def test_prune_keeps_rows_inside_window(pool):
    recent_id = await _insert(datetime.now(UTC) - timedelta(days=1), "Ztest recent error")

    async with async_session_maker() as session:
        await prune_frontend_errors(session, retention_days=14)

    assert await _still_exists(recent_id)


@pytest.mark.asyncio
async def test_prune_boundary_behaves_sanely(pool):
    # Just inside the 14-day window: must survive.
    inside_id = await _insert(
        datetime.now(UTC) - timedelta(days=13, hours=23), "Ztest just inside window"
    )
    # Just outside the 14-day window: must be deleted.
    outside_id = await _insert(
        datetime.now(UTC) - timedelta(days=14, hours=1), "Ztest just outside window"
    )

    async with async_session_maker() as session:
        await prune_frontend_errors(session, retention_days=14)

    assert await _still_exists(inside_id)
    assert not await _still_exists(outside_id)


@pytest.mark.asyncio
async def test_prune_returns_deleted_row_count(pool):
    baseline = await _count_older_than(14)
    await _insert(datetime.now(UTC) - timedelta(days=20), "Ztest count one")
    await _insert(datetime.now(UTC) - timedelta(days=21), "Ztest count two")
    kept_id = await _insert(datetime.now(UTC) - timedelta(days=1), "Ztest count keep")

    async with async_session_maker() as session:
        deleted = await prune_frontend_errors(session, retention_days=14)

    assert deleted == baseline + 2
    assert await _still_exists(kept_id)


@pytest.mark.asyncio
async def test_prune_defaults_to_the_shared_retention_constant(pool):
    # No explicit `retention_days` — must fall back to
    # `FRONTEND_ERROR_RETENTION_DAYS`, the same constant the CLI command
    # defaults to (see test_cli_errors.py's prune tests).
    old_id = await _insert(
        datetime.now(UTC) - timedelta(days=FRONTEND_ERROR_RETENTION_DAYS + 1),
        "Ztest default retention",
    )

    async with async_session_maker() as session:
        await prune_frontend_errors(session)

    assert not await _still_exists(old_id)


@pytest.mark.asyncio
async def test_periodic_prune_task_stops_when_cancelled(pool):
    """The lifespan's background task must end on cancel, not hang or loop on.

    Cancelling right after the task starts lands the cancellation *inside*
    the first database pass rather than the 24-hour sleep — the case where
    the pass's broad `except Exception` could wrongly swallow it (it can't:
    `CancelledError` is a `BaseException`). `wait_for` bounds the await so a
    task that ignored cancellation fails the test instead of hanging the
    suite, exactly as it would hang a redeploy.
    """
    from cherryai_api.api import _prune_frontend_errors_periodically

    task = asyncio.create_task(_prune_frontend_errors_periodically())
    await asyncio.sleep(0)  # let it start and reach its first await
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)

    assert task.done()
    assert task.cancelled()
