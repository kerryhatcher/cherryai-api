"""Frontend error reports: SQLAlchemy model backing ``POST /api/log/error``.

Replaces the old JSONL file sink (lost on every DO App Platform restart,
since the container filesystem is ephemeral) with a Postgres table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, Index, Integer, String, Text, delete, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.orm import Base

# How long a frontend error report is kept before `prune_frontend_errors`
# deletes it. One constant shared by the periodic in-process prune (see
# `api.py`'s lifespan) and the `cherryai errors prune` CLI command, so the
# retention window can never drift between the two call sites.
FRONTEND_ERROR_RETENTION_DAYS = 14


class FrontendError(Base):
    """One reported browser error or unhandled promise rejection."""

    __tablename__ = "frontend_errors"
    # Newest-first is the only read pattern (see the admin/CLI listing), so
    # the index is created descending rather than relying on a plain
    # ascending b-tree scanned in reverse.
    __table_args__ = (Index("ix_frontend_errors_created_at", text("created_at DESC")),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Nullable even though the endpoint now always has an authenticated caller:
    # `ondelete="SET NULL"` (see migration 0005) requires it, and it keeps the
    # historical row around after the reporting account is deleted. The FK
    # constraint itself is created in the migration, not declared here — this
    # codebase's other user-owned tables (e.g. `UserFastmailCredential` in
    # integrations.py) follow the same split.
    user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    lineno: Mapped[int | None] = mapped_column(Integer, nullable=True)
    colno: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stack: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String, nullable=True)
    # Raw client-supplied string (see ``LogErrorRequest.timestamp`` in
    # api.py) — not parsed/validated, so it's text rather than a timestamp
    # column.
    client_timestamp: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


async def prune_frontend_errors(
    session: AsyncSession, retention_days: int = FRONTEND_ERROR_RETENTION_DAYS
) -> int:
    """Delete ``frontend_errors`` rows older than ``retention_days`` and return the count.

    Shared by the periodic in-process prune in `api.py`'s lifespan and the
    `cherryai errors prune` CLI command — both pass through here rather than
    each building their own DELETE, so the two call sites cannot drift.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    result = await session.execute(delete(FrontendError).where(FrontendError.created_at < cutoff))
    await session.commit()
    return result.rowcount
