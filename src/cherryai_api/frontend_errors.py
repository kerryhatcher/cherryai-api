"""Frontend error reports: SQLAlchemy model backing ``POST /api/log/error``.

Replaces the old JSONL file sink (lost on every DO App Platform restart,
since the container filesystem is ephemeral) with a Postgres table.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.orm import Base


class FrontendError(Base):
    """One reported browser error or unhandled promise rejection."""

    __tablename__ = "frontend_errors"
    # Newest-first is the only read pattern (see the admin/CLI listing), so
    # the index is created descending rather than relying on a plain
    # ascending b-tree scanned in reverse.
    __table_args__ = (Index("ix_frontend_errors_created_at", text("created_at DESC")),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
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
