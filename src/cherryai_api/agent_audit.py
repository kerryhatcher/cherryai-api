"""Agent audit log: records every tool call made in a child session.

Stored in Postgres for parent-facing visibility (future UI). Logging is
best-effort: a failed log write must never crash the agent turn.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.orm import Base

CREATE_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id UUID NOT NULL,
    tool_name TEXT NOT NULL,
    args_summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_audit_log_session
    ON agent_audit_log (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_audit_log_user_created
    ON agent_audit_log (user_id, created_at DESC);
"""


async def log_tool_call(
    pool, session_id: uuid.UUID, user_id: uuid.UUID, tool_name: str, args_summary: str = ""
) -> None:
    """Record an agent tool invocation. Best-effort: never raises."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_audit_log (id, session_id, user_id, tool_name, args_summary) "
                "VALUES ($1, $2, $3, $4, $5)",
                uuid.uuid4(),
                session_id,
                user_id,
                tool_name,
                args_summary[:500],
            )
    except Exception:
        pass  # audit log failure must never crash the agent turn


CREATE_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id UUID NOT NULL,
    tool_name TEXT NOT NULL,
    args_summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_audit_log_session
    ON agent_audit_log (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_audit_log_user_created
    ON agent_audit_log (user_id, created_at DESC);
"""


class AgentAuditLog(Base):
    """One agent tool invocation in a child (or otherwise audited) session."""

    __tablename__ = "agent_audit_log"
    __table_args__ = (
        Index("ix_audit_log_session", text("session_id, created_at")),
        Index("ix_audit_log_user_created", text("user_id, created_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID]
    user_id: Mapped[uuid.UUID]
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    args_summary: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
