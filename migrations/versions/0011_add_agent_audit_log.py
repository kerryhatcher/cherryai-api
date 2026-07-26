"""add_agent_audit_log

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-26

Creates the agent_audit_log table for recording tool calls in child sessions.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_audit_log (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL,
            user_id UUID NOT NULL,
            tool_name TEXT NOT NULL,
            args_summary TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_audit_log_session "
        "ON agent_audit_log (session_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_audit_log_user_created "
        "ON agent_audit_log (user_id, created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_audit_log")
