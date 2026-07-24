"""frontend-errors: Postgres-backed storage for POST /api/log/error

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "frontend_errors",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("lineno", sa.Integer(), nullable=True),
        sa.Column("colno", sa.Integer(), nullable=True),
        sa.Column("stack", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("client_ip", sa.String(), nullable=True),
        sa.Column("client_timestamp", sa.Text(), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Newest-first is the only read pattern, hence a descending index rather
    # than a plain ascending one.
    op.create_index(
        "ix_frontend_errors_created_at",
        "frontend_errors",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_frontend_errors_created_at", "frontend_errors")
    op.drop_table("frontend_errors")
