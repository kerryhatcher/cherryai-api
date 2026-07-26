"""add_model_configs_table

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-26

Creates the model_configs table for per-call-site AI model/provider overrides.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | Sequence[str] | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS model_configs (
            id UUID PRIMARY KEY,
            call_site TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            model_name TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_model_configs_call_site ON model_configs (call_site)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS model_configs")
