"""fastmail-credentials: per-user Fastmail credential storage

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-20
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_fastmail_credentials",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(), nullable=False, server_default="Fastmail"),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("app_password_encrypted", sa.String(), nullable=False),
        sa.Column("api_token_encrypted", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_user_fastmail_credentials_user_id",
        "user_fastmail_credentials",
        ["user_id"],
    )
    op.create_foreign_key(
        "user_fastmail_credentials_user_id_fkey",
        "user_fastmail_credentials",
        "user",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_table("user_fastmail_credentials")
