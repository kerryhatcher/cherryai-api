"""identity: user and accesstoken tables

Revision ID: 0001
Revises:
Create Date: 2026-07-19

"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(length=1024), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="chat"),
        sa.Column("display_name", sa.String(), nullable=False, server_default=""),
        sa.Column("memory_dataset", sa.String(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_user_email", "user", ["email"], unique=True)
    op.create_table(
        "accesstoken",
        sa.Column("token", sa.String(length=43), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("user.id", ondelete="cascade"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_accesstoken_created_at", "accesstoken", ["created_at"])


def downgrade() -> None:
    op.drop_table("accesstoken")
    op.drop_table("user")
