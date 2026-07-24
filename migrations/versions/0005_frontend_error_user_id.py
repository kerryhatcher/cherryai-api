"""frontend-error-user-id: link frontend error reports to the reporting user

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("frontend_errors", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.create_index(
        "ix_frontend_errors_user_id",
        "frontend_errors",
        ["user_id"],
    )
    op.create_foreign_key(
        "frontend_errors_user_id_fkey",
        "frontend_errors",
        "user",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("frontend_errors_user_id_fkey", "frontend_errors", type_="foreignkey")
    op.drop_index("ix_frontend_errors_user_id", "frontend_errors")
    op.drop_column("frontend_errors", "user_id")
