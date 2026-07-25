"""families and family_memberships

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "families",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "family_memberships",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "family_id",
            sa.Uuid(),
            sa.ForeignKey("families.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("perm_wiki", sa.String(), nullable=False, server_default="none"),
        sa.Column("perm_meals", sa.String(), nullable=False, server_default="none"),
        sa.Column("perm_planner", sa.String(), nullable=False, server_default="none"),
        sa.Column("chat_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("web_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("family_id", "user_id", name="uq_membership_family_user"),
        sa.CheckConstraint(
            "role IN ('organizer','admin','adult','child')",
            name="ck_membership_role",
        ),
        sa.CheckConstraint(
            "perm_wiki IN ('none','view','edit') AND "
            "perm_meals IN ('none','view','edit') AND "
            "perm_planner IN ('none','view','edit')",
            name="ck_membership_perms",
        ),
    )
    op.create_index(
        "uq_family_organizer",
        "family_memberships",
        ["family_id"],
        unique=True,
        postgresql_where=sa.text("role = 'organizer'"),
    )
    op.create_index(
        "ix_family_memberships_user",
        "family_memberships",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("family_memberships")
    op.drop_table("families")
