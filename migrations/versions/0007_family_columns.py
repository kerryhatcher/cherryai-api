"""family_id + audience columns on shared-module tables

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

# Import app's startup DDL for meal/pantry/planner tables.
from cherryai_api.meals import CREATE_MEALS_TABLES
from cherryai_api.planner import CREATE_PLANNER_TABLES

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

SHARED_TABLES = (
    "wiki_entries",
    "meal_plans",
    "recipes",
    "shopping_lists",
    "pantry_items",
    "stores",
    "planner_projects",
)


def _execute_ddl(bind, ddl: str) -> None:
    """Run a multi-statement DDL block one statement at a time.

    The asyncpg dialect prepares single statements, so the app's
    semicolon-joined CREATE TABLE blocks must be split. None of the app's
    DDL contains semicolons inside string literals.
    """
    for statement in ddl.split(";"):
        if statement.strip():
            bind.execute(sa.text(statement))


def upgrade() -> None:
    bind = op.get_bind()
    # Adopt shared module tables if they don't already exist (fresh migrations).
    # These are "CREATE TABLE IF NOT EXISTS", so they're safe to run always.
    _execute_ddl(bind, CREATE_MEALS_TABLES)
    _execute_ddl(bind, CREATE_PLANNER_TABLES)

    for table in SHARED_TABLES:
        op.add_column(table, sa.Column("family_id", sa.Uuid(), nullable=True))
        op.create_foreign_key(
            f"fk_{table}_family",
            table,
            "families",
            ["family_id"],
            ["id"],
            ondelete="CASCADE",
        )
        # family_id leads for RLS/index discipline (research: multitenancy doc)
        # pantry_items has no created_at, so index it with just family_id
        if table == "pantry_items":
            op.create_index(f"ix_{table}_family", table, ["family_id"])
        else:
            op.create_index(f"ix_{table}_family", table, ["family_id", "created_at"])

    op.add_column(
        "wiki_entries",
        sa.Column("audience", sa.String(), nullable=False, server_default="adults"),
    )
    # Replace the global (owner_id, slug) uniqueness with per-scope partial
    # unique indexes. NOTE: constraint name confirmed in 0002_ownership.py
    op.drop_constraint("wiki_entries_owner_slug_key", "wiki_entries", type_="unique")
    op.create_index(
        "uq_wiki_slug_personal",
        "wiki_entries",
        ["owner_id", "slug"],
        unique=True,
        postgresql_where=sa.text("family_id IS NULL"),
    )
    op.create_index(
        "uq_wiki_slug_family",
        "wiki_entries",
        ["family_id", "slug"],
        unique=True,
        postgresql_where=sa.text("family_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_wiki_slug_family", table_name="wiki_entries")
    op.drop_index("uq_wiki_slug_personal", table_name="wiki_entries")
    op.create_unique_constraint("wiki_entries_owner_slug_key", "wiki_entries", ["owner_id", "slug"])
    op.drop_column("wiki_entries", "audience")
    for table in SHARED_TABLES:
        op.drop_index(f"ix_{table}_family", table_name=table)
        op.drop_constraint(f"fk_{table}_family", table, type_="foreignkey")
        op.drop_column(table, "family_id")
