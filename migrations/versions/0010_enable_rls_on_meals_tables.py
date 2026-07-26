"""enable_rls_on_meals_tables

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-26

Enables row-level security on meals tables: meal_plans, recipes,
shopping_lists, pantry_items, stores.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_POLICY = """
((family_id IS NULL
  AND owner_id::text = current_setting('app.user_id', true))
 OR (family_id IS NOT NULL
     AND family_id::text = current_setting('app.family_id', true)))
"""

_TABLES = ("meal_plans", "recipes", "shopping_lists", "pantry_items", "stores")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_family_isolation ON {table} "
            f"USING ({_POLICY}) "
            f"WITH CHECK ({_POLICY})"
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_family_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
