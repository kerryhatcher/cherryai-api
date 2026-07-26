"""enable_rls_on_planner_projects

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-26

Enables row-level security on planner_projects — the second shared table
to adopt RLS after wiki_entries (Phase 2).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_POLICY = """
((family_id IS NULL
  AND owner_id::text = current_setting('app.user_id', true))
 OR (family_id IS NOT NULL
     AND family_id::text = current_setting('app.family_id', true)))
"""


def upgrade() -> None:
    op.execute("ALTER TABLE planner_projects ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE planner_projects FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY planner_projects_family_isolation ON planner_projects "
        f"USING ({_POLICY}) "
        f"WITH CHECK ({_POLICY})"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS planner_projects_family_isolation ON planner_projects")
    op.execute("ALTER TABLE planner_projects NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE planner_projects DISABLE ROW LEVEL SECURITY")
