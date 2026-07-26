"""enable_rls_on_wiki_entries

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-25

Enables row-level security on wiki_entries — the first shared table to
adopt the RLS machinery built in Phase 1. The policy uses the same
app.user_id / app.family_id GUCs as the ORM listener and scoped_connection
contextmanager, so both ORM and raw-asyncpg queries are covered.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WIKI_POLICY = """
((family_id IS NULL
  AND owner_id::text = current_setting('app.user_id', true))
 OR (family_id IS NOT NULL
     AND family_id::text = current_setting('app.family_id', true)))
"""


def upgrade() -> None:
    op.execute("ALTER TABLE wiki_entries ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_entries FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY wiki_entries_family_isolation ON wiki_entries "
        f"USING ({_WIKI_POLICY}) "
        f"WITH CHECK ({_WIKI_POLICY})"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS wiki_entries_family_isolation ON wiki_entries")
    op.execute("ALTER TABLE wiki_entries NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_entries DISABLE ROW LEVEL SECURITY")
