"""ownership: adopt legacy tables, add owner columns, backfill to admin

On a fresh database the legacy tables don't exist yet (the app's startup
DDL has never run), so this revision first executes the app's own
CREATE TABLE IF NOT EXISTS blocks — Alembic thereby adopts those tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-19
"""

import os
import uuid

import sqlalchemy as sa
from alembic import op

# The app's startup DDL, imported so the schemas can never drift apart.
from cherryai_api.db import _CREATE_TABLES
from cherryai_api.feedback import CREATE_FEEDBACK_TABLE
from cherryai_api.settings import get_settings
from cherryai_api.wiki import CREATE_WIKI_TABLE

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_BOOTSTRAP_HELP = (
    "Existing data needs an owner but no admin user exists. Either run "
    "`uv run alembic upgrade 0001` then `uv run cherryai users bootstrap` "
    "then `uv run alembic upgrade head`, or set CHERRYAI_ADMIN_EMAIL and "
    "CHERRYAI_ADMIN_PASSWORD and re-run the upgrade."
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


def _ensure_admin(bind) -> uuid.UUID | None:
    """Return an admin id, creating one from env vars if orphan rows exist."""
    orphans = 0
    for table in ("sessions", "wiki_entries"):
        orphans += bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
    admin_id = bind.execute(
        sa.text("SELECT id FROM \"user\" WHERE role = 'admin' ORDER BY created_at LIMIT 1")
    ).scalar()
    if admin_id is not None:
        return admin_id
    if orphans == 0:
        return None
    email = os.environ.get("CHERRYAI_ADMIN_EMAIL", "")
    password = os.environ.get("CHERRYAI_ADMIN_PASSWORD", "")
    if not email or not password:
        raise RuntimeError(_BOOTSTRAP_HELP)
    from fastapi_users.password import PasswordHelper

    admin_id = uuid.uuid4()
    bind.execute(
        sa.text(
            'INSERT INTO "user" (id, email, hashed_password, is_active, '
            "is_superuser, is_verified, role, display_name, memory_dataset) "
            "VALUES (:id, :email, :hp, true, true, true, 'admin', 'Admin', :ds)"
        ),
        {
            "id": admin_id,
            "email": email,
            "hp": PasswordHelper().hash(password),
            "ds": get_settings().cognee_dataset,
        },
    )
    return admin_id


def upgrade() -> None:
    bind = op.get_bind()
    # 1. Adopt legacy tables (no-ops when they already exist).
    _execute_ddl(bind, _CREATE_TABLES)
    _execute_ddl(bind, CREATE_WIKI_TABLE)
    _execute_ddl(bind, CREATE_FEEDBACK_TABLE)

    # 2. Resolve/create the owning admin.
    admin_id = _ensure_admin(bind)

    # 3. Nullable columns first, so backfill can run.
    op.add_column("sessions", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.add_column("wiki_entries", sa.Column("owner_id", sa.Uuid(), nullable=True))
    op.add_column("feedback_entries", sa.Column("user_id", sa.Uuid(), nullable=True))

    # 4. Backfill.
    if admin_id is not None:
        bind.execute(
            sa.text("UPDATE sessions SET user_id = :a WHERE user_id IS NULL"),
            {"a": admin_id},
        )
        bind.execute(
            sa.text("UPDATE wiki_entries SET owner_id = :a WHERE owner_id IS NULL"),
            {"a": admin_id},
        )

    # 5. Tighten and constrain.
    op.alter_column("sessions", "user_id", nullable=False)
    op.alter_column("wiki_entries", "owner_id", nullable=False)
    op.create_foreign_key(
        "sessions_user_id_fkey",
        "sessions",
        "user",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "wiki_entries_owner_id_fkey",
        "wiki_entries",
        "user",
        ["owner_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "feedback_entries_user_id_fkey",
        "feedback_entries",
        "user",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    bind.execute(
        sa.text("ALTER TABLE wiki_entries DROP CONSTRAINT IF EXISTS wiki_entries_slug_key")
    )
    op.create_unique_constraint("wiki_entries_owner_slug_key", "wiki_entries", ["owner_id", "slug"])
    op.create_index("sessions_user_created_idx", "sessions", ["user_id", "created_at"])
    op.create_index("wiki_entries_owner_folder_idx", "wiki_entries", ["owner_id", "folder"])


def downgrade() -> None:
    op.drop_index("wiki_entries_owner_folder_idx", "wiki_entries")
    op.drop_index("sessions_user_created_idx", "sessions")
    op.drop_constraint("wiki_entries_owner_slug_key", "wiki_entries")
    op.create_unique_constraint("wiki_entries_slug_key", "wiki_entries", ["slug"])
    op.drop_column("feedback_entries", "user_id")
    op.drop_column("wiki_entries", "owner_id")
    op.drop_column("sessions", "user_id")
