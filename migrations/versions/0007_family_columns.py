"""family_id + audience columns on shared-module tables

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

# Frozen snapshot of cherryai_api.meals.CREATE_MEALS_TABLES as of revision
# 0007. Migrations must replay identically on a fresh database no matter how
# the app-side DDL constants change later, so this is a verbatim copy rather
# than an import -- app-side DDL blocks are conventionally frozen too, but
# this makes the migration self-contained regardless.
CREATE_MEALS_TABLES = """
CREATE TABLE IF NOT EXISTS meal_plans (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    owner_id UUID NOT NULL,
    week_start DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meal_plan_days (
    id UUID PRIMARY KEY,
    plan_id UUID NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE,
    day_date DATE NOT NULL,
    meal_type TEXT NOT NULL DEFAULT 'dinner',
    notes TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS meal_plan_days_plan_idx
    ON meal_plan_days (plan_id, day_date, meal_type);

CREATE TABLE IF NOT EXISTS recipes (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT NOT NULL DEFAULT '',
    prep_minutes INTEGER,
    cook_minutes INTEGER,
    servings INTEGER NOT NULL DEFAULT 4,
    source_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meal_plan_day_recipes (
    id UUID PRIMARY KEY,
    day_id UUID NOT NULL REFERENCES meal_plan_days(id) ON DELETE CASCADE,
    recipe_id UUID NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS meal_plan_day_recipes_day_idx
    ON meal_plan_day_recipes (day_id, sort_order);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id UUID PRIMARY KEY,
    recipe_id UUID NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    quantity REAL,
    unit TEXT,
    notes TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS recipe_ingredients_recipe_idx
    ON recipe_ingredients (recipe_id, sort_order);

CREATE TABLE IF NOT EXISTS shopping_lists (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    name TEXT NOT NULL,
    plan_id UUID REFERENCES meal_plans(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shopping_list_items (
    id UUID PRIMARY KEY,
    list_id UUID NOT NULL REFERENCES shopping_lists(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    quantity REAL,
    unit TEXT,
    category TEXT NOT NULL DEFAULT '',
    purchased BOOLEAN NOT NULL DEFAULT false,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS shopping_list_items_list_idx
    ON shopping_list_items (list_id, sort_order);

CREATE TABLE IF NOT EXISTS pantry_items (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    name TEXT NOT NULL,
    quantity REAL,
    unit TEXT,
    category TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pantry_items_owner_idx ON pantry_items (owner_id, name);

CREATE TABLE IF NOT EXISTS stores (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    name TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS stores_owner_idx ON stores (owner_id);

CREATE TABLE IF NOT EXISTS store_products (
    id UUID PRIMARY KEY,
    store_id UUID NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
    ingredient_name TEXT NOT NULL,
    product_name TEXT NOT NULL,
    package_quantity REAL NOT NULL,
    package_unit TEXT NOT NULL,
    price_cents INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS store_products_store_idx ON store_products (store_id, ingredient_name);
"""

# Frozen snapshot of cherryai_api.planner.CREATE_PLANNER_TABLES as of
# revision 0007 -- see the comment on CREATE_MEALS_TABLES above.
CREATE_PLANNER_TABLES = """
CREATE TABLE IF NOT EXISTS planner_projects (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    color TEXT,
    owner_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS planner_tasks (
    id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES planner_projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    assigned_to UUID,
    due_date DATE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS planner_tasks_project_idx
    ON planner_tasks (project_id, sort_order);

CREATE TABLE IF NOT EXISTS planner_subtasks (
    id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES planner_tasks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT false,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS planner_subtasks_task_idx
    ON planner_subtasks (task_id, sort_order);
"""

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
