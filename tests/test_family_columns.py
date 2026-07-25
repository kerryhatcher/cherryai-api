"""Schema-level assertions for the 0007 family columns."""

import pytest

SHARED_TABLES = (
    "wiki_entries",
    "meal_plans",
    "recipes",
    "shopping_lists",
    "pantry_items",
    "stores",
    "planner_projects",
)


@pytest.mark.asyncio
async def test_family_id_columns_exist(pool):
    for table in SHARED_TABLES:
        row = await pool.fetchrow(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = $1 AND column_name = 'family_id'",
            table,
        )
        assert row is not None, f"{table}.family_id missing"
        assert row["is_nullable"] == "YES"


@pytest.mark.asyncio
async def test_wiki_audience_column(pool):
    row = await pool.fetchrow(
        "SELECT column_default, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'wiki_entries' AND column_name = 'audience'"
    )
    assert row is not None and row["is_nullable"] == "NO"
    assert "adults" in row["column_default"]


@pytest.mark.asyncio
async def test_partial_unique_slug_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'wiki_entries'"
    )
    defs = {r["indexname"]: r["indexdef"] for r in rows}
    assert "uq_wiki_slug_personal" in defs and "family_id IS NULL" in defs["uq_wiki_slug_personal"]
    assert "uq_wiki_slug_family" in defs and "family_id IS NOT NULL" in defs["uq_wiki_slug_family"]
