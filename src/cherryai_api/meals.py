"""Meal planning: weekly plans, recipes, and shopping lists.

This module owns the meal planning end to end: pydantic models, asyncpg data
access helpers, and the FastAPI router mounted under ``/api/meals``.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Literal

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator
from pydantic_ai import Agent, RunContext

from cherryai_api.agent import AgentDeps
from cherryai_api.auth import current_verified_user
from cherryai_api.db import Database
from cherryai_api.meal_units import (
    AggregatedIngredient,
    aggregate,
    canonicalize,
    format_display,
    packages_needed,
    to_canonical_unit_factor,
)
from cherryai_api.users import User

# AgentDeps is imported here for real (not TYPE_CHECKING-only) because
# pydantic-ai resolves tool function annotations at runtime via
# typing.get_type_hints() to build tool schemas — a TYPE_CHECKING-only
# import would leave `AgentDeps` undefined in this module's globals at call
# time despite `from __future__ import annotations` deferring the syntax
# check. agent.py avoids the resulting cycle by importing register_meal_tools
# from *this* module lazily, inside build_agent() rather than at its own
# module top-level (see the comment there).

# ------------------------------------------------------------------
# SQL
# ------------------------------------------------------------------

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

CREATE TABLE IF NOT EXISTS meal_plan_day_recipes (
    id UUID PRIMARY KEY,
    day_id UUID NOT NULL REFERENCES meal_plan_days(id) ON DELETE CASCADE,
    recipe_id UUID NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS meal_plan_day_recipes_day_idx
    ON meal_plan_day_recipes (day_id, sort_order);

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

# Additive schema evolution: CREATE TABLE IF NOT EXISTS (above) can't alter
# existing tables, so new columns on already-shipped tables go here as
# idempotent ALTER TABLE statements. Executed in order by Database.connect()
# right after CREATE_MEALS_TABLES.
MEALS_MIGRATIONS: list[str] = [
    "ALTER TABLE meal_plan_days ADD COLUMN IF NOT EXISTS consumed_at TIMESTAMPTZ",
    "ALTER TABLE shopping_list_items ADD COLUMN IF NOT EXISTS packages INTEGER",
    "ALTER TABLE shopping_list_items ADD COLUMN IF NOT EXISTS store_product_id UUID",
    "ALTER TABLE shopping_list_items ADD COLUMN IF NOT EXISTS store_name TEXT",
    "ALTER TABLE shopping_list_items ADD COLUMN IF NOT EXISTS package_label TEXT",
]


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


# ------------------------------------------------------------------
# Pydantic models — Meal Plans
# ------------------------------------------------------------------


def _require_monday(value: date) -> date:
    if value.weekday() != 0:
        raise ValueError("week_start must be a Monday")
    return value


class MealPlanCreate(BaseModel):
    name: str
    week_start: date

    @field_validator("week_start")
    @classmethod
    def _validate_week_start(cls, value: date) -> date:
        return _require_monday(value)


class MealPlanUpdate(BaseModel):
    name: str | None = None
    week_start: date | None = None

    @field_validator("week_start")
    @classmethod
    def _validate_week_start(cls, value: date | None) -> date | None:
        return _require_monday(value) if value is not None else None


class MealPlan(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    week_start: date
    created_at: datetime
    updated_at: datetime


class MealPlanListItem(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    week_start: date
    day_count: int
    recipe_count: int
    created_at: datetime
    updated_at: datetime


class RecipeRef(BaseModel):
    """Lightweight recipe reference for embedding in meal plan days."""

    id: uuid.UUID
    name: str


class MealPlanDayCreate(BaseModel):
    day_date: date
    meal_type: MealType = MealType.DINNER
    recipe_ids: list[uuid.UUID] = []
    notes: str = ""


class MealPlanDayUpdate(BaseModel):
    notes: str | None = None


class MealPlanDay(BaseModel):
    id: uuid.UUID
    plan_id: uuid.UUID
    day_date: date
    meal_type: str
    recipes: list[RecipeRef] = []
    notes: str
    sort_order: int
    consumed_at: datetime | None = None


# ------------------------------------------------------------------
# Pydantic models — Recipes
# ------------------------------------------------------------------


class RecipeIngredientCreate(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    notes: str = ""
    category: str = ""


class RecipeCreate(BaseModel):
    name: str
    description: str = ""
    instructions: str = ""
    prep_minutes: int | None = None
    cook_minutes: int | None = None
    servings: int = 4
    source_url: str | None = None
    ingredients: list[RecipeIngredientCreate] = []


class RecipeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    instructions: str | None = None
    prep_minutes: int | None = None
    cook_minutes: int | None = None
    servings: int | None = None
    source_url: str | None = None


class RecipeIngredient(BaseModel):
    id: uuid.UUID
    recipe_id: uuid.UUID
    name: str
    quantity: float | None
    unit: str | None
    notes: str
    category: str
    sort_order: int


class Recipe(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: str
    instructions: str
    prep_minutes: int | None
    cook_minutes: int | None
    servings: int
    source_url: str | None
    ingredients: list[RecipeIngredient] = []
    created_at: datetime
    updated_at: datetime


class RecipeListItem(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: str
    prep_minutes: int | None
    cook_minutes: int | None
    servings: int
    ingredient_count: int
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------
# Pydantic models — Shopping Lists
# ------------------------------------------------------------------


class ShoppingListCreate(BaseModel):
    name: str
    plan_id: uuid.UUID | None = None


class ShoppingListUpdate(BaseModel):
    name: str | None = None


class ShoppingListItemCreate(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    category: str = ""
    packages: int | None = None
    store_product_id: uuid.UUID | None = None
    store_name: str | None = None
    package_label: str | None = None


class ShoppingListItemUpdate(BaseModel):
    name: str | None = None
    quantity: float | None = None
    unit: str | None = None
    category: str | None = None
    purchased: bool | None = None


class ShoppingListItem(BaseModel):
    id: uuid.UUID
    list_id: uuid.UUID
    name: str
    quantity: float | None
    unit: str | None
    category: str
    purchased: bool
    sort_order: int
    packages: int | None = None
    store_product_id: uuid.UUID | None = None
    store_name: str | None = None
    package_label: str | None = None


class ShoppingList(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    plan_id: uuid.UUID | None
    items: list[ShoppingListItem] = []
    created_at: datetime
    updated_at: datetime


class ShoppingListListItem(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    plan_id: uuid.UUID | None
    item_total: int
    item_purchased: int
    created_at: datetime
    updated_at: datetime


class GenerateListRequest(BaseModel):
    """Body for POST /lists/generate: pick a plan selection, then generation options."""

    plan_ids: list[uuid.UUID] | None = None
    scope: Literal["future"] | None = None
    name: str | None = None
    deduct_pantry: bool = False
    store_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _validate_selection(self) -> GenerateListRequest:
        if bool(self.plan_ids) == (self.scope is not None):
            raise ValueError("Exactly one of plan_ids or scope must be provided")
        return self


# ------------------------------------------------------------------
# Pydantic models — Pantry
# ------------------------------------------------------------------


class PantryItemCreate(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    category: str = ""


class PantryItemUpdate(BaseModel):
    name: str | None = None
    quantity: float | None = None
    unit: str | None = None
    category: str | None = None


class PantryItem(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    quantity: float | None
    unit: str | None
    category: str
    updated_at: datetime


# ------------------------------------------------------------------
# Pydantic models — Consume / commit-to-pantry
# ------------------------------------------------------------------


class ConsumeRequest(BaseModel):
    force: bool = False


class ConsumeReportLine(BaseModel):
    name: str
    status: Literal["deducted", "insufficient", "not_tracked"]
    deducted_quantity: float | None
    unit: str | None


class ConsumeResponse(BaseModel):
    day: MealPlanDay
    report: list[ConsumeReportLine]


class CommitToPantryLine(BaseModel):
    name: str
    quantity: float | None
    unit: str | None


class CommitToPantryResponse(BaseModel):
    added: list[CommitToPantryLine]


class DayAlreadyConsumed(Exception):
    """Raised by :func:`consume_day` when ``consumed_at`` is set and ``force`` is False."""

    def __init__(self, day_id: uuid.UUID) -> None:
        self.day_id = day_id
        super().__init__(f"Day {day_id} is already marked consumed")


# ------------------------------------------------------------------
# Pydantic models — Stores & Store Products
# ------------------------------------------------------------------


class StoreCreate(BaseModel):
    name: str
    notes: str | None = None


class StoreUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None


class Store(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    notes: str | None
    created_at: datetime
    updated_at: datetime


class StoreProductCreate(BaseModel):
    ingredient_name: str
    product_name: str
    package_quantity: float
    package_unit: str
    price_cents: int | None = None
    notes: str | None = None


class StoreProductUpdate(BaseModel):
    ingredient_name: str | None = None
    product_name: str | None = None
    package_quantity: float | None = None
    package_unit: str | None = None
    price_cents: int | None = None
    notes: str | None = None


class StoreProduct(BaseModel):
    id: uuid.UUID
    store_id: uuid.UUID
    ingredient_name: str
    product_name: str
    package_quantity: float
    package_unit: str
    price_cents: int | None
    notes: str | None


# ------------------------------------------------------------------
# Data access — Meal Plans
# ------------------------------------------------------------------

_PLAN_COLUMNS = "id, name, owner_id, week_start, created_at, updated_at"
_PLAN_LIST_COLUMNS = """
    p.id, p.name, p.owner_id, p.week_start, p.created_at, p.updated_at,
    COALESCE(dc.day_count, 0) AS day_count,
    COALESCE(dc.recipe_count, 0) AS recipe_count
"""


async def list_meal_plans(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[MealPlanListItem]:
    rows = await pool.fetch(
        f"""
        SELECT {_PLAN_LIST_COLUMNS}
          FROM meal_plans p
          LEFT JOIN LATERAL (
              SELECT count(*) AS day_count,
                     count(*) FILTER (WHERE recipe_id IS NOT NULL) AS recipe_count
                FROM meal_plan_days
               WHERE plan_id = p.id
          ) dc ON true
         WHERE p.owner_id = $1
         ORDER BY p.week_start DESC
        """,
        owner_id,
    )
    return [MealPlanListItem(**dict(row)) for row in rows]


async def get_meal_plan(
    pool: asyncpg.Pool, owner_id: uuid.UUID, plan_id: uuid.UUID
) -> MealPlan | None:
    row = await pool.fetchrow(
        f"SELECT {_PLAN_COLUMNS} FROM meal_plans WHERE id = $1 AND owner_id = $2",
        plan_id,
        owner_id,
    )
    return MealPlan(**dict(row)) if row else None


async def create_meal_plan(
    pool: asyncpg.Pool, owner_id: uuid.UUID, data: MealPlanCreate
) -> MealPlan:
    name = data.name.strip()
    if not name:
        raise ValueError("Meal plan name must not be empty")
    row = await pool.fetchrow(
        f"INSERT INTO meal_plans (id, name, owner_id, week_start) "
        f"VALUES ($1, $2, $3, $4) RETURNING {_PLAN_COLUMNS}",
        uuid.uuid4(),
        name,
        owner_id,
        data.week_start,
    )
    return MealPlan(**dict(row))


async def update_meal_plan(
    pool: asyncpg.Pool, owner_id: uuid.UUID, plan_id: uuid.UUID, data: MealPlanUpdate
) -> MealPlan | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Meal plan name must not be empty")
    row = await pool.fetchrow(
        f"UPDATE meal_plans SET "
        f"name = COALESCE($3, name), "
        f"week_start = COALESCE($4, week_start), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_PLAN_COLUMNS}",
        plan_id,
        owner_id,
        name,
        data.week_start,
    )
    return MealPlan(**dict(row)) if row else None


async def delete_meal_plan(pool: asyncpg.Pool, owner_id: uuid.UUID, plan_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM meal_plans WHERE id = $1 AND owner_id = $2", plan_id, owner_id
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Meal Plan Days
# ------------------------------------------------------------------

_DAY_COLUMNS = "id, plan_id, day_date, meal_type, notes, sort_order, consumed_at"


async def _load_day_recipes(pool: asyncpg.Pool, day_id: uuid.UUID) -> list[RecipeRef]:
    rows = await pool.fetch(
        "SELECT r.id, r.name "
        "FROM meal_plan_day_recipes dr "
        "JOIN recipes r ON r.id = dr.recipe_id "
        "WHERE dr.day_id = $1 "
        "ORDER BY dr.sort_order",
        day_id,
    )
    return [RecipeRef(**dict(row)) for row in rows]


async def _load_days_with_recipes(pool: asyncpg.Pool, plan_id: uuid.UUID) -> list[MealPlanDay]:
    """Load all days for a plan with their recipes pre-joined."""
    day_rows = await pool.fetch(
        f"SELECT {_DAY_COLUMNS} FROM meal_plan_days "
        f"WHERE plan_id = $1 ORDER BY day_date, sort_order",
        plan_id,
    )
    # Fetch all recipe refs for all days in one query
    day_ids = [row["id"] for row in day_rows]
    recipes_by_day: dict[uuid.UUID, list[RecipeRef]] = {did: [] for did in day_ids}
    if day_ids:
        recipe_rows = await pool.fetch(
            "SELECT dr.day_id, r.id, r.name "
            "FROM meal_plan_day_recipes dr "
            "JOIN recipes r ON r.id = dr.recipe_id "
            "WHERE dr.day_id = ANY($1::uuid[]) "
            "ORDER BY dr.sort_order",
            day_ids,
        )
        for rr in recipe_rows:
            recipes_by_day[rr["day_id"]].append(RecipeRef(id=rr["id"], name=rr["name"]))
    return [MealPlanDay(**{**dict(row), "recipes": recipes_by_day[row["id"]]}) for row in day_rows]


async def list_plan_days(pool: asyncpg.Pool, plan_id: uuid.UUID) -> list[MealPlanDay]:
    return await _load_days_with_recipes(pool, plan_id)


async def upsert_plan_day(
    pool: asyncpg.Pool, plan_id: uuid.UUID, data: MealPlanDayCreate
) -> MealPlanDay:
    # Get next sort_order for this day_date + meal_type
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM meal_plan_days "
        "WHERE plan_id = $1 AND day_date = $2 AND meal_type = $3",
        plan_id,
        data.day_date,
        data.meal_type.value,
    )
    sort_order = (max_order or -1) + 1

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"INSERT INTO meal_plan_days "
                f"(id, plan_id, day_date, meal_type, notes, sort_order) "
                f"VALUES ($1, $2, $3, $4, $5, $6) "
                f"RETURNING {_DAY_COLUMNS}",
                uuid.uuid4(),
                plan_id,
                data.day_date,
                data.meal_type.value,
                data.notes,
                sort_order,
            )
            day = dict(row)
            # Insert initial recipes
            recipes: list[RecipeRef] = []
            for i, rid in enumerate(data.recipe_ids):
                await conn.execute(
                    "INSERT INTO meal_plan_day_recipes (id, day_id, recipe_id, sort_order) "
                    "VALUES ($1, $2, $3, $4)",
                    uuid.uuid4(),
                    day["id"],
                    rid,
                    i,
                )
                # Fetch recipe name
                name_row = await conn.fetchrow("SELECT name FROM recipes WHERE id = $1", rid)
                recipes.append(RecipeRef(id=rid, name=name_row["name"] if name_row else "Unknown"))
            day["recipes"] = recipes
    return MealPlanDay(**day)


async def update_plan_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID, data: MealPlanDayUpdate
) -> MealPlanDay | None:
    row = await pool.fetchrow(
        "UPDATE meal_plan_days d SET "
        "notes = COALESCE($3, d.notes) "
        "FROM meal_plans p "
        "WHERE d.id = $1 AND d.plan_id = p.id AND p.owner_id = $2 "
        "RETURNING d.id, d.plan_id, d.day_date, d.meal_type, d.notes, d.sort_order, d.consumed_at",
        day_id,
        owner_id,
        data.notes,
    )
    if row is None:
        return None
    day = dict(row)
    day["recipes"] = await _load_day_recipes(pool, day_id)
    return MealPlanDay(**day)


async def delete_plan_day(pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM meal_plan_days d USING meal_plans p "
        "WHERE d.id = $1 AND d.plan_id = p.id AND p.owner_id = $2",
        day_id,
        owner_id,
    )
    return result.endswith("1")


async def get_plan_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID
) -> MealPlanDay | None:
    """Fetch a single day (with its recipes), verifying ownership through its plan."""
    row = await pool.fetchrow(
        "SELECT d.id, d.plan_id, d.day_date, d.meal_type, d.notes, d.sort_order, d.consumed_at "
        "FROM meal_plan_days d JOIN meal_plans p ON p.id = d.plan_id "
        "WHERE d.id = $1 AND p.owner_id = $2",
        day_id,
        owner_id,
    )
    if row is None:
        return None
    day = dict(row)
    day["recipes"] = await _load_day_recipes(pool, day_id)
    return MealPlanDay(**day)


# ------------------------------------------------------------------
# Data access — Day Recipes (many-to-many)
# ------------------------------------------------------------------


async def add_recipe_to_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID, recipe_id: uuid.UUID
) -> RecipeRef | None:
    owns_day = await pool.fetchval(
        "SELECT 1 FROM meal_plan_days d JOIN meal_plans p ON p.id = d.plan_id "
        "WHERE d.id = $1 AND p.owner_id = $2",
        day_id,
        owner_id,
    )
    if owns_day is None:
        return None
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM meal_plan_day_recipes WHERE day_id = $1",
        day_id,
    )
    sort_order = (max_order or -1) + 1
    await pool.execute(
        "INSERT INTO meal_plan_day_recipes (id, day_id, recipe_id, sort_order) "
        "VALUES ($1, $2, $3, $4)",
        uuid.uuid4(),
        day_id,
        recipe_id,
        sort_order,
    )
    name_row = await pool.fetchrow("SELECT name FROM recipes WHERE id = $1", recipe_id)
    return RecipeRef(id=recipe_id, name=name_row["name"] if name_row else "Unknown")


async def remove_recipe_from_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID, recipe_id: uuid.UUID
) -> bool:
    result = await pool.execute(
        "DELETE FROM meal_plan_day_recipes dr "
        "USING meal_plan_days d, meal_plans p "
        "WHERE dr.day_id = $1 AND dr.recipe_id = $2 "
        "AND dr.day_id = d.id AND d.plan_id = p.id AND p.owner_id = $3",
        day_id,
        recipe_id,
        owner_id,
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Recipes
# ------------------------------------------------------------------

_RECIPE_COLUMNS = (
    "id, owner_id, name, description, instructions, "
    "prep_minutes, cook_minutes, servings, source_url, created_at, updated_at"
)
_RECIPE_LIST_COLUMNS = f"""
    r.{_RECIPE_COLUMNS},
    COALESCE(ic.ingredient_count, 0) AS ingredient_count
"""


async def list_recipes(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[RecipeListItem]:
    rows = await pool.fetch(
        f"""
        SELECT {_RECIPE_LIST_COLUMNS}
          FROM recipes r
          LEFT JOIN LATERAL (
              SELECT count(*) AS ingredient_count
                FROM recipe_ingredients
               WHERE recipe_id = r.id
          ) ic ON true
         WHERE r.owner_id = $1
         ORDER BY r.updated_at DESC
        """,
        owner_id,
    )
    return [RecipeListItem(**dict(row)) for row in rows]


async def get_recipe(
    pool: asyncpg.Pool, owner_id: uuid.UUID, recipe_id: uuid.UUID
) -> Recipe | None:
    row = await pool.fetchrow(
        f"SELECT {_RECIPE_COLUMNS} FROM recipes WHERE id = $1 AND owner_id = $2",
        recipe_id,
        owner_id,
    )
    if row is None:
        return None
    recipe = dict(row)
    ingredient_rows = await pool.fetch(
        "SELECT id, recipe_id, name, quantity, unit, notes, category, sort_order "
        "FROM recipe_ingredients WHERE recipe_id = $1 ORDER BY sort_order",
        recipe_id,
    )
    recipe["ingredients"] = [RecipeIngredient(**dict(ir)) for ir in ingredient_rows]
    return Recipe(**recipe)


async def create_recipe(pool: asyncpg.Pool, owner_id: uuid.UUID, data: RecipeCreate) -> Recipe:
    name = data.name.strip()
    if not name:
        raise ValueError("Recipe name must not be empty")

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"INSERT INTO recipes "
                f"(id, owner_id, name, description, instructions, "
                f"prep_minutes, cook_minutes, servings, source_url) "
                f"VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
                f"RETURNING {_RECIPE_COLUMNS}",
                uuid.uuid4(),
                owner_id,
                name,
                data.description,
                data.instructions,
                data.prep_minutes,
                data.cook_minutes,
                data.servings,
                data.source_url,
            )
            recipe = dict(row)
            ingredients: list[RecipeIngredient] = []
            for i, ing in enumerate(data.ingredients):
                ir = await conn.fetchrow(
                    "INSERT INTO recipe_ingredients "
                    "(id, recipe_id, name, quantity, unit, notes, category, sort_order) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                    "RETURNING id, recipe_id, name, quantity, unit, notes, category, sort_order",
                    uuid.uuid4(),
                    recipe["id"],
                    ing.name.strip(),
                    ing.quantity,
                    ing.unit,
                    ing.notes,
                    ing.category,
                    i,
                )
                ingredients.append(RecipeIngredient(**dict(ir)))
            recipe["ingredients"] = ingredients
    return Recipe(**recipe)


async def update_recipe(
    pool: asyncpg.Pool, owner_id: uuid.UUID, recipe_id: uuid.UUID, data: RecipeUpdate
) -> Recipe | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Recipe name must not be empty")

    row = await pool.fetchrow(
        f"UPDATE recipes SET "
        f"name = COALESCE($3, name), "
        f"description = COALESCE($4, description), "
        f"instructions = COALESCE($5, instructions), "
        f"prep_minutes = COALESCE($6, prep_minutes), "
        f"cook_minutes = COALESCE($7, cook_minutes), "
        f"servings = COALESCE($8, servings), "
        f"source_url = COALESCE($9, source_url), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_RECIPE_COLUMNS}",
        recipe_id,
        owner_id,
        name,
        data.description,
        data.instructions,
        data.prep_minutes,
        data.cook_minutes,
        data.servings,
        data.source_url,
    )
    if row is None:
        return None
    recipe = dict(row)
    ingredient_rows = await pool.fetch(
        "SELECT id, recipe_id, name, quantity, unit, notes, category, sort_order "
        "FROM recipe_ingredients WHERE recipe_id = $1 ORDER BY sort_order",
        recipe_id,
    )
    recipe["ingredients"] = [RecipeIngredient(**dict(ir)) for ir in ingredient_rows]
    return Recipe(**recipe)


async def delete_recipe(pool: asyncpg.Pool, owner_id: uuid.UUID, recipe_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM recipes WHERE id = $1 AND owner_id = $2", recipe_id, owner_id
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Recipe Ingredients
# ------------------------------------------------------------------


class IngredientUpdate(BaseModel):
    name: str | None = None
    quantity: float | None = None
    unit: str | None = None
    notes: str | None = None
    category: str | None = None


async def add_ingredient(
    pool: asyncpg.Pool, recipe_id: uuid.UUID, data: RecipeIngredientCreate
) -> RecipeIngredient:
    name = data.name.strip()
    if not name:
        raise ValueError("Ingredient name must not be empty")
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM recipe_ingredients WHERE recipe_id = $1",
        recipe_id,
    )
    sort_order = (max_order or -1) + 1
    row = await pool.fetchrow(
        "INSERT INTO recipe_ingredients "
        "(id, recipe_id, name, quantity, unit, notes, category, sort_order) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
        "RETURNING id, recipe_id, name, quantity, unit, notes, category, sort_order",
        uuid.uuid4(),
        recipe_id,
        name,
        data.quantity,
        data.unit,
        data.notes,
        data.category,
        sort_order,
    )
    return RecipeIngredient(**dict(row))


async def update_ingredient(
    pool: asyncpg.Pool, owner_id: uuid.UUID, ingredient_id: uuid.UUID, data: IngredientUpdate
) -> RecipeIngredient | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Ingredient name must not be empty")
    row = await pool.fetchrow(
        "UPDATE recipe_ingredients i SET "
        "name = COALESCE($3, i.name), "
        "quantity = COALESCE($4, i.quantity), "
        "unit = COALESCE($5, i.unit), "
        "notes = COALESCE($6, i.notes), "
        "category = COALESCE($7, i.category) "
        "FROM recipes r "
        "WHERE i.id = $1 AND i.recipe_id = r.id AND r.owner_id = $2 "
        "RETURNING i.id, i.recipe_id, i.name, i.quantity, i.unit, i.notes, i.category, "
        "i.sort_order",
        ingredient_id,
        owner_id,
        name,
        data.quantity,
        data.unit,
        data.notes,
        data.category,
    )
    return RecipeIngredient(**dict(row)) if row else None


async def delete_ingredient(
    pool: asyncpg.Pool, owner_id: uuid.UUID, ingredient_id: uuid.UUID
) -> bool:
    result = await pool.execute(
        "DELETE FROM recipe_ingredients i USING recipes r "
        "WHERE i.id = $1 AND i.recipe_id = r.id AND r.owner_id = $2",
        ingredient_id,
        owner_id,
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Shopping Lists
# ------------------------------------------------------------------

_LIST_COLUMNS = "id, owner_id, name, plan_id, created_at, updated_at"
_LIST_LIST_COLUMNS = """
    l.id, l.owner_id, l.name, l.plan_id, l.created_at, l.updated_at,
    COALESCE(ic.item_total, 0) AS item_total,
    COALESCE(ic.item_purchased, 0) AS item_purchased
"""
_ITEM_COLUMNS = (
    "id, list_id, name, quantity, unit, category, purchased, sort_order, "
    "packages, store_product_id, store_name, package_label"
)


async def list_shopping_lists(
    pool: asyncpg.Pool, owner_id: uuid.UUID
) -> list[ShoppingListListItem]:
    rows = await pool.fetch(
        f"""
        SELECT {_LIST_LIST_COLUMNS}
          FROM shopping_lists l
          LEFT JOIN LATERAL (
              SELECT count(*) AS item_total,
                     count(*) FILTER (WHERE purchased) AS item_purchased
                FROM shopping_list_items
               WHERE list_id = l.id
          ) ic ON true
         WHERE l.owner_id = $1
         ORDER BY l.updated_at DESC
        """,
        owner_id,
    )
    return [ShoppingListListItem(**dict(row)) for row in rows]


async def get_shopping_list(
    pool: asyncpg.Pool, owner_id: uuid.UUID, list_id: uuid.UUID
) -> ShoppingList | None:
    row = await pool.fetchrow(
        f"SELECT {_LIST_COLUMNS} FROM shopping_lists WHERE id = $1 AND owner_id = $2",
        list_id,
        owner_id,
    )
    if row is None:
        return None
    slist = dict(row)
    item_rows = await pool.fetch(
        f"SELECT {_ITEM_COLUMNS} FROM shopping_list_items WHERE list_id = $1 ORDER BY sort_order",
        list_id,
    )
    slist["items"] = [ShoppingListItem(**dict(ir)) for ir in item_rows]
    return ShoppingList(**slist)


async def create_shopping_list(
    pool: asyncpg.Pool, owner_id: uuid.UUID, data: ShoppingListCreate
) -> ShoppingList:
    name = data.name.strip()
    if not name:
        raise ValueError("Shopping list name must not be empty")
    row = await pool.fetchrow(
        f"INSERT INTO shopping_lists (id, owner_id, name, plan_id) "
        f"VALUES ($1, $2, $3, $4) RETURNING {_LIST_COLUMNS}",
        uuid.uuid4(),
        owner_id,
        name,
        data.plan_id,
    )
    result = dict(row)
    result["items"] = []
    return ShoppingList(**result)


async def update_shopping_list(
    pool: asyncpg.Pool, owner_id: uuid.UUID, list_id: uuid.UUID, data: ShoppingListUpdate
) -> ShoppingList | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Shopping list name must not be empty")
    row = await pool.fetchrow(
        f"UPDATE shopping_lists SET "
        f"name = COALESCE($3, name), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_LIST_COLUMNS}",
        list_id,
        owner_id,
        name,
    )
    if row is None:
        return None
    result = dict(row)
    item_rows = await pool.fetch(
        f"SELECT {_ITEM_COLUMNS} FROM shopping_list_items WHERE list_id = $1 ORDER BY sort_order",
        list_id,
    )
    result["items"] = [ShoppingListItem(**dict(ir)) for ir in item_rows]
    return ShoppingList(**result)


async def delete_shopping_list(pool: asyncpg.Pool, owner_id: uuid.UUID, list_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM shopping_lists WHERE id = $1 AND owner_id = $2", list_id, owner_id
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Shopping List Items
# ------------------------------------------------------------------


async def add_list_item(
    pool: asyncpg.Pool, list_id: uuid.UUID, data: ShoppingListItemCreate
) -> ShoppingListItem:
    name = data.name.strip()
    if not name:
        raise ValueError("Item name must not be empty")
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM shopping_list_items WHERE list_id = $1",
        list_id,
    )
    sort_order = (max_order or -1) + 1
    row = await pool.fetchrow(
        "INSERT INTO shopping_list_items "
        "(id, list_id, name, quantity, unit, category, sort_order, "
        "packages, store_product_id, store_name, package_label) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
        f"RETURNING {_ITEM_COLUMNS}",
        uuid.uuid4(),
        list_id,
        name,
        data.quantity,
        data.unit,
        data.category,
        sort_order,
        data.packages,
        data.store_product_id,
        data.store_name,
        data.package_label,
    )
    return ShoppingListItem(**dict(row))


async def update_list_item(
    pool: asyncpg.Pool, owner_id: uuid.UUID, item_id: uuid.UUID, data: ShoppingListItemUpdate
) -> ShoppingListItem | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Item name must not be empty")
    row = await pool.fetchrow(
        "UPDATE shopping_list_items it SET "
        "name = COALESCE($3, it.name), "
        "quantity = COALESCE($4, it.quantity), "
        "unit = COALESCE($5, it.unit), "
        "category = COALESCE($6, it.category), "
        "purchased = COALESCE($7, it.purchased) "
        "FROM shopping_lists l "
        "WHERE it.id = $1 AND it.list_id = l.id AND l.owner_id = $2 "
        "RETURNING it.id, it.list_id, it.name, it.quantity, it.unit, it.category, "
        "it.purchased, it.sort_order, it.packages, it.store_product_id, it.store_name, "
        "it.package_label",
        item_id,
        owner_id,
        name,
        data.quantity,
        data.unit,
        data.category,
        data.purchased,
    )
    return ShoppingListItem(**dict(row)) if row else None


async def delete_list_item(pool: asyncpg.Pool, owner_id: uuid.UUID, item_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM shopping_list_items it USING shopping_lists l "
        "WHERE it.id = $1 AND it.list_id = l.id AND l.owner_id = $2",
        item_id,
        owner_id,
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Pantry
# ------------------------------------------------------------------

_PANTRY_COLUMNS = "id, owner_id, name, quantity, unit, category, updated_at"


async def list_pantry_items(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[PantryItem]:
    rows = await pool.fetch(
        f"SELECT {_PANTRY_COLUMNS} FROM pantry_items WHERE owner_id = $1 ORDER BY name",
        owner_id,
    )
    return [PantryItem(**dict(row)) for row in rows]


async def _find_matching_pantry_row(
    executor, owner_id: uuid.UUID, name: str, dimension: str
) -> asyncpg.Record | None:
    """Find the owner's pantry row for `name` whose unit shares `dimension`, if any.

    Uniqueness per (owner, normalized name, dimension) is enforced here in
    code rather than a DB constraint, since "dimension" is derived from the
    free-text unit via :func:`canonicalize`, not a stored column.
    """
    rows = await executor.fetch(
        "SELECT id, quantity, unit FROM pantry_items "
        "WHERE owner_id = $1 AND lower(trim(name)) = $2",
        owner_id,
        name.strip().lower(),
    )
    for row in rows:
        if canonicalize(None, row["unit"])[1] == dimension:
            return row
    return None


async def upsert_pantry_item(
    pool: asyncpg.Pool, owner_id: uuid.UUID, data: PantryItemCreate
) -> PantryItem:
    """Insert a pantry item, or add its quantity onto an existing (owner, name, dimension) match.

    Category on a merge is overwritten by the new value (last-write-wins);
    quantity stays ``None`` only when both the existing row and the new data
    have no quantity, otherwise a missing quantity counts as 0 for the sum.
    """
    name = data.name.strip()
    if not name:
        raise ValueError("Pantry item name must not be empty")
    _, dimension, _ = canonicalize(None, data.unit)
    match = await _find_matching_pantry_row(pool, owner_id, name, dimension)

    if match is None:
        row = await pool.fetchrow(
            f"INSERT INTO pantry_items (id, owner_id, name, quantity, unit, category) "
            f"VALUES ($1, $2, $3, $4, $5, $6) RETURNING {_PANTRY_COLUMNS}",
            uuid.uuid4(),
            owner_id,
            name,
            data.quantity,
            data.unit,
            data.category,
        )
        return PantryItem(**dict(row))

    if match["quantity"] is None and data.quantity is None:
        merged_qty, merged_unit = None, match["unit"] or data.unit
    else:
        existing_canonical, _, _ = canonicalize(match["quantity"] or 0.0, match["unit"])
        added_canonical, _, _ = canonicalize(data.quantity or 0.0, data.unit)
        total = (existing_canonical or 0.0) + (added_canonical or 0.0)
        merged_unit = match["unit"] or data.unit
        factor = to_canonical_unit_factor(merged_unit)
        merged_qty, merged_unit = format_display(total / factor, merged_unit)

    row = await pool.fetchrow(
        f"UPDATE pantry_items SET quantity = $2, unit = $3, category = $4, updated_at = now() "
        f"WHERE id = $1 RETURNING {_PANTRY_COLUMNS}",
        match["id"],
        merged_qty,
        merged_unit,
        data.category,
    )
    return PantryItem(**dict(row))


async def update_pantry_item(
    pool: asyncpg.Pool, owner_id: uuid.UUID, item_id: uuid.UUID, data: PantryItemUpdate
) -> PantryItem | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Pantry item name must not be empty")
    row = await pool.fetchrow(
        f"UPDATE pantry_items SET "
        f"name = COALESCE($3, name), "
        f"quantity = COALESCE($4, quantity), "
        f"unit = COALESCE($5, unit), "
        f"category = COALESCE($6, category), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_PANTRY_COLUMNS}",
        item_id,
        owner_id,
        name,
        data.quantity,
        data.unit,
        data.category,
    )
    return PantryItem(**dict(row)) if row else None


async def delete_pantry_item(pool: asyncpg.Pool, owner_id: uuid.UUID, item_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM pantry_items WHERE id = $1 AND owner_id = $2", item_id, owner_id
    )
    return result.endswith("1")


async def _add_to_pantry(
    executor, owner_id: uuid.UUID, name: str, quantity: float | None, unit: str | None
) -> None:
    """Add `quantity unit` of `name` into the owner's pantry (upsert-add), best-effort.

    `executor` is a pool or an open connection (both share asyncpg's
    fetch/fetchrow/execute interface). A no-op when `quantity` is None or
    `name` is blank — there's nothing quantifiable to add.
    """
    name = name.strip()
    if quantity is None or not name:
        return
    _, dimension, _ = canonicalize(None, unit)
    match = await _find_matching_pantry_row(executor, owner_id, name, dimension)
    added_canonical, _, _ = canonicalize(quantity, unit)

    if match is None:
        await executor.execute(
            "INSERT INTO pantry_items (id, owner_id, name, quantity, unit, category) "
            "VALUES ($1, $2, $3, $4, $5, '')",
            uuid.uuid4(),
            owner_id,
            name,
            quantity,
            unit,
        )
        return

    existing_canonical, _, _ = canonicalize(match["quantity"] or 0.0, match["unit"])
    total = (existing_canonical or 0.0) + (added_canonical or 0.0)
    display_unit = match["unit"] or unit
    factor = to_canonical_unit_factor(display_unit)
    new_qty, new_unit = format_display(total / factor, display_unit)
    await executor.execute(
        "UPDATE pantry_items SET quantity = $2, unit = $3, updated_at = now() WHERE id = $1",
        match["id"],
        new_qty,
        new_unit,
    )


async def _deduct_from_pantry(
    conn, owner_id: uuid.UUID, name: str, quantity: float | None, unit: str | None
) -> ConsumeReportLine:
    """Subtract `quantity unit` of `name` from the owner's pantry, clamped at 0.

    Runs on an open transaction connection. Returns a report line: "deducted"
    (enough stock, or nothing was needed), "insufficient" (stock existed but
    ran out, clamped to 0), or "not_tracked" (no matching pantry row, or the
    matching row itself has no quantity to draw down).
    """
    needed_canonical, dimension, _ = canonicalize(quantity, unit)
    match = await _find_matching_pantry_row(conn, owner_id, name, dimension)
    if match is None:
        return ConsumeReportLine(name=name, status="not_tracked", deducted_quantity=None, unit=unit)

    have_canonical, _, _ = canonicalize(match["quantity"], match["unit"])
    if have_canonical is None:
        return ConsumeReportLine(name=name, status="not_tracked", deducted_quantity=None, unit=unit)

    need = needed_canonical or 0.0
    deducted_canonical = min(have_canonical, need)
    remaining_canonical = max(have_canonical - need, 0.0)

    factor = to_canonical_unit_factor(match["unit"])
    remaining_qty, remaining_unit = format_display(remaining_canonical / factor, match["unit"])
    await conn.execute(
        "UPDATE pantry_items SET quantity = $2, updated_at = now() WHERE id = $1",
        match["id"],
        remaining_qty,
    )

    deducted_qty, deducted_unit = format_display(deducted_canonical / factor, match["unit"])
    status: Literal["deducted", "insufficient"] = (
        "deducted" if have_canonical >= need else "insufficient"
    )
    return ConsumeReportLine(
        name=name,
        status=status,
        deducted_quantity=deducted_qty,
        unit=deducted_unit or remaining_unit,
    )


async def _day_recipe_ingredients(conn, day_id: uuid.UUID):
    return await conn.fetch(
        "SELECT ri.name, ri.quantity, ri.unit "
        "FROM meal_plan_day_recipes dr "
        "JOIN recipe_ingredients ri ON ri.recipe_id = dr.recipe_id "
        "WHERE dr.day_id = $1",
        day_id,
    )


async def consume_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID, force: bool = False
) -> tuple[MealPlanDay, list[ConsumeReportLine]] | None:
    """Mark a day consumed, deducting every ingredient of every recipe on it from pantry.

    Returns None if the day isn't found or isn't owned by `owner_id`. Raises
    :class:`DayAlreadyConsumed` when the day is already marked consumed and
    `force` is False.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            day_row = await conn.fetchrow(
                "SELECT d.id, d.consumed_at "
                "FROM meal_plan_days d JOIN meal_plans p ON p.id = d.plan_id "
                "WHERE d.id = $1 AND p.owner_id = $2 "
                "FOR UPDATE OF d",
                day_id,
                owner_id,
            )
            if day_row is None:
                return None
            if day_row["consumed_at"] is not None and not force:
                raise DayAlreadyConsumed(day_id)

            ingredient_rows = await _day_recipe_ingredients(conn, day_id)
            report = [
                await _deduct_from_pantry(conn, owner_id, ing["name"], ing["quantity"], ing["unit"])
                for ing in ingredient_rows
            ]
            await conn.execute(
                "UPDATE meal_plan_days SET consumed_at = now() WHERE id = $1", day_id
            )

    day = await get_plan_day(pool, owner_id, day_id)
    assert day is not None  # verified ownership above, inside the same transaction
    return day, report


async def unconsume_day(
    pool: asyncpg.Pool, owner_id: uuid.UUID, day_id: uuid.UUID
) -> MealPlanDay | None:
    """Clear `consumed_at` and best-effort restore the day's ingredients to pantry.

    A no-op restoration (but still returns the day) when the day was never
    consumed — there's nothing to credit back. Returns None if the day isn't
    found or isn't owned by `owner_id`.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            day_row = await conn.fetchrow(
                "SELECT d.id, d.consumed_at "
                "FROM meal_plan_days d JOIN meal_plans p ON p.id = d.plan_id "
                "WHERE d.id = $1 AND p.owner_id = $2 "
                "FOR UPDATE OF d",
                day_id,
                owner_id,
            )
            if day_row is None:
                return None
            if day_row["consumed_at"] is not None:
                ingredient_rows = await _day_recipe_ingredients(conn, day_id)
                for ing in ingredient_rows:
                    await _add_to_pantry(conn, owner_id, ing["name"], ing["quantity"], ing["unit"])
            await conn.execute("UPDATE meal_plan_days SET consumed_at = NULL WHERE id = $1", day_id)

    return await get_plan_day(pool, owner_id, day_id)


async def commit_list_to_pantry(
    pool: asyncpg.Pool, owner_id: uuid.UUID, list_id: uuid.UUID
) -> list[CommitToPantryLine] | None:
    """Add a shopping list's purchased items into pantry stock (upsert-add).

    When an item carries package data (`packages` + `store_product_id` both
    set), the family bought whole packages — credit `packages *
    package_quantity` in `package_unit` to pantry, not the raw needed
    `quantity`/`unit` (e.g. buying 2 x 5 lb for an 8 lb need credits 10 lb,
    not 8). Items without package data keep crediting their raw
    quantity/unit.

    Returns None if the list isn't found or isn't owned by `owner_id`.
    """
    slist = await get_shopping_list(pool, owner_id, list_id)
    if slist is None:
        return None

    added: list[CommitToPantryLine] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for item in slist.items:
                if not item.purchased:
                    continue
                credit_qty, credit_unit = item.quantity, item.unit
                if item.packages is not None and item.store_product_id is not None:
                    product = await conn.fetchrow(
                        "SELECT package_quantity, package_unit FROM store_products WHERE id = $1",
                        item.store_product_id,
                    )
                    if product is not None:
                        credit_qty = item.packages * product["package_quantity"]
                        credit_unit = product["package_unit"]
                if credit_qty is None:
                    continue
                await _add_to_pantry(conn, owner_id, item.name, credit_qty, credit_unit)
                added.append(
                    CommitToPantryLine(name=item.name, quantity=credit_qty, unit=credit_unit)
                )
    return added


# ------------------------------------------------------------------
# Data access — Stores
# ------------------------------------------------------------------

_STORE_COLUMNS = "id, owner_id, name, notes, created_at, updated_at"
_PRODUCT_COLUMNS = (
    "id, store_id, ingredient_name, product_name, package_quantity, "
    "package_unit, price_cents, notes"
)


async def list_stores(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[Store]:
    rows = await pool.fetch(
        f"SELECT {_STORE_COLUMNS} FROM stores WHERE owner_id = $1 ORDER BY name", owner_id
    )
    return [Store(**dict(row)) for row in rows]


async def get_store(pool: asyncpg.Pool, owner_id: uuid.UUID, store_id: uuid.UUID) -> Store | None:
    row = await pool.fetchrow(
        f"SELECT {_STORE_COLUMNS} FROM stores WHERE id = $1 AND owner_id = $2",
        store_id,
        owner_id,
    )
    return Store(**dict(row)) if row else None


async def create_store(pool: asyncpg.Pool, owner_id: uuid.UUID, data: StoreCreate) -> Store:
    name = data.name.strip()
    if not name:
        raise ValueError("Store name must not be empty")
    row = await pool.fetchrow(
        f"INSERT INTO stores (id, owner_id, name, notes) "
        f"VALUES ($1, $2, $3, $4) RETURNING {_STORE_COLUMNS}",
        uuid.uuid4(),
        owner_id,
        name,
        data.notes,
    )
    return Store(**dict(row))


async def update_store(
    pool: asyncpg.Pool, owner_id: uuid.UUID, store_id: uuid.UUID, data: StoreUpdate
) -> Store | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Store name must not be empty")
    row = await pool.fetchrow(
        f"UPDATE stores SET "
        f"name = COALESCE($3, name), "
        f"notes = COALESCE($4, notes), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_STORE_COLUMNS}",
        store_id,
        owner_id,
        name,
        data.notes,
    )
    return Store(**dict(row)) if row else None


async def delete_store(pool: asyncpg.Pool, owner_id: uuid.UUID, store_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM stores WHERE id = $1 AND owner_id = $2", store_id, owner_id
    )
    return result.endswith("1")


async def list_store_products(
    pool: asyncpg.Pool, owner_id: uuid.UUID, store_id: uuid.UUID
) -> list[StoreProduct] | None:
    """Returns None if the store isn't found or isn't owned by `owner_id`."""
    store = await get_store(pool, owner_id, store_id)
    if store is None:
        return None
    rows = await pool.fetch(
        f"SELECT {_PRODUCT_COLUMNS} FROM store_products "
        f"WHERE store_id = $1 ORDER BY ingredient_name",
        store_id,
    )
    return [StoreProduct(**dict(row)) for row in rows]


async def add_store_product(
    pool: asyncpg.Pool, owner_id: uuid.UUID, store_id: uuid.UUID, data: StoreProductCreate
) -> StoreProduct | None:
    """Returns None if the store isn't found or isn't owned by `owner_id`."""
    store = await get_store(pool, owner_id, store_id)
    if store is None:
        return None
    ingredient_name = data.ingredient_name.strip()
    product_name = data.product_name.strip()
    if not ingredient_name:
        raise ValueError("Ingredient name must not be empty")
    if not product_name:
        raise ValueError("Product name must not be empty")
    row = await pool.fetchrow(
        f"INSERT INTO store_products "
        f"(id, store_id, ingredient_name, product_name, package_quantity, package_unit, "
        f"price_cents, notes) "
        f"VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING {_PRODUCT_COLUMNS}",
        uuid.uuid4(),
        store_id,
        ingredient_name,
        product_name,
        data.package_quantity,
        data.package_unit,
        data.price_cents,
        data.notes,
    )
    return StoreProduct(**dict(row))


async def update_store_product(
    pool: asyncpg.Pool, owner_id: uuid.UUID, product_id: uuid.UUID, data: StoreProductUpdate
) -> StoreProduct | None:
    ingredient_name = data.ingredient_name.strip() if data.ingredient_name is not None else None
    product_name = data.product_name.strip() if data.product_name is not None else None
    if data.ingredient_name is not None and not ingredient_name:
        raise ValueError("Ingredient name must not be empty")
    if data.product_name is not None and not product_name:
        raise ValueError("Product name must not be empty")
    row = await pool.fetchrow(
        "UPDATE store_products sp SET "
        "ingredient_name = COALESCE($3, sp.ingredient_name), "
        "product_name = COALESCE($4, sp.product_name), "
        "package_quantity = COALESCE($5, sp.package_quantity), "
        "package_unit = COALESCE($6, sp.package_unit), "
        "price_cents = COALESCE($7, sp.price_cents), "
        "notes = COALESCE($8, sp.notes), "
        "updated_at = now() "
        "FROM stores st "
        "WHERE sp.id = $1 AND sp.store_id = st.id AND st.owner_id = $2 "
        "RETURNING sp.id, sp.store_id, sp.ingredient_name, sp.product_name, "
        "sp.package_quantity, sp.package_unit, sp.price_cents, sp.notes",
        product_id,
        owner_id,
        ingredient_name,
        product_name,
        data.package_quantity,
        data.package_unit,
        data.price_cents,
        data.notes,
    )
    return StoreProduct(**dict(row)) if row else None


async def delete_store_product(
    pool: asyncpg.Pool, owner_id: uuid.UUID, product_id: uuid.UUID
) -> bool:
    result = await pool.execute(
        "DELETE FROM store_products sp USING stores st "
        "WHERE sp.id = $1 AND sp.store_id = st.id AND st.owner_id = $2",
        product_id,
        owner_id,
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Shopping list generation from meal plan(s)
# ------------------------------------------------------------------


async def _owned_plan_ids(
    pool: asyncpg.Pool, owner_id: uuid.UUID, requested_ids: list[uuid.UUID]
) -> list[uuid.UUID]:
    rows = await pool.fetch(
        "SELECT id FROM meal_plans WHERE id = ANY($1::uuid[]) AND owner_id = $2",
        requested_ids,
        owner_id,
    )
    return [row["id"] for row in rows]


async def _future_plan_ids(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[uuid.UUID]:
    today = date.today()
    this_monday = today - timedelta(days=today.weekday())
    rows = await pool.fetch(
        "SELECT id FROM meal_plans WHERE owner_id = $1 AND week_start >= $2",
        owner_id,
        this_monday,
    )
    return [row["id"] for row in rows]


async def _subtract_pantry(
    pool: asyncpg.Pool, owner_id: uuid.UUID, items: list[AggregatedIngredient]
) -> list[AggregatedIngredient]:
    """Subtract pantry on-hand from aggregated needed quantities.

    Matches by (lower(trim(name)), dimension) — the same key aggregate()
    groups by. Clamps each line at >= 0 and drops lines that hit exactly 0
    (fully covered by pantry stock). Lines with no quantity (can't be
    reduced) pass through unchanged.
    """
    pantry_rows = await pool.fetch(
        "SELECT name, quantity, unit FROM pantry_items WHERE owner_id = $1", owner_id
    )
    on_hand: dict[tuple[str, str], float] = {}
    for row in pantry_rows:
        canonical_qty, dimension, _ = canonicalize(row["quantity"], row["unit"])
        if canonical_qty is None:
            continue
        key = (row["name"].strip().lower(), dimension)
        on_hand[key] = on_hand.get(key, 0.0) + canonical_qty

    remaining: list[AggregatedIngredient] = []
    for item in items:
        if item.quantity is None:
            remaining.append(item)
            continue
        needed_canonical, dimension, _ = canonicalize(item.quantity, item.unit)
        key = (item.name.strip().lower(), dimension)
        have = on_hand.get(key, 0.0)
        remaining_canonical = max((needed_canonical or 0.0) - have, 0.0)
        if remaining_canonical <= 0:
            continue
        factor = to_canonical_unit_factor(item.unit)
        remaining_qty, remaining_unit = format_display(remaining_canonical / factor, item.unit)
        remaining.append(
            AggregatedIngredient(
                name=item.name, quantity=remaining_qty, unit=remaining_unit, category=item.category
            )
        )
    return remaining


async def _map_to_store_products(
    pool: asyncpg.Pool,
    owner_id: uuid.UUID,
    items: list[AggregatedIngredient],
    store_id: uuid.UUID | None,
) -> list[ShoppingListItemCreate]:
    """Map aggregated ingredient lines to a matching store product, when one exists.

    "Best match" when no `store_id` preference is given is the cheapest
    product across the owner's stores (nulls-last), an arbitrary but
    deterministic tie-break when price isn't recorded.
    """
    if store_id is not None:
        product_rows = await pool.fetch(
            "SELECT sp.id, sp.ingredient_name, sp.product_name, sp.package_quantity, "
            "sp.package_unit, st.name AS store_name "
            "FROM store_products sp JOIN stores st ON st.id = sp.store_id "
            "WHERE sp.store_id = $1 AND st.owner_id = $2 "
            "ORDER BY sp.price_cents NULLS LAST",
            store_id,
            owner_id,
        )
    else:
        product_rows = await pool.fetch(
            "SELECT sp.id, sp.ingredient_name, sp.product_name, sp.package_quantity, "
            "sp.package_unit, st.name AS store_name "
            "FROM store_products sp JOIN stores st ON st.id = sp.store_id "
            "WHERE st.owner_id = $1 "
            "ORDER BY sp.price_cents NULLS LAST",
            owner_id,
        )
    products_by_name: dict[str, asyncpg.Record] = {}
    for row in product_rows:
        products_by_name.setdefault(row["ingredient_name"].strip().lower(), row)

    results: list[ShoppingListItemCreate] = []
    for item in items:
        product = products_by_name.get(item.name.strip().lower())
        math_result = (
            packages_needed(
                item.quantity, item.unit, product["package_quantity"], product["package_unit"]
            )
            if product is not None and item.quantity is not None
            else None
        )
        if product is None or math_result is None:
            results.append(
                ShoppingListItemCreate(
                    name=item.name, quantity=item.quantity, unit=item.unit, category=item.category
                )
            )
            continue

        packages, _leftover_qty, _leftover_unit = math_result
        package_qty: float = product["package_quantity"]
        qty_str = str(int(package_qty)) if package_qty == int(package_qty) else str(package_qty)
        package_label = f"{qty_str} {product['package_unit']}"
        results.append(
            ShoppingListItemCreate(
                name=item.name,
                quantity=item.quantity,
                unit=item.unit,
                category=item.category,
                packages=packages,
                store_product_id=product["id"],
                store_name=product["store_name"],
                package_label=package_label,
            )
        )
    return results


async def generate_shopping_list(
    pool: asyncpg.Pool, owner_id: uuid.UUID, data: GenerateListRequest
) -> ShoppingList:
    """Generate an aggregated shopping list from one plan, a selection, or all future plans.

    Pipeline: gather ingredients across every day-recipe link in the
    selected plans (a recipe on N days contributes N copies) -> aggregate
    (D1) -> optionally subtract pantry on-hand, clamped >= 0, dropping
    covered lines -> map each line to a store product for package math,
    preferring `store_id` if given -> insert the shopping_lists row + items.
    """
    if data.plan_ids:
        plan_ids = await _owned_plan_ids(pool, owner_id, data.plan_ids)
        if not plan_ids:
            raise ValueError("No matching meal plans found")
    else:
        plan_ids = await _future_plan_ids(pool, owner_id)
        if not plan_ids:
            raise ValueError("No future meal plans found")

    # One row per day-recipe link joined to its ingredients — deliberately
    # not DISTINCT, so a recipe assigned to N days contributes N sets of
    # ingredients for aggregate() to sum.
    rows = await pool.fetch(
        "SELECT ri.name, ri.quantity, ri.unit, ri.category "
        "FROM meal_plan_day_recipes dr "
        "JOIN meal_plan_days d ON d.id = dr.day_id "
        "JOIN recipe_ingredients ri ON ri.recipe_id = dr.recipe_id "
        "WHERE d.plan_id = ANY($1::uuid[]) "
        "ORDER BY ri.category, ri.name",
        plan_ids,
    )
    if not rows:
        raise ValueError("No recipes assigned to the selected plans")

    aggregated = aggregate(
        (row["name"], row["quantity"], row["unit"], row["category"]) for row in rows
    )

    if data.deduct_pantry:
        # An empty result here is a legitimate outcome (pantry already covers
        # everything needed), not an error — the list is created with 0 items.
        aggregated = await _subtract_pantry(pool, owner_id, aggregated)

    mapped = await _map_to_store_products(pool, owner_id, aggregated, data.store_id)

    name = data.name
    if not name:
        if len(plan_ids) == 1:
            plan = await get_meal_plan(pool, owner_id, plan_ids[0])
            name = f"Shopping list for {plan.name}" if plan else "Shopping list"
        else:
            name = f"Shopping list for {len(plan_ids)} plans"

    slist = await create_shopping_list(
        pool,
        owner_id,
        ShoppingListCreate(
            name=name,
            plan_id=plan_ids[0] if len(plan_ids) == 1 else None,
        ),
    )

    items: list[ShoppingListItem] = []
    for entry in mapped:
        item = await add_list_item(pool, slist.id, entry)
        items.append(item)

    slist.items = items
    return slist


async def generate_shopping_list_from_plan(
    pool: asyncpg.Pool, owner_id: uuid.UUID, plan_id: uuid.UUID
) -> ShoppingList:
    """Backwards-compatible single-plan wrapper over :func:`generate_shopping_list`."""
    return await generate_shopping_list(pool, owner_id, GenerateListRequest(plan_ids=[plan_id]))


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(prefix="/api/meals", tags=["meals"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


# -- Meal Plans --


@router.get("/plans")
async def list_plans(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    plans = await list_meal_plans(_pool(request), user.id)
    return [p.model_dump(mode="json") for p in plans]


@router.post("/plans", status_code=201)
async def create_plan(
    request: Request,
    body: MealPlanCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        plan = await create_meal_plan(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return plan.model_dump(mode="json")


@router.get("/plans/{plan_id}")
async def get_plan(
    request: Request,
    plan_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    plan = await get_meal_plan(_pool(request), user.id, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return plan.model_dump(mode="json")


@router.put("/plans/{plan_id}")
async def update_plan(
    request: Request,
    plan_id: uuid.UUID,
    body: MealPlanUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        plan = await update_meal_plan(_pool(request), user.id, plan_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if plan is None:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return plan.model_dump(mode="json")


@router.delete("/plans/{plan_id}", status_code=204)
async def delete_plan(
    request: Request,
    plan_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_meal_plan(_pool(request), user.id, plan_id):
        raise HTTPException(status_code=404, detail="Meal plan not found")


# -- Meal Plan Days --


@router.get("/plans/{plan_id}/days")
async def list_days(
    request: Request,
    plan_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    plan = await get_meal_plan(_pool(request), user.id, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    days = await list_plan_days(_pool(request), plan_id)
    return [d.model_dump(mode="json") for d in days]


@router.post("/plans/{plan_id}/days", status_code=201)
async def create_day(
    request: Request,
    plan_id: uuid.UUID,
    body: MealPlanDayCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    plan = await get_meal_plan(_pool(request), user.id, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    day = await upsert_plan_day(_pool(request), plan_id, body)
    return day.model_dump(mode="json")


@router.put("/plans/days/{day_id}")
async def update_day(
    request: Request,
    day_id: uuid.UUID,
    body: MealPlanDayUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    day = await update_plan_day(_pool(request), user.id, day_id, body)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    return day.model_dump(mode="json")


class DayRecipeAction(BaseModel):
    recipe_id: uuid.UUID


@router.post("/plans/days/{day_id}/recipes", status_code=201)
async def add_recipe_to_day_endpoint(
    request: Request,
    day_id: uuid.UUID,
    body: DayRecipeAction,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    ref = await add_recipe_to_day(_pool(request), user.id, day_id, body.recipe_id)
    if ref is None:
        raise HTTPException(status_code=404, detail="Day not found")
    return ref.model_dump(mode="json")


@router.delete("/plans/days/{day_id}/recipes/{recipe_id}", status_code=204)
async def remove_recipe_from_day_endpoint(
    request: Request,
    day_id: uuid.UUID,
    recipe_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await remove_recipe_from_day(_pool(request), user.id, day_id, recipe_id):
        raise HTTPException(status_code=404, detail="Recipe not found on this day")


@router.delete("/plans/days/{day_id}", status_code=204)
async def delete_day(
    request: Request,
    day_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_plan_day(_pool(request), user.id, day_id):
        raise HTTPException(status_code=404, detail="Day not found")


# -- Generate shopping list from plan --


@router.post("/plans/{plan_id}/generate-list", status_code=201)
async def generate_list(
    request: Request,
    plan_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        slist = await generate_shopping_list_from_plan(_pool(request), user.id, plan_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return slist.model_dump(mode="json")


@router.post("/lists/generate", status_code=201)
async def generate_list_endpoint(
    request: Request,
    body: GenerateListRequest,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        slist = await generate_shopping_list(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return slist.model_dump(mode="json")


# -- Recipes --


@router.get("/recipes")
async def list_recipe_list(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    recipes = await list_recipes(_pool(request), user.id)
    return [r.model_dump(mode="json") for r in recipes]


@router.post("/recipes", status_code=201)
async def create_recipe_endpoint(
    request: Request,
    body: RecipeCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        recipe = await create_recipe(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return recipe.model_dump(mode="json")


@router.get("/recipes/{recipe_id}")
async def get_recipe_endpoint(
    request: Request,
    recipe_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    recipe = await get_recipe(_pool(request), user.id, recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe.model_dump(mode="json")


@router.put("/recipes/{recipe_id}")
async def update_recipe_endpoint(
    request: Request,
    recipe_id: uuid.UUID,
    body: RecipeUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        recipe = await update_recipe(_pool(request), user.id, recipe_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe.model_dump(mode="json")


@router.delete("/recipes/{recipe_id}", status_code=204)
async def delete_recipe_endpoint(
    request: Request,
    recipe_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_recipe(_pool(request), user.id, recipe_id):
        raise HTTPException(status_code=404, detail="Recipe not found")


# -- Recipe Ingredients --


@router.post("/recipes/{recipe_id}/ingredients", status_code=201)
async def add_recipe_ingredient(
    request: Request,
    recipe_id: uuid.UUID,
    body: RecipeIngredientCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    recipe = await get_recipe(_pool(request), user.id, recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    try:
        ingredient = await add_ingredient(_pool(request), recipe_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ingredient.model_dump(mode="json")


@router.put("/recipes/ingredients/{ingredient_id}")
async def update_recipe_ingredient(
    request: Request,
    ingredient_id: uuid.UUID,
    body: IngredientUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        ingredient = await update_ingredient(_pool(request), user.id, ingredient_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if ingredient is None:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    return ingredient.model_dump(mode="json")


@router.delete("/recipes/ingredients/{ingredient_id}", status_code=204)
async def delete_recipe_ingredient(
    request: Request,
    ingredient_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_ingredient(_pool(request), user.id, ingredient_id):
        raise HTTPException(status_code=404, detail="Ingredient not found")


# -- Shopping Lists --


@router.get("/lists")
async def list_shopping(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    lists = await list_shopping_lists(_pool(request), user.id)
    return [item.model_dump(mode="json") for item in lists]


@router.post("/lists", status_code=201)
async def create_shopping(
    request: Request,
    body: ShoppingListCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        slist = await create_shopping_list(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return slist.model_dump(mode="json")


@router.get("/lists/{list_id}")
async def get_shopping(
    request: Request,
    list_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    slist = await get_shopping_list(_pool(request), user.id, list_id)
    if slist is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    return slist.model_dump(mode="json")


@router.put("/lists/{list_id}")
async def update_shopping(
    request: Request,
    list_id: uuid.UUID,
    body: ShoppingListUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        slist = await update_shopping_list(_pool(request), user.id, list_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if slist is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    return slist.model_dump(mode="json")


@router.delete("/lists/{list_id}", status_code=204)
async def delete_shopping(
    request: Request,
    list_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_shopping_list(_pool(request), user.id, list_id):
        raise HTTPException(status_code=404, detail="Shopping list not found")


# -- Shopping List Items --


@router.post("/lists/{list_id}/items", status_code=201)
async def add_list_item_endpoint(
    request: Request,
    list_id: uuid.UUID,
    body: ShoppingListItemCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    slist = await get_shopping_list(_pool(request), user.id, list_id)
    if slist is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    try:
        item = await add_list_item(_pool(request), list_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return item.model_dump(mode="json")


@router.put("/lists/items/{item_id}")
async def update_list_item_endpoint(
    request: Request,
    item_id: uuid.UUID,
    body: ShoppingListItemUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        item = await update_list_item(_pool(request), user.id, item_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item.model_dump(mode="json")


@router.delete("/lists/items/{item_id}", status_code=204)
async def delete_list_item_endpoint(
    request: Request,
    item_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_list_item(_pool(request), user.id, item_id):
        raise HTTPException(status_code=404, detail="Item not found")


# -- Pantry --


@router.get("/pantry")
async def list_pantry_endpoint(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    items = await list_pantry_items(_pool(request), user.id)
    return [item.model_dump(mode="json") for item in items]


@router.post("/pantry", status_code=201)
async def upsert_pantry_endpoint(
    request: Request,
    body: PantryItemCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        item = await upsert_pantry_item(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return item.model_dump(mode="json")


@router.put("/pantry/{item_id}")
async def update_pantry_endpoint(
    request: Request,
    item_id: uuid.UUID,
    body: PantryItemUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        item = await update_pantry_item(_pool(request), user.id, item_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if item is None:
        raise HTTPException(status_code=404, detail="Pantry item not found")
    return item.model_dump(mode="json")


@router.delete("/pantry/{item_id}", status_code=204)
async def delete_pantry_endpoint(
    request: Request,
    item_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_pantry_item(_pool(request), user.id, item_id):
        raise HTTPException(status_code=404, detail="Pantry item not found")


# -- Consume / unconsume --


@router.post("/plans/days/{day_id}/consume")
async def consume_day_endpoint(
    request: Request,
    day_id: uuid.UUID,
    body: ConsumeRequest | None = None,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    force = body.force if body is not None else False
    try:
        result = await consume_day(_pool(request), user.id, day_id, force=force)
    except DayAlreadyConsumed as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Day not found")
    day, report = result
    return ConsumeResponse(day=day, report=report).model_dump(mode="json")


@router.post("/plans/days/{day_id}/unconsume")
async def unconsume_day_endpoint(
    request: Request,
    day_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    day = await unconsume_day(_pool(request), user.id, day_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    return day.model_dump(mode="json")


# -- Commit shopping list to pantry --


@router.post("/lists/{list_id}/commit-to-pantry")
async def commit_to_pantry_endpoint(
    request: Request,
    list_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    added = await commit_list_to_pantry(_pool(request), user.id, list_id)
    if added is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    return CommitToPantryResponse(added=added).model_dump(mode="json")


# -- Stores --


@router.get("/stores")
async def list_stores_endpoint(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    stores = await list_stores(_pool(request), user.id)
    return [store.model_dump(mode="json") for store in stores]


@router.post("/stores", status_code=201)
async def create_store_endpoint(
    request: Request,
    body: StoreCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        store = await create_store(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return store.model_dump(mode="json")


@router.put("/stores/{store_id}")
async def update_store_endpoint(
    request: Request,
    store_id: uuid.UUID,
    body: StoreUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        store = await update_store(_pool(request), user.id, store_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if store is None:
        raise HTTPException(status_code=404, detail="Store not found")
    return store.model_dump(mode="json")


@router.delete("/stores/{store_id}", status_code=204)
async def delete_store_endpoint(
    request: Request,
    store_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_store(_pool(request), user.id, store_id):
        raise HTTPException(status_code=404, detail="Store not found")


# -- Store Products --


@router.get("/stores/{store_id}/products")
async def list_store_products_endpoint(
    request: Request,
    store_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    products = await list_store_products(_pool(request), user.id, store_id)
    if products is None:
        raise HTTPException(status_code=404, detail="Store not found")
    return [product.model_dump(mode="json") for product in products]


@router.post("/stores/{store_id}/products", status_code=201)
async def add_store_product_endpoint(
    request: Request,
    store_id: uuid.UUID,
    body: StoreProductCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        product = await add_store_product(_pool(request), user.id, store_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if product is None:
        raise HTTPException(status_code=404, detail="Store not found")
    return product.model_dump(mode="json")


@router.put("/stores/products/{product_id}")
async def update_store_product_endpoint(
    request: Request,
    product_id: uuid.UUID,
    body: StoreProductUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        product = await update_store_product(_pool(request), user.id, product_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if product is None:
        raise HTTPException(status_code=404, detail="Store product not found")
    return product.model_dump(mode="json")


@router.delete("/stores/products/{product_id}", status_code=204)
async def delete_store_product_endpoint(
    request: Request,
    product_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_store_product(_pool(request), user.id, product_id):
        raise HTTPException(status_code=404, detail="Store product not found")


# ------------------------------------------------------------------
# Agent tool registration
# ------------------------------------------------------------------

# Several tool functions below intentionally share a name with the
# module-level data-access function they wrap (e.g. the "get_recipe" tool
# wraps get_recipe()). A nested `def get_recipe(...):` inside
# register_meal_tools would make "get_recipe" local to that *entire*
# function per Python's scoping rules — including before the def line — so
# referencing the module-level function from inside the tool would recurse
# into itself instead. These module-level aliases, captured once at import
# time before any such shadowing exists, are what the tool bodies actually
# call.
_recipes_fn = list_recipes
_get_recipe_fn = get_recipe
_create_recipe_fn = create_recipe
_update_recipe_fn = update_recipe
_list_meal_plans_fn = list_meal_plans
_get_meal_plan_fn = get_meal_plan
_create_meal_plan_fn = create_meal_plan
_remove_recipe_from_day_fn = remove_recipe_from_day
_generate_shopping_list_fn = generate_shopping_list
_list_shopping_lists_fn = list_shopping_lists
_get_shopping_list_fn = get_shopping_list
_list_stores_fn = list_stores
_list_store_products_fn = list_store_products


def register_meal_tools(agent: Agent[AgentDeps, str], database: Database | None) -> None:
    """Register user-scoped meal-planning tools on `agent`.

    Mirrors ``workflows.py:_register_search_tools``'s shape (a plain
    function taking the agent plus its dependencies, called from
    ``build_agent``), but every tool here is ``@agent.tool`` (not
    ``tool_plain``) and scopes its query by ``ctx.deps.user_id`` — identical
    bounds to the HTTP API, so RBAC is inherited: `AgentDeps` is only ever
    built after the `require_chat` role gate, and restricted users can't
    reach chat (and therefore these tools) at all. Never call this on the
    workflow agents in workflows.py — those run workspace-scoped, unattached
    to any one user, and meals is personal per-owner data.

    Tools are thin wrappers over the module's own data-access functions,
    referenced via the ``_*_fn`` module-level aliases above where a tool's
    name would otherwise shadow the function it calls. Every tool returns a
    JSON-serializable dict (or list of dicts) built from the same pydantic
    models the HTTP API returns (``model.model_dump(mode="json")``), so the
    model sees the identical shape a REST client would. Failures return
    ``{"error": "..."}`` (or ``[{"error": "..."}]`` for list-returning
    tools) rather than raising, so a bad call never crashes the turn.
    """

    def _unavailable(name: str) -> str:
        return f"{name} is unavailable: no database is configured."

    async def _resolve_plan_ids(owner_id: uuid.UUID, plans: list[str]) -> list[uuid.UUID] | str:
        """Resolve a mixed list of plan UUIDs and/or names (case-insensitive) to UUIDs.

        Returns an error string instead of a list if any entry can't be resolved.
        """
        all_plans = await _list_meal_plans_fn(database.pool, owner_id)  # type: ignore[union-attr]
        by_name = {p.name.strip().lower(): p.id for p in all_plans}
        resolved: list[uuid.UUID] = []
        for entry in plans:
            try:
                resolved.append(uuid.UUID(entry))
                continue
            except ValueError:
                pass
            match = by_name.get(entry.strip().lower())
            if match is None:
                return f"No meal plan found named or with id '{entry}'."
            resolved.append(match)
        return resolved

    # -- Recipes --

    @agent.tool
    async def search_recipes(ctx: RunContext[AgentDeps], query: str = "") -> list[dict]:
        """Search this user's recipes by name (case-insensitive substring match).

        Leave `query` empty to list every recipe. Returns a list of recipe
        summaries (id, name, description, prep/cook time, ingredient_count) —
        call get_recipe with an id from this list for the full ingredient
        list and instructions.
        """
        if database is None:
            return [{"error": _unavailable("search_recipes")}]
        try:
            recipes = await _recipes_fn(database.pool, ctx.deps.user_id)
        except Exception as error:
            logger.bind(query=query).warning(f"search_recipes failed: {error}")
            return [{"error": f"search_recipes failed: {error}"}]
        needle = query.strip().lower()
        if needle:
            recipes = [r for r in recipes if needle in r.name.lower()]
        return [r.model_dump(mode="json") for r in recipes]

    @agent.tool
    async def get_recipe(ctx: RunContext[AgentDeps], recipe_id: str) -> dict:
        """Fetch one recipe in full, including its ingredients and instructions.

        `recipe_id` is the UUID from search_recipes or create_recipe.
        """
        if database is None:
            return {"error": _unavailable("get_recipe")}
        try:
            recipe = await _get_recipe_fn(database.pool, ctx.deps.user_id, uuid.UUID(recipe_id))
        except ValueError as error:
            return {"error": f"get_recipe failed: {error}"}
        except Exception as error:
            logger.bind(recipe_id=recipe_id).warning(f"get_recipe failed: {error}")
            return {"error": f"get_recipe failed: {error}"}
        if recipe is None:
            return {"error": f"No recipe found with id {recipe_id}."}
        return recipe.model_dump(mode="json")

    @agent.tool
    async def create_recipe(
        ctx: RunContext[AgentDeps],
        name: str,
        ingredients: list[RecipeIngredientCreate] | None = None,
        description: str = "",
        instructions: str = "",
        servings: int = 4,
    ) -> dict:
        """Create a new recipe, optionally with its ingredient list.

        Each ingredient needs a `name`; `quantity` (a number) and `unit`
        (free text like "cup", "lb", "oz") are optional — omit both for a
        to-taste ingredient like salt.
        """
        if database is None:
            return {"error": _unavailable("create_recipe")}
        try:
            recipe = await _create_recipe_fn(
                database.pool,
                ctx.deps.user_id,
                RecipeCreate(
                    name=name,
                    description=description,
                    instructions=instructions,
                    servings=servings,
                    ingredients=ingredients or [],
                ),
            )
        except ValueError as error:
            return {"error": f"create_recipe failed: {error}"}
        except Exception as error:
            logger.bind(name=name).warning(f"create_recipe failed: {error}")
            return {"error": f"create_recipe failed: {error}"}
        return recipe.model_dump(mode="json")

    @agent.tool
    async def update_recipe(
        ctx: RunContext[AgentDeps],
        recipe_id: str,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        servings: int | None = None,
    ) -> dict:
        """Update a recipe's name, description, instructions, and/or servings.

        Only fields you provide are changed; omitted fields keep their
        current value. Does not touch ingredients — there's no separate
        ingredient-editing tool yet, so mention that limitation if asked.
        """
        if database is None:
            return {"error": _unavailable("update_recipe")}
        try:
            recipe = await _update_recipe_fn(
                database.pool,
                ctx.deps.user_id,
                uuid.UUID(recipe_id),
                RecipeUpdate(
                    name=name,
                    description=description,
                    instructions=instructions,
                    servings=servings,
                ),
            )
        except ValueError as error:
            return {"error": f"update_recipe failed: {error}"}
        except Exception as error:
            logger.bind(recipe_id=recipe_id).warning(f"update_recipe failed: {error}")
            return {"error": f"update_recipe failed: {error}"}
        if recipe is None:
            return {"error": f"No recipe found with id {recipe_id}."}
        return recipe.model_dump(mode="json")

    # -- Meal plans --

    @agent.tool
    async def list_meal_plans(ctx: RunContext[AgentDeps]) -> list[dict]:
        """List this user's meal plans (id, name, week_start, day/recipe counts)."""
        if database is None:
            return [{"error": _unavailable("list_meal_plans")}]
        try:
            plans = await _list_meal_plans_fn(database.pool, ctx.deps.user_id)
        except Exception as error:
            logger.warning(f"list_meal_plans failed: {error}")
            return [{"error": f"list_meal_plans failed: {error}"}]
        return [p.model_dump(mode="json") for p in plans]

    @agent.tool
    async def get_meal_plan(ctx: RunContext[AgentDeps], plan_id: str) -> dict:
        """Fetch one meal plan with all its days, assigned recipes, and consumed status.

        `plan_id` is the UUID from list_meal_plans or create_meal_plan. Each
        day includes `consumed_at` (null if not yet marked consumed).
        """
        if database is None:
            return {"error": _unavailable("get_meal_plan")}
        try:
            plan_uuid = uuid.UUID(plan_id)
            plan = await _get_meal_plan_fn(database.pool, ctx.deps.user_id, plan_uuid)
        except ValueError as error:
            return {"error": f"get_meal_plan failed: {error}"}
        except Exception as error:
            logger.bind(plan_id=plan_id).warning(f"get_meal_plan failed: {error}")
            return {"error": f"get_meal_plan failed: {error}"}
        if plan is None:
            return {"error": f"No meal plan found with id {plan_id}."}
        days = await list_plan_days(database.pool, plan_uuid)
        return {**plan.model_dump(mode="json"), "days": [d.model_dump(mode="json") for d in days]}

    @agent.tool
    async def create_meal_plan(ctx: RunContext[AgentDeps], name: str, week_start: str) -> dict:
        """Create a new meal plan (a calendar week).

        `week_start` must be a Monday, formatted YYYY-MM-DD — if the date
        you're given isn't a Monday, find that week's Monday yourself before
        calling this (e.g. "next week" -> the Monday starting next week).
        """
        if database is None:
            return {"error": _unavailable("create_meal_plan")}
        try:
            data = MealPlanCreate(name=name, week_start=date.fromisoformat(week_start))
            plan = await _create_meal_plan_fn(database.pool, ctx.deps.user_id, data)
        except ValueError as error:
            return {"error": f"create_meal_plan failed: {error}"}
        except Exception as error:
            logger.bind(name=name).warning(f"create_meal_plan failed: {error}")
            return {"error": f"create_meal_plan failed: {error}"}
        return plan.model_dump(mode="json")

    @agent.tool
    async def assign_recipe_to_day(
        ctx: RunContext[AgentDeps],
        plan_id: str,
        day_date: str,
        recipe_id: str,
        meal_type: str = "dinner",
    ) -> dict:
        """Assign a recipe to a day on a meal plan.

        `day_date` is YYYY-MM-DD and must fall within the plan's week.
        `meal_type` is one of breakfast/lunch/dinner/snack. Creates the day
        slot automatically if one doesn't already exist for that date and
        meal type; adding a second recipe to an existing slot just adds it
        alongside the first. Returns the day's current state.
        """
        if database is None:
            return {"error": _unavailable("assign_recipe_to_day")}
        try:
            plan_uuid = uuid.UUID(plan_id)
            recipe_uuid = uuid.UUID(recipe_id)
            parsed_date = date.fromisoformat(day_date)
            meal_type_enum = MealType(meal_type)
        except ValueError as error:
            return {"error": f"assign_recipe_to_day failed: {error}"}

        plan = await _get_meal_plan_fn(database.pool, ctx.deps.user_id, plan_uuid)
        if plan is None:
            return {"error": f"No meal plan found with id {plan_id}."}

        try:
            days = await list_plan_days(database.pool, plan_uuid)
            existing = next(
                (d for d in days if d.day_date == parsed_date and d.meal_type == meal_type),
                None,
            )
            if existing is None:
                day = await upsert_plan_day(
                    database.pool,
                    plan_uuid,
                    MealPlanDayCreate(
                        day_date=parsed_date, meal_type=meal_type_enum, recipe_ids=[recipe_uuid]
                    ),
                )
            else:
                ref = await add_recipe_to_day(
                    database.pool, ctx.deps.user_id, existing.id, recipe_uuid
                )
                if ref is None:
                    return {"error": "assign_recipe_to_day failed: day not found."}
                day = await get_plan_day(database.pool, ctx.deps.user_id, existing.id)
        except Exception as error:
            logger.bind(plan_id=plan_id).warning(f"assign_recipe_to_day failed: {error}")
            return {"error": f"assign_recipe_to_day failed: {error}"}
        assert day is not None
        return day.model_dump(mode="json")

    @agent.tool
    async def remove_recipe_from_day(
        ctx: RunContext[AgentDeps], day_id: str, recipe_id: str
    ) -> dict:
        """Remove a recipe from a day on a meal plan. Leaves the day slot itself in place."""
        if database is None:
            return {"error": _unavailable("remove_recipe_from_day")}
        try:
            removed = await _remove_recipe_from_day_fn(
                database.pool, ctx.deps.user_id, uuid.UUID(day_id), uuid.UUID(recipe_id)
            )
        except ValueError as error:
            return {"error": f"remove_recipe_from_day failed: {error}"}
        except Exception as error:
            logger.bind(day_id=day_id).warning(f"remove_recipe_from_day failed: {error}")
            return {"error": f"remove_recipe_from_day failed: {error}"}
        if not removed:
            return {"error": "That recipe wasn't found on that day.", "removed": False}
        return {"removed": True}

    @agent.tool
    async def mark_meal_consumed(
        ctx: RunContext[AgentDeps], day_id: str, force: bool = False
    ) -> dict:
        """Mark a day's meal as consumed, deducting its ingredients from pantry.

        `day_id` is a day's UUID (from get_meal_plan or assign_recipe_to_day).
        Fails with an error if that day is already marked consumed — pass
        `force=true` to re-consume it anyway (deducts pantry again). Returns
        the day plus a per-ingredient report: each line's `status` is
        "deducted" (enough stock), "insufficient" (ran out, clamped at 0),
        or "not_tracked" (that ingredient isn't in pantry at all).
        """
        if database is None:
            return {"error": _unavailable("mark_meal_consumed")}
        try:
            day_uuid = uuid.UUID(day_id)
        except ValueError as error:
            return {"error": f"mark_meal_consumed failed: {error}"}
        try:
            result = await consume_day(database.pool, ctx.deps.user_id, day_uuid, force=force)
        except DayAlreadyConsumed:
            return {
                "error": "That day is already marked consumed. Pass force=true to re-consume it."
            }
        except Exception as error:
            logger.bind(day_id=day_id).warning(f"mark_meal_consumed failed: {error}")
            return {"error": f"mark_meal_consumed failed: {error}"}
        if result is None:
            return {"error": f"No day found with id {day_id}."}
        day, report = result
        return ConsumeResponse(day=day, report=report).model_dump(mode="json")

    @agent.tool
    async def unmark_meal_consumed(ctx: RunContext[AgentDeps], day_id: str) -> dict:
        """Clear a day's consumed status and best-effort restore its ingredients to pantry.

        A no-op restoration (but still succeeds) if the day was never marked
        consumed in the first place — nothing to credit back.
        """
        if database is None:
            return {"error": _unavailable("unmark_meal_consumed")}
        try:
            day = await unconsume_day(database.pool, ctx.deps.user_id, uuid.UUID(day_id))
        except ValueError as error:
            return {"error": f"unmark_meal_consumed failed: {error}"}
        except Exception as error:
            logger.bind(day_id=day_id).warning(f"unmark_meal_consumed failed: {error}")
            return {"error": f"unmark_meal_consumed failed: {error}"}
        if day is None:
            return {"error": f"No day found with id {day_id}."}
        return day.model_dump(mode="json")

    # -- Shopping lists --

    @agent.tool
    async def generate_shopping_list(
        ctx: RunContext[AgentDeps],
        plans: list[str] | None = None,
        scope: str | None = None,
        deduct_pantry: bool = False,
        store_id: str | None = None,
        name: str | None = None,
    ) -> dict:
        """Generate an aggregated shopping list from one or more meal plans.

        Provide exactly one of `plans` (a list of plan names and/or plan
        UUIDs — names are matched case-insensitively against this user's
        plans) or `scope="future"` (every one of this user's plans from this
        week onward). A recipe used on multiple days contributes multiple
        times, then identical ingredients are summed. Set `deduct_pantry` to
        subtract on-hand pantry stock first (items pantry fully covers are
        dropped from the list). `store_id` prefers package sizing from one
        store; otherwise the cheapest matching product across all stores is
        used. Returns the created shopping list with its items.
        """
        if database is None:
            return {"error": _unavailable("generate_shopping_list")}
        try:
            plan_ids: list[uuid.UUID] | None = None
            if plans:
                resolved = await _resolve_plan_ids(ctx.deps.user_id, plans)
                if isinstance(resolved, str):
                    return {"error": resolved}
                plan_ids = resolved
            request = GenerateListRequest(
                plan_ids=plan_ids,
                scope=scope,  # type: ignore[arg-type]
                deduct_pantry=deduct_pantry,
                store_id=uuid.UUID(store_id) if store_id else None,
                name=name,
            )
            slist = await _generate_shopping_list_fn(database.pool, ctx.deps.user_id, request)
        except ValueError as error:
            return {"error": f"generate_shopping_list failed: {error}"}
        except Exception as error:
            logger.warning(f"generate_shopping_list failed: {error}")
            return {"error": f"generate_shopping_list failed: {error}"}
        return slist.model_dump(mode="json")

    @agent.tool
    async def list_shopping_lists(ctx: RunContext[AgentDeps]) -> list[dict]:
        """List this user's shopping lists (id, name, purchased/total item counts)."""
        if database is None:
            return [{"error": _unavailable("list_shopping_lists")}]
        try:
            lists = await _list_shopping_lists_fn(database.pool, ctx.deps.user_id)
        except Exception as error:
            logger.warning(f"list_shopping_lists failed: {error}")
            return [{"error": f"list_shopping_lists failed: {error}"}]
        return [ls.model_dump(mode="json") for ls in lists]

    @agent.tool
    async def get_shopping_list(ctx: RunContext[AgentDeps], list_id: str) -> dict:
        """Fetch one shopping list with all items, including purchased state and package info."""
        if database is None:
            return {"error": _unavailable("get_shopping_list")}
        try:
            slist = await _get_shopping_list_fn(database.pool, ctx.deps.user_id, uuid.UUID(list_id))
        except ValueError as error:
            return {"error": f"get_shopping_list failed: {error}"}
        except Exception as error:
            logger.bind(list_id=list_id).warning(f"get_shopping_list failed: {error}")
            return {"error": f"get_shopping_list failed: {error}"}
        if slist is None:
            return {"error": f"No shopping list found with id {list_id}."}
        return slist.model_dump(mode="json")

    @agent.tool
    async def add_shopping_item(
        ctx: RunContext[AgentDeps],
        list_id: str,
        name: str,
        quantity: float | None = None,
        unit: str | None = None,
        category: str = "",
    ) -> dict:
        """Add an item to a shopping list."""
        if database is None:
            return {"error": _unavailable("add_shopping_item")}
        try:
            list_uuid = uuid.UUID(list_id)
        except ValueError as error:
            return {"error": f"add_shopping_item failed: {error}"}
        slist = await _get_shopping_list_fn(database.pool, ctx.deps.user_id, list_uuid)
        if slist is None:
            return {"error": f"No shopping list found with id {list_id}."}
        try:
            item = await add_list_item(
                database.pool,
                list_uuid,
                ShoppingListItemCreate(name=name, quantity=quantity, unit=unit, category=category),
            )
        except ValueError as error:
            return {"error": f"add_shopping_item failed: {error}"}
        except Exception as error:
            logger.bind(list_id=list_id).warning(f"add_shopping_item failed: {error}")
            return {"error": f"add_shopping_item failed: {error}"}
        return item.model_dump(mode="json")

    @agent.tool
    async def check_off_item(
        ctx: RunContext[AgentDeps], item_id: str, purchased: bool = True
    ) -> dict:
        """Check (or, with purchased=false, uncheck) a shopping list item as purchased."""
        if database is None:
            return {"error": _unavailable("check_off_item")}
        try:
            item = await update_list_item(
                database.pool,
                ctx.deps.user_id,
                uuid.UUID(item_id),
                ShoppingListItemUpdate(purchased=purchased),
            )
        except ValueError as error:
            return {"error": f"check_off_item failed: {error}"}
        except Exception as error:
            logger.bind(item_id=item_id).warning(f"check_off_item failed: {error}")
            return {"error": f"check_off_item failed: {error}"}
        if item is None:
            return {"error": f"No shopping list item found with id {item_id}."}
        return item.model_dump(mode="json")

    # -- Pantry --

    @agent.tool
    async def get_pantry(ctx: RunContext[AgentDeps]) -> list[dict]:
        """List everything currently tracked in this user's pantry."""
        if database is None:
            return [{"error": _unavailable("get_pantry")}]
        try:
            items = await list_pantry_items(database.pool, ctx.deps.user_id)
        except Exception as error:
            logger.warning(f"get_pantry failed: {error}")
            return [{"error": f"get_pantry failed: {error}"}]
        return [i.model_dump(mode="json") for i in items]

    @agent.tool
    async def set_pantry_item(
        ctx: RunContext[AgentDeps],
        name: str,
        quantity: float | None = None,
        unit: str | None = None,
        category: str = "",
    ) -> dict:
        """Add to (or create) a pantry item.

        Upserts: matches an existing item by name and unit dimension and
        adds the quantities together, rather than creating a duplicate entry.
        """
        if database is None:
            return {"error": _unavailable("set_pantry_item")}
        try:
            item = await upsert_pantry_item(
                database.pool,
                ctx.deps.user_id,
                PantryItemCreate(name=name, quantity=quantity, unit=unit, category=category),
            )
        except ValueError as error:
            return {"error": f"set_pantry_item failed: {error}"}
        except Exception as error:
            logger.bind(name=name).warning(f"set_pantry_item failed: {error}")
            return {"error": f"set_pantry_item failed: {error}"}
        return item.model_dump(mode="json")

    # -- Stores --

    @agent.tool
    async def list_stores(ctx: RunContext[AgentDeps]) -> list[dict]:
        """List this user's stores."""
        if database is None:
            return [{"error": _unavailable("list_stores")}]
        try:
            stores = await _list_stores_fn(database.pool, ctx.deps.user_id)
        except Exception as error:
            logger.warning(f"list_stores failed: {error}")
            return [{"error": f"list_stores failed: {error}"}]
        return [s.model_dump(mode="json") for s in stores]

    @agent.tool
    async def list_store_products(ctx: RunContext[AgentDeps], store_id: str) -> list[dict]:
        """List a store's products: ingredient -> product name mappings with package sizes."""
        if database is None:
            return [{"error": _unavailable("list_store_products")}]
        try:
            products = await _list_store_products_fn(
                database.pool, ctx.deps.user_id, uuid.UUID(store_id)
            )
        except ValueError as error:
            return [{"error": f"list_store_products failed: {error}"}]
        except Exception as error:
            logger.bind(store_id=store_id).warning(f"list_store_products failed: {error}")
            return [{"error": f"list_store_products failed: {error}"}]
        if products is None:
            return [{"error": f"No store found with id {store_id}."}]
        return [p.model_dump(mode="json") for p in products]

    @agent.tool
    async def upsert_store_product(
        ctx: RunContext[AgentDeps],
        store_id: str,
        ingredient_name: str,
        product_name: str,
        package_quantity: float,
        package_unit: str,
        price_cents: int | None = None,
    ) -> dict:
        """Add or update a store's product mapping for an ingredient.

        Matches an existing product at that store by `ingredient_name`
        (case-insensitive); updates it if found, otherwise creates a new
        one. `package_quantity`/`package_unit` describe one package as sold
        (e.g. 5 and "lb" for a 5 lb bag).
        """
        if database is None:
            return {"error": _unavailable("upsert_store_product")}
        try:
            store_uuid = uuid.UUID(store_id)
        except ValueError as error:
            return {"error": f"upsert_store_product failed: {error}"}
        try:
            existing = await _list_store_products_fn(database.pool, ctx.deps.user_id, store_uuid)
        except Exception as error:
            logger.bind(store_id=store_id).warning(f"upsert_store_product failed: {error}")
            return {"error": f"upsert_store_product failed: {error}"}
        if existing is None:
            return {"error": f"No store found with id {store_id}."}

        match = next(
            (
                p
                for p in existing
                if p.ingredient_name.strip().lower() == ingredient_name.strip().lower()
            ),
            None,
        )
        try:
            if match is not None:
                product = await update_store_product(
                    database.pool,
                    ctx.deps.user_id,
                    match.id,
                    StoreProductUpdate(
                        product_name=product_name,
                        package_quantity=package_quantity,
                        package_unit=package_unit,
                        price_cents=price_cents,
                    ),
                )
            else:
                product = await add_store_product(
                    database.pool,
                    ctx.deps.user_id,
                    store_uuid,
                    StoreProductCreate(
                        ingredient_name=ingredient_name,
                        product_name=product_name,
                        package_quantity=package_quantity,
                        package_unit=package_unit,
                        price_cents=price_cents,
                    ),
                )
        except ValueError as error:
            return {"error": f"upsert_store_product failed: {error}"}
        except Exception as error:
            logger.bind(store_id=store_id).warning(f"upsert_store_product failed: {error}")
            return {"error": f"upsert_store_product failed: {error}"}
        if product is None:
            return {"error": f"No store found with id {store_id}."}
        return {**product.model_dump(mode="json"), "created": match is None}
