"""Meal planning: weekly plans, recipes, and shopping lists.

This module owns the meal planning end to end: pydantic models, asyncpg data
access helpers, and the FastAPI router mounted under ``/api/meals``.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from cherryai_api.auth import current_verified_user
from cherryai_api.meal_units import aggregate
from cherryai_api.users import User

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
"""


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


class MealPlanCreate(BaseModel):
    name: str
    week_start: date


class MealPlanUpdate(BaseModel):
    name: str | None = None
    week_start: date | None = None


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

_DAY_COLUMNS = "id, plan_id, day_date, meal_type, notes, sort_order"


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
    pool: asyncpg.Pool, day_id: uuid.UUID, data: MealPlanDayUpdate
) -> MealPlanDay | None:
    row = await pool.fetchrow(
        f"UPDATE meal_plan_days SET "
        f"notes = COALESCE($2, notes) "
        f"WHERE id = $1 "
        f"RETURNING {_DAY_COLUMNS}",
        day_id,
        data.notes,
    )
    if row is None:
        return None
    day = dict(row)
    day["recipes"] = await _load_day_recipes(pool, day_id)
    return MealPlanDay(**day)


async def delete_plan_day(pool: asyncpg.Pool, day_id: uuid.UUID) -> bool:
    result = await pool.execute("DELETE FROM meal_plan_days WHERE id = $1", day_id)
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Day Recipes (many-to-many)
# ------------------------------------------------------------------


async def add_recipe_to_day(
    pool: asyncpg.Pool, day_id: uuid.UUID, recipe_id: uuid.UUID
) -> RecipeRef:
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
    pool: asyncpg.Pool, day_id: uuid.UUID, recipe_id: uuid.UUID
) -> bool:
    result = await pool.execute(
        "DELETE FROM meal_plan_day_recipes WHERE day_id = $1 AND recipe_id = $2",
        day_id,
        recipe_id,
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
    pool: asyncpg.Pool, ingredient_id: uuid.UUID, data: IngredientUpdate
) -> RecipeIngredient | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Ingredient name must not be empty")
    row = await pool.fetchrow(
        "UPDATE recipe_ingredients SET "
        "name = COALESCE($2, name), "
        "quantity = COALESCE($3, quantity), "
        "unit = COALESCE($4, unit), "
        "notes = COALESCE($5, notes), "
        "category = COALESCE($6, category) "
        "WHERE id = $1 "
        "RETURNING id, recipe_id, name, quantity, unit, notes, category, sort_order",
        ingredient_id,
        name,
        data.quantity,
        data.unit,
        data.notes,
        data.category,
    )
    return RecipeIngredient(**dict(row)) if row else None


async def delete_ingredient(pool: asyncpg.Pool, ingredient_id: uuid.UUID) -> bool:
    result = await pool.execute("DELETE FROM recipe_ingredients WHERE id = $1", ingredient_id)
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
        "SELECT id, list_id, name, quantity, unit, category, purchased, sort_order "
        "FROM shopping_list_items WHERE list_id = $1 ORDER BY sort_order",
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
        "SELECT id, list_id, name, quantity, unit, category, purchased, sort_order "
        "FROM shopping_list_items WHERE list_id = $1 ORDER BY sort_order",
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
        "(id, list_id, name, quantity, unit, category, sort_order) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) "
        "RETURNING id, list_id, name, quantity, unit, category, purchased, sort_order",
        uuid.uuid4(),
        list_id,
        name,
        data.quantity,
        data.unit,
        data.category,
        sort_order,
    )
    return ShoppingListItem(**dict(row))


async def update_list_item(
    pool: asyncpg.Pool, item_id: uuid.UUID, data: ShoppingListItemUpdate
) -> ShoppingListItem | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Item name must not be empty")
    row = await pool.fetchrow(
        "UPDATE shopping_list_items SET "
        "name = COALESCE($2, name), "
        "quantity = COALESCE($3, quantity), "
        "unit = COALESCE($4, unit), "
        "category = COALESCE($5, category), "
        "purchased = COALESCE($6, purchased) "
        "WHERE id = $1 "
        "RETURNING id, list_id, name, quantity, unit, category, purchased, sort_order",
        item_id,
        name,
        data.quantity,
        data.unit,
        data.category,
        data.purchased,
    )
    return ShoppingListItem(**dict(row)) if row else None


async def delete_list_item(pool: asyncpg.Pool, item_id: uuid.UUID) -> bool:
    result = await pool.execute("DELETE FROM shopping_list_items WHERE id = $1", item_id)
    return result.endswith("1")


# ------------------------------------------------------------------
# Shopping list generation from meal plan
# ------------------------------------------------------------------


async def generate_shopping_list_from_plan(
    pool: asyncpg.Pool, owner_id: uuid.UUID, plan_id: uuid.UUID
) -> ShoppingList:
    """Aggregate all recipe ingredients from a meal plan into a shopping list.

    Ingredients are collected per day-recipe *link*, not per distinct recipe:
    a recipe assigned to N days on the plan contributes N copies of its
    ingredients, which :func:`~cherryai_api.meal_units.aggregate` then sums
    (in a shared canonical unit) into one line per (name, unit dimension).
    """
    plan = await get_meal_plan(pool, owner_id, plan_id)
    if plan is None:
        raise ValueError("Meal plan not found")

    # One row per day-recipe link joined to its ingredients — deliberately
    # not DISTINCT, so a recipe assigned to N days contributes N sets of
    # ingredients for aggregate() to sum.
    rows = await pool.fetch(
        "SELECT ri.name, ri.quantity, ri.unit, ri.category "
        "FROM meal_plan_day_recipes dr "
        "JOIN meal_plan_days d ON d.id = dr.day_id "
        "JOIN recipe_ingredients ri ON ri.recipe_id = dr.recipe_id "
        "WHERE d.plan_id = $1 "
        "ORDER BY ri.category, ri.name",
        plan_id,
    )

    if not rows:
        raise ValueError("Meal plan has no recipes assigned")

    aggregated = aggregate(
        (row["name"], row["quantity"], row["unit"], row["category"]) for row in rows
    )

    # Create the shopping list
    slist = await create_shopping_list(
        pool,
        owner_id,
        ShoppingListCreate(
            name=f"Shopping list for {plan.name}",
            plan_id=plan_id,
        ),
    )

    # Add aggregated items
    items: list[ShoppingListItem] = []
    for entry in aggregated:
        item = await add_list_item(
            pool,
            slist.id,
            ShoppingListItemCreate(
                name=entry.name,
                quantity=entry.quantity,
                unit=entry.unit,
                category=entry.category,
            ),
        )
        items.append(item)

    slist.items = items
    return slist


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
    day = await update_plan_day(_pool(request), day_id, body)
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
    ref = await add_recipe_to_day(_pool(request), day_id, body.recipe_id)
    return ref.model_dump(mode="json")


@router.delete("/plans/days/{day_id}/recipes/{recipe_id}", status_code=204)
async def remove_recipe_from_day_endpoint(
    request: Request,
    day_id: uuid.UUID,
    recipe_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await remove_recipe_from_day(_pool(request), day_id, recipe_id):
        raise HTTPException(status_code=404, detail="Recipe not found on this day")


@router.delete("/plans/days/{day_id}", status_code=204)
async def delete_day(
    request: Request,
    day_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_plan_day(_pool(request), day_id):
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
        ingredient = await update_ingredient(_pool(request), ingredient_id, body)
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
    if not await delete_ingredient(_pool(request), ingredient_id):
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
        item = await update_list_item(_pool(request), item_id, body)
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
    if not await delete_list_item(_pool(request), item_id):
        raise HTTPException(status_code=404, detail="Item not found")
