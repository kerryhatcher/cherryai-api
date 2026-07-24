"""Tests for meal planning: unit aggregation, IDOR guards, and Monday validation.

Pure-function tests for ``meal_units`` need no database. DB-backed tests use
the ``pool`` fixture (dev Postgres) and the ``make_user`` fixture for a fresh
owner per test, matching the pattern in ``test_wiki.py``.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from cherryai_api.meal_units import aggregate, canonicalize, format_display
from cherryai_api.meals import (
    IngredientUpdate,
    MealPlanCreate,
    MealPlanDayCreate,
    MealPlanDayUpdate,
    RecipeCreate,
    RecipeIngredientCreate,
    ShoppingListCreate,
    ShoppingListItemCreate,
    ShoppingListItemUpdate,
    add_ingredient,
    add_list_item,
    add_recipe_to_day,
    create_meal_plan,
    create_recipe,
    create_shopping_list,
    delete_ingredient,
    delete_list_item,
    delete_plan_day,
    generate_shopping_list_from_plan,
    remove_recipe_from_day,
    update_ingredient,
    update_list_item,
    update_plan_day,
    upsert_plan_day,
)

# --- canonicalize --------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected_qty", "expected_dimension", "expected_unit"),
    [
        ("lb", 453.592, "mass", "g"),
        ("lbs", 453.592, "mass", "g"),
        ("pound", 453.592, "mass", "g"),
        ("oz", 28.3495, "mass", "g"),
        ("kg", 1000.0, "mass", "g"),
        ("g", 1.0, "mass", "g"),
        ("cup", 236.588, "volume", "ml"),
        ("cups", 236.588, "volume", "ml"),
        ("tbsp", 14.787, "volume", "ml"),
        ("tsp", 4.929, "volume", "ml"),
        ("fl oz", 29.574, "volume", "ml"),
        ("ml", 1.0, "volume", "ml"),
        ("l", 1000.0, "volume", "ml"),
        ("qt", 946.353, "volume", "ml"),
        ("pt", 473.176, "volume", "ml"),
        ("gal", 3785.41, "volume", "ml"),
        ("each", 1.0, "count", "count"),
        ("dozen", 12.0, "count", "count"),
    ],
)
def test_canonicalize_known_units(unit, expected_qty, expected_dimension, expected_unit) -> None:
    qty, dimension, canonical_unit = canonicalize(1, unit)
    assert qty == pytest.approx(expected_qty)
    assert dimension == expected_dimension
    assert canonical_unit == expected_unit


def test_canonicalize_is_case_and_whitespace_insensitive() -> None:
    assert canonicalize(1, "LB") == canonicalize(1, " lb ")
    assert canonicalize(1, "Fl  Oz")[1] == "volume"


def test_canonicalize_none_unit_is_count() -> None:
    assert canonicalize(3, None) == (3.0, "count", "count")


def test_canonicalize_blank_unit_is_count() -> None:
    assert canonicalize(3, "   ") == (3.0, "count", "count")


def test_canonicalize_unknown_unit_gets_its_own_dimension() -> None:
    qty, dimension, unit = canonicalize(2, "pinch")
    assert qty == 2.0
    assert dimension == "unknown:pinch"
    assert unit == "pinch"


def test_canonicalize_none_quantity_passes_through() -> None:
    qty, dimension, unit = canonicalize(None, "cup")
    assert qty is None
    assert dimension == "volume"
    assert unit == "ml"


# --- format_display --------------------------------------------------------------


def test_format_display_rounds_to_two_decimals() -> None:
    assert format_display(1.23456, "cup") == (1.23, "cup")


def test_format_display_collapses_integral_values() -> None:
    qty, unit = format_display(2.0, "cup")
    assert qty == 2
    assert unit == "cup"


def test_format_display_passes_through_none() -> None:
    assert format_display(None, "cup") == (None, "cup")


# --- aggregate ---------------------------------------------------------------


def test_aggregate_same_unit_sums() -> None:
    result = aggregate(
        [
            ("Flour", 1.0, "cup", "baking"),
            ("flour", 1.0, "cup", "baking"),
        ]
    )
    assert len(result) == 1
    assert result[0].name == "Flour"
    assert result[0].quantity == 2.0
    assert result[0].unit == "cup"
    assert result[0].category == "baking"


def test_aggregate_cross_unit_same_dimension_sums_in_first_seen_unit() -> None:
    result = aggregate(
        [
            ("Chicken", 1.0, "lb", "meat"),
            ("chicken", 8.0, "oz", "meat"),
        ]
    )
    assert len(result) == 1
    assert result[0].quantity == 1.5
    assert result[0].unit == "lb"


def test_aggregate_cups_and_ml_merge_by_dimension() -> None:
    result = aggregate(
        [
            ("Milk", 1.0, "cup", "dairy"),
            ("milk", 236.588, "ml", "dairy"),
        ]
    )
    assert len(result) == 1
    assert result[0].quantity == 2.0
    assert result[0].unit == "cup"


def test_aggregate_dimension_mismatch_not_merged() -> None:
    result = aggregate(
        [
            ("Flour", 2.0, "cup", "baking"),
            ("Flour", 300.0, "g", "baking"),
        ]
    )
    assert len(result) == 2
    units = {r.unit for r in result}
    assert units == {"cup", "g"}


def test_aggregate_unknown_units_merge_only_on_exact_match() -> None:
    result = aggregate(
        [
            ("Salt", 1.0, "pinch", "spices"),
            ("salt", 2.0, "pinch", "spices"),
            ("salt", 1.0, "dash", "spices"),
        ]
    )
    by_unit = {r.unit: r.quantity for r in result}
    assert by_unit == {"pinch": 3.0, "dash": 1.0}


def test_aggregate_none_quantity_grouped_separately_from_quantified() -> None:
    result = aggregate(
        [
            ("Salt", None, None, "spices"),
            ("Salt", 1.0, "tsp", "spices"),
        ]
    )
    assert len(result) == 2
    assert any(r.quantity is None for r in result)
    assert any(r.quantity == 1.0 for r in result)


def test_aggregate_none_quantity_multiplicity_noted() -> None:
    result = aggregate(
        [
            ("Salt", None, None, "spices"),
            ("salt", None, None, "spices"),
            ("salt", None, None, "spices"),
        ]
    )
    assert len(result) == 1
    assert result[0].quantity is None
    assert result[0].name == "Salt — ×3"


def test_aggregate_none_quantity_single_occurrence_has_no_multiplicity_note() -> None:
    result = aggregate([("Salt", None, None, "spices")])
    assert result[0].name == "Salt"


def test_aggregate_none_quantity_separate_per_exact_unit() -> None:
    result = aggregate(
        [
            ("Garlic", None, "clove", "produce"),
            ("garlic", None, "head", "produce"),
        ]
    )
    assert len(result) == 2
    units = {r.unit for r in result}
    assert units == {"clove", "head"}


def test_aggregate_keeps_first_seen_category() -> None:
    result = aggregate(
        [
            ("Butter", 1.0, "tbsp", "dairy"),
            ("butter", 1.0, "tbsp", "baking"),
        ]
    )
    assert result[0].category == "dairy"


def test_aggregate_unitless_quantities_sum_without_unit() -> None:
    result = aggregate(
        [
            ("Eggs", 2.0, None, "dairy"),
            ("eggs", 1.0, None, "dairy"),
        ]
    )
    assert len(result) == 1
    assert result[0].quantity == 3.0
    assert result[0].unit is None


def test_aggregate_empty_input_returns_empty_list() -> None:
    assert aggregate([]) == []


# --- Shopping list generation pipeline (DB-backed) ----------------------------


@pytest.fixture
async def owner(make_user):
    user = await make_user(f"ztest-meals-{uuid.uuid4().hex[:8]}@example.com")
    return user["id"]


async def test_generate_list_aggregates_duplicate_ingredients(pool, owner) -> None:
    plan = await create_meal_plan(
        pool, owner, MealPlanCreate(name="Ztest Plan", week_start=date(2026, 7, 20))
    )
    recipe = await create_recipe(pool, owner, RecipeCreate(name="Ztest Recipe"))
    await add_ingredient(
        pool, recipe.id, RecipeIngredientCreate(name="Ztest Flour", quantity=1.0, unit="cup")
    )
    day1 = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    day2 = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 21)))
    await add_recipe_to_day(pool, owner, day1.id, recipe.id)
    await add_recipe_to_day(pool, owner, day2.id, recipe.id)

    slist = await generate_shopping_list_from_plan(pool, owner, plan.id)
    matching = [i for i in slist.items if i.name == "Ztest Flour"]
    assert len(matching) == 1
    # A recipe on 2 days contributes 2x its ingredients: 1 cup + 1 cup = 2 cups.
    assert matching[0].quantity == 2.0
    assert matching[0].unit == "cup"


async def test_generate_list_no_recipes_raises(pool, owner) -> None:
    plan = await create_meal_plan(
        pool, owner, MealPlanCreate(name="Ztest Empty Plan", week_start=date(2026, 7, 20))
    )
    with pytest.raises(ValueError, match="no recipes"):
        await generate_shopping_list_from_plan(pool, owner, plan.id)


# --- IDOR regression: child-object endpoints verify ownership ----------------


@pytest.fixture
async def alice(make_user):
    user = await make_user(f"ztest-alice-{uuid.uuid4().hex[:8]}@example.com")
    return user["id"]


@pytest.fixture
async def bob(make_user):
    user = await make_user(f"ztest-bob-{uuid.uuid4().hex[:8]}@example.com")
    return user["id"]


async def test_update_plan_day_rejects_other_owner(pool, alice, bob) -> None:
    plan = await create_meal_plan(
        pool, alice, MealPlanCreate(name="Ztest Alice Plan", week_start=date(2026, 7, 20))
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))

    assert await update_plan_day(pool, bob, day.id, MealPlanDayUpdate(notes="hacked")) is None
    updated = await update_plan_day(pool, alice, day.id, MealPlanDayUpdate(notes="mine"))
    assert updated is not None and updated.notes == "mine"


async def test_delete_plan_day_rejects_other_owner(pool, alice, bob) -> None:
    plan = await create_meal_plan(
        pool, alice, MealPlanCreate(name="Ztest Alice Plan 2", week_start=date(2026, 7, 20))
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))

    assert await delete_plan_day(pool, bob, day.id) is False
    assert await delete_plan_day(pool, alice, day.id) is True


async def test_add_recipe_to_day_rejects_other_owner(pool, alice, bob) -> None:
    plan = await create_meal_plan(
        pool, alice, MealPlanCreate(name="Ztest Alice Plan 3", week_start=date(2026, 7, 20))
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    recipe = await create_recipe(pool, alice, RecipeCreate(name="Ztest Alice Recipe"))

    assert await add_recipe_to_day(pool, bob, day.id, recipe.id) is None
    ref = await add_recipe_to_day(pool, alice, day.id, recipe.id)
    assert ref is not None and ref.id == recipe.id


async def test_remove_recipe_from_day_rejects_other_owner(pool, alice, bob) -> None:
    plan = await create_meal_plan(
        pool, alice, MealPlanCreate(name="Ztest Alice Plan 4", week_start=date(2026, 7, 20))
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    recipe = await create_recipe(pool, alice, RecipeCreate(name="Ztest Alice Recipe 2"))
    await add_recipe_to_day(pool, alice, day.id, recipe.id)

    assert await remove_recipe_from_day(pool, bob, day.id, recipe.id) is False
    assert await remove_recipe_from_day(pool, alice, day.id, recipe.id) is True


async def test_ingredient_mutation_rejects_other_owner(pool, alice, bob) -> None:
    recipe = await create_recipe(pool, alice, RecipeCreate(name="Ztest Alice Recipe 5"))
    ingredient = await add_ingredient(pool, recipe.id, RecipeIngredientCreate(name="Ztest Salt"))

    assert (
        await update_ingredient(pool, bob, ingredient.id, IngredientUpdate(name="Hacked")) is None
    )
    updated = await update_ingredient(pool, alice, ingredient.id, IngredientUpdate(name="Renamed"))
    assert updated is not None and updated.name == "Renamed"

    assert await delete_ingredient(pool, bob, ingredient.id) is False
    assert await delete_ingredient(pool, alice, ingredient.id) is True


async def test_list_item_mutation_rejects_other_owner(pool, alice, bob) -> None:
    slist = await create_shopping_list(pool, alice, ShoppingListCreate(name="Ztest Alice List"))
    item = await add_list_item(pool, slist.id, ShoppingListItemCreate(name="Ztest Milk"))

    assert await update_list_item(pool, bob, item.id, ShoppingListItemUpdate(name="Hacked")) is None
    updated = await update_list_item(pool, alice, item.id, ShoppingListItemUpdate(name="Renamed"))
    assert updated is not None and updated.name == "Renamed"

    assert await delete_list_item(pool, bob, item.id) is False
    assert await delete_list_item(pool, alice, item.id) is True
