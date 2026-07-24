"""Tests for meal planning: unit aggregation, IDOR guards, and Monday validation.

Pure-function tests for ``meal_units`` need no database. DB-backed tests use
the ``pool`` fixture (dev Postgres) and the ``make_user`` fixture for a fresh
owner per test, matching the pattern in ``test_wiki.py``.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from cherryai_api.meal_units import (
    aggregate,
    canonicalize,
    format_display,
    packages_needed,
    to_canonical_unit_factor,
)
from cherryai_api.meals import (
    DayAlreadyConsumed,
    IngredientUpdate,
    MealPlanCreate,
    MealPlanDayCreate,
    MealPlanDayUpdate,
    MealPlanUpdate,
    PantryItemCreate,
    PantryItemUpdate,
    RecipeCreate,
    RecipeIngredientCreate,
    ShoppingListCreate,
    ShoppingListItemCreate,
    ShoppingListItemUpdate,
    StoreCreate,
    StoreProductCreate,
    StoreProductUpdate,
    StoreUpdate,
    add_ingredient,
    add_list_item,
    add_recipe_to_day,
    add_store_product,
    commit_list_to_pantry,
    consume_day,
    create_meal_plan,
    create_recipe,
    create_shopping_list,
    create_store,
    delete_ingredient,
    delete_list_item,
    delete_pantry_item,
    delete_plan_day,
    delete_store,
    delete_store_product,
    generate_shopping_list_from_plan,
    get_plan_day,
    list_pantry_items,
    list_store_products,
    list_stores,
    remove_recipe_from_day,
    unconsume_day,
    update_ingredient,
    update_list_item,
    update_pantry_item,
    update_plan_day,
    update_store,
    update_store_product,
    upsert_pantry_item,
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


# --- Monday week_start validation ----------------------------------------------


def test_meal_plan_create_accepts_monday() -> None:
    plan = MealPlanCreate(name="Ztest", week_start=date(2026, 7, 20))  # a Monday
    assert plan.week_start == date(2026, 7, 20)


def test_meal_plan_create_rejects_non_monday() -> None:
    with pytest.raises(ValueError, match="Monday"):
        MealPlanCreate(name="Ztest", week_start=date(2026, 7, 21))  # a Tuesday


def test_meal_plan_update_accepts_monday() -> None:
    update = MealPlanUpdate(week_start=date(2026, 7, 27))
    assert update.week_start == date(2026, 7, 27)


def test_meal_plan_update_accepts_none() -> None:
    assert MealPlanUpdate(week_start=None).week_start is None


def test_meal_plan_update_rejects_non_monday() -> None:
    with pytest.raises(ValueError, match="Monday"):
        MealPlanUpdate(week_start=date(2026, 7, 22))  # a Wednesday


# --- to_canonical_unit_factor --------------------------------------------------


def test_to_canonical_unit_factor_known_unit() -> None:
    assert to_canonical_unit_factor("lb") == pytest.approx(453.592)


def test_to_canonical_unit_factor_blank_or_none_is_one() -> None:
    assert to_canonical_unit_factor(None) == 1.0
    assert to_canonical_unit_factor("  ") == 1.0


def test_to_canonical_unit_factor_unknown_unit_is_one() -> None:
    assert to_canonical_unit_factor("pinch") == 1.0


# --- Pantry: upsert-merge semantics --------------------------------------------


async def test_upsert_pantry_creates_new_item(pool, owner) -> None:
    item = await upsert_pantry_item(
        pool,
        owner,
        PantryItemCreate(name="Ztest Flour", quantity=2.0, unit="cup", category="baking"),
    )
    assert item.name == "Ztest Flour"
    assert item.quantity == 2.0
    assert item.unit == "cup"
    assert item.category == "baking"


async def test_upsert_pantry_merges_same_dimension(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Sugar", quantity=1.0, unit="lb")
    )
    merged = await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="ztest sugar", quantity=8.0, unit="oz")
    )
    items = await list_pantry_items(pool, owner)
    matching = [i for i in items if i.name.lower() == "ztest sugar"]
    assert len(matching) == 1
    assert matching[0].id == merged.id
    assert matching[0].quantity == 1.5
    assert matching[0].unit == "lb"


async def test_upsert_pantry_keeps_different_dimensions_separate(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Milk", quantity=1.0, unit="cup")
    )
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Milk", quantity=200.0, unit="g")
    )
    items = await list_pantry_items(pool, owner)
    matching = [i for i in items if i.name.lower() == "ztest milk"]
    assert len(matching) == 2
    units = {i.unit for i in matching}
    assert units == {"cup", "g"}


async def test_upsert_pantry_both_none_quantity_stays_none(pool, owner) -> None:
    await upsert_pantry_item(pool, owner, PantryItemCreate(name="Ztest Salt"))
    merged = await upsert_pantry_item(pool, owner, PantryItemCreate(name="Ztest Salt"))
    assert merged.quantity is None


async def test_upsert_pantry_none_quantity_does_not_zero_existing(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Rice", quantity=5.0, unit="lb")
    )
    merged = await upsert_pantry_item(pool, owner, PantryItemCreate(name="Ztest Rice", unit="lb"))
    assert merged.quantity == 5.0


async def test_upsert_pantry_category_overwritten_on_merge(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Butter", quantity=1.0, unit="lb", category="old")
    )
    merged = await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Butter", quantity=1.0, unit="lb", category="new")
    )
    assert merged.category == "new"


async def test_upsert_pantry_rejects_blank_name(pool, owner) -> None:
    with pytest.raises(ValueError):
        await upsert_pantry_item(pool, owner, PantryItemCreate(name="   "))


# --- Pantry: CRUD + ownership ---------------------------------------------------


async def test_pantry_update_and_delete_rejects_other_owner(pool, alice, bob) -> None:
    item = await upsert_pantry_item(pool, alice, PantryItemCreate(name="Ztest Eggs", quantity=12))
    assert await update_pantry_item(pool, bob, item.id, PantryItemUpdate(quantity=1)) is None
    updated = await update_pantry_item(pool, alice, item.id, PantryItemUpdate(quantity=6))
    assert updated is not None and updated.quantity == 6

    assert await delete_pantry_item(pool, bob, item.id) is False
    assert await delete_pantry_item(pool, alice, item.id) is True


async def test_pantry_list_scoped_to_owner(pool, alice, bob) -> None:
    await upsert_pantry_item(pool, alice, PantryItemCreate(name="Ztest Alice Only", quantity=1))
    bob_items = await list_pantry_items(pool, bob)
    assert all(i.name != "Ztest Alice Only" for i in bob_items)


# --- Consume / unconsume --------------------------------------------------------


async def _plan_day_with_recipe(
    pool, owner_id, *, ingredient_qty, ingredient_unit, ingredient_name
):
    plan = await create_meal_plan(
        pool, owner_id, MealPlanCreate(name="Ztest Consume Plan", week_start=date(2026, 7, 20))
    )
    recipe = await create_recipe(pool, owner_id, RecipeCreate(name="Ztest Consume Recipe"))
    await add_ingredient(
        pool,
        recipe.id,
        RecipeIngredientCreate(name=ingredient_name, quantity=ingredient_qty, unit=ingredient_unit),
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    await add_recipe_to_day(pool, owner_id, day.id, recipe.id)
    return day


async def test_consume_deducts_sufficient_pantry_stock(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Flour A", quantity=5.0, unit="cup")
    )
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=2.0, ingredient_unit="cup", ingredient_name="Ztest Flour A"
    )

    result = await consume_day(pool, owner, day.id)
    assert result is not None
    updated_day, report = result
    assert updated_day.consumed_at is not None
    assert len(report) == 1
    assert report[0].status == "deducted"
    assert report[0].deducted_quantity == 2.0

    pantry = await list_pantry_items(pool, owner)
    remaining = next(i for i in pantry if i.name == "Ztest Flour A")
    assert remaining.quantity == 3.0


async def test_consume_clamps_at_zero_when_insufficient(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Flour B", quantity=1.0, unit="cup")
    )
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=5.0, ingredient_unit="cup", ingredient_name="Ztest Flour B"
    )

    result = await consume_day(pool, owner, day.id)
    assert result is not None
    _, report = result
    assert report[0].status == "insufficient"
    assert report[0].deducted_quantity == 1.0

    pantry = await list_pantry_items(pool, owner)
    remaining = next(i for i in pantry if i.name == "Ztest Flour B")
    assert remaining.quantity == 0.0


async def test_consume_reports_not_tracked_when_no_pantry_row(pool, owner) -> None:
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=1.0, ingredient_unit="cup", ingredient_name="Ztest Untracked"
    )
    result = await consume_day(pool, owner, day.id)
    assert result is not None
    _, report = result
    assert report[0].status == "not_tracked"
    assert report[0].deducted_quantity is None


async def test_consume_twice_without_force_raises_409_signal(pool, owner) -> None:
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=1.0, ingredient_unit="cup", ingredient_name="Ztest Repeat"
    )
    await consume_day(pool, owner, day.id)
    with pytest.raises(DayAlreadyConsumed):
        await consume_day(pool, owner, day.id)


async def test_consume_twice_with_force_succeeds(pool, owner) -> None:
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=1.0, ingredient_unit="cup", ingredient_name="Ztest Force"
    )
    await consume_day(pool, owner, day.id)
    result = await consume_day(pool, owner, day.id, force=True)
    assert result is not None


async def test_consume_rejects_other_owner(pool, alice, bob) -> None:
    day = await _plan_day_with_recipe(
        pool,
        alice,
        ingredient_qty=1.0,
        ingredient_unit="cup",
        ingredient_name="Ztest Alice Ingredient",
    )
    assert await consume_day(pool, bob, day.id) is None


async def test_unconsume_restores_pantry_and_clears_consumed_at(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Flour C", quantity=5.0, unit="cup")
    )
    day = await _plan_day_with_recipe(
        pool, owner, ingredient_qty=2.0, ingredient_unit="cup", ingredient_name="Ztest Flour C"
    )
    await consume_day(pool, owner, day.id)

    restored_day = await unconsume_day(pool, owner, day.id)
    assert restored_day is not None
    assert restored_day.consumed_at is None

    pantry = await list_pantry_items(pool, owner)
    restored = next(i for i in pantry if i.name == "Ztest Flour C")
    assert restored.quantity == 5.0


async def test_unconsume_never_consumed_is_a_noop(pool, owner) -> None:
    day = await _plan_day_with_recipe(
        pool,
        owner,
        ingredient_qty=1.0,
        ingredient_unit="cup",
        ingredient_name="Ztest Never Consumed",
    )
    result = await unconsume_day(pool, owner, day.id)
    assert result is not None
    assert result.consumed_at is None
    # No pantry row should have been created by a no-op restore.
    pantry = await list_pantry_items(pool, owner)
    assert all(i.name != "Ztest Never Consumed" for i in pantry)


async def test_unconsume_rejects_other_owner(pool, alice, bob) -> None:
    day = await _plan_day_with_recipe(
        pool,
        alice,
        ingredient_qty=1.0,
        ingredient_unit="cup",
        ingredient_name="Ztest Alice Unconsume",
    )
    await consume_day(pool, alice, day.id)
    assert await unconsume_day(pool, bob, day.id) is None


async def test_get_plan_day_rejects_other_owner(pool, alice, bob) -> None:
    day = await _plan_day_with_recipe(
        pool,
        alice,
        ingredient_qty=1.0,
        ingredient_unit="cup",
        ingredient_name="Ztest Alice Get Day",
    )
    assert await get_plan_day(pool, bob, day.id) is None
    assert await get_plan_day(pool, alice, day.id) is not None


# --- Commit shopping list to pantry ---------------------------------------------


async def test_commit_to_pantry_adds_only_purchased_items(pool, owner) -> None:
    slist = await create_shopping_list(pool, owner, ShoppingListCreate(name="Ztest Commit List"))
    purchased = await add_list_item(
        pool, slist.id, ShoppingListItemCreate(name="Ztest Bought Milk", quantity=1.0, unit="gal")
    )
    unpurchased = await add_list_item(
        pool, slist.id, ShoppingListItemCreate(name="Ztest Unbought Eggs", quantity=12)
    )
    await update_list_item(pool, owner, purchased.id, ShoppingListItemUpdate(purchased=True))

    added = await commit_list_to_pantry(pool, owner, slist.id)
    assert added is not None
    names = {a.name for a in added}
    assert "Ztest Bought Milk" in names
    assert "Ztest Unbought Eggs" not in names

    pantry = await list_pantry_items(pool, owner)
    assert any(i.name == "Ztest Bought Milk" and i.quantity == 1.0 for i in pantry)
    assert unpurchased.name == "Ztest Unbought Eggs"  # keeps the reference used, not just created


async def test_commit_to_pantry_merges_into_existing_stock(pool, owner) -> None:
    await upsert_pantry_item(
        pool, owner, PantryItemCreate(name="Ztest Existing Rice", quantity=1.0, unit="lb")
    )
    slist = await create_shopping_list(pool, owner, ShoppingListCreate(name="Ztest Commit Merge"))
    item = await add_list_item(
        pool, slist.id, ShoppingListItemCreate(name="Ztest Existing Rice", quantity=2.0, unit="lb")
    )
    await update_list_item(pool, owner, item.id, ShoppingListItemUpdate(purchased=True))

    await commit_list_to_pantry(pool, owner, slist.id)
    pantry = await list_pantry_items(pool, owner)
    merged = next(i for i in pantry if i.name == "Ztest Existing Rice")
    assert merged.quantity == 3.0


async def test_commit_to_pantry_rejects_other_owner(pool, alice, bob) -> None:
    slist = await create_shopping_list(
        pool, alice, ShoppingListCreate(name="Ztest Alice List Commit")
    )
    assert await commit_list_to_pantry(pool, bob, slist.id) is None


# --- packages_needed (package math) --------------------------------------------


def test_packages_needed_exact_fit() -> None:
    result = packages_needed(10.0, "lb", 5.0, "lb")
    assert result == (2, 0.0, "lb")


def test_packages_needed_rounds_up_with_leftover() -> None:
    result = packages_needed(8.0, "lb", 5.0, "lb")
    assert result is not None
    packages, leftover_qty, leftover_unit = result
    assert packages == 2
    assert leftover_qty == pytest.approx(2.0)
    assert leftover_unit == "lb"


def test_packages_needed_converts_across_units_same_dimension() -> None:
    # Need 24 oz, packages are sold in 1 lb (453.592g) units -> ceil(24*28.3495/453.592) = 2
    result = packages_needed(24.0, "oz", 1.0, "lb")
    assert result is not None
    packages, leftover_qty, leftover_unit = result
    assert packages == 2
    assert leftover_unit == "oz"
    assert leftover_qty > 0


def test_packages_needed_dimension_mismatch_returns_none() -> None:
    assert packages_needed(2.0, "cup", 5.0, "lb") is None


def test_packages_needed_non_positive_package_size_returns_none() -> None:
    assert packages_needed(2.0, "lb", 0.0, "lb") is None


def test_packages_needed_zero_needed_is_zero_packages() -> None:
    result = packages_needed(0.0, "lb", 5.0, "lb")
    assert result is not None
    packages, leftover_qty, _ = result
    assert packages == 0
    assert leftover_qty == 0.0


def test_packages_needed_blank_needed_unit_falls_back_to_package_unit() -> None:
    result = packages_needed(3.0, None, 2.0, "count")
    assert result is not None
    packages, _, leftover_unit = result
    assert packages == 2
    assert leftover_unit == "count"


# --- Stores & store products ----------------------------------------------------


async def test_store_crud_round_trip(pool, owner) -> None:
    store = await create_store(pool, owner, StoreCreate(name="Ztest Sam's Club", notes="bulk"))
    assert store.name == "Ztest Sam's Club"
    assert store.notes == "bulk"

    updated = await update_store(pool, owner, store.id, StoreUpdate(name="Ztest Costco"))
    assert updated is not None and updated.name == "Ztest Costco"

    stores = await list_stores(pool, owner)
    assert any(s.id == store.id for s in stores)

    assert await delete_store(pool, owner, store.id) is True
    assert await delete_store(pool, owner, store.id) is False


async def test_create_store_rejects_blank_name(pool, owner) -> None:
    with pytest.raises(ValueError):
        await create_store(pool, owner, StoreCreate(name="   "))


async def test_store_update_and_delete_rejects_other_owner(pool, alice, bob) -> None:
    store = await create_store(pool, alice, StoreCreate(name="Ztest Alice Store"))
    assert await update_store(pool, bob, store.id, StoreUpdate(name="Hacked")) is None
    assert await delete_store(pool, bob, store.id) is False
    assert await delete_store(pool, alice, store.id) is True


async def test_store_product_crud_round_trip(pool, owner) -> None:
    store = await create_store(pool, owner, StoreCreate(name="Ztest Product Store"))
    product = await add_store_product(
        pool,
        owner,
        store.id,
        StoreProductCreate(
            ingredient_name="Ztest Chicken Tenders",
            product_name="Breaded Chicken Tenders",
            package_quantity=5.0,
            package_unit="lb",
            price_cents=1299,
        ),
    )
    assert product is not None
    assert product.ingredient_name == "Ztest Chicken Tenders"
    assert product.package_quantity == 5.0

    products = await list_store_products(pool, owner, store.id)
    assert products is not None
    assert any(p.id == product.id for p in products)

    updated = await update_store_product(
        pool, owner, product.id, StoreProductUpdate(price_cents=999)
    )
    assert updated is not None and updated.price_cents == 999

    assert await delete_store_product(pool, owner, product.id) is True
    assert await delete_store_product(pool, owner, product.id) is False


async def test_add_store_product_rejects_unowned_store(pool, alice, bob) -> None:
    store = await create_store(pool, alice, StoreCreate(name="Ztest Alice Product Store"))
    result = await add_store_product(
        pool,
        bob,
        store.id,
        StoreProductCreate(
            ingredient_name="Ztest X", product_name="X", package_quantity=1.0, package_unit="lb"
        ),
    )
    assert result is None


async def test_list_store_products_rejects_unowned_store(pool, alice, bob) -> None:
    store = await create_store(pool, alice, StoreCreate(name="Ztest Alice List Store"))
    assert await list_store_products(pool, bob, store.id) is None


async def test_store_product_update_and_delete_rejects_other_owner(pool, alice, bob) -> None:
    store = await create_store(pool, alice, StoreCreate(name="Ztest Alice Product Store 2"))
    product = await add_store_product(
        pool,
        alice,
        store.id,
        StoreProductCreate(
            ingredient_name="Ztest Y", product_name="Y", package_quantity=1.0, package_unit="lb"
        ),
    )
    assert product is not None

    assert (
        await update_store_product(pool, bob, product.id, StoreProductUpdate(product_name="Hack"))
        is None
    )
    assert await delete_store_product(pool, bob, product.id) is False
    assert await delete_store_product(pool, alice, product.id) is True


async def test_create_store_product_rejects_blank_names(pool, owner) -> None:
    store = await create_store(pool, owner, StoreCreate(name="Ztest Blank Names Store"))
    with pytest.raises(ValueError):
        await add_store_product(
            pool,
            owner,
            store.id,
            StoreProductCreate(
                ingredient_name="   ", product_name="X", package_quantity=1.0, package_unit="lb"
            ),
        )
    with pytest.raises(ValueError):
        await add_store_product(
            pool,
            owner,
            store.id,
            StoreProductCreate(
                ingredient_name="X", product_name="   ", package_quantity=1.0, package_unit="lb"
            ),
        )
