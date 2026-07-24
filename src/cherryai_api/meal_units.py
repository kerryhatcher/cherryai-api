"""Unit canonicalization and aggregation for shopping list generation.

Pure functions, no I/O and no new dependencies. Ingredient names stay
free-text (no master ingredient table); only the *unit* is canonicalized so
quantities can be safely summed across recipes.

Three canonical dimensions: ``mass`` (grams), ``volume`` (millilitres), and
``count`` (bare units). A small alias table maps common cooking units onto
one of those dimensions with a conversion factor. Units we don't recognize
get their own per-unit "unknown" dimension so they only ever merge with an
identical unit string — never silently coerced into a wrong conversion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

MASS = "mass"
VOLUME = "volume"
COUNT = "count"

_MASS_UNIT = "g"
_VOLUME_UNIT = "ml"
_COUNT_UNIT = "count"

_DIMENSION_UNIT = {MASS: _MASS_UNIT, VOLUME: _VOLUME_UNIT, COUNT: _COUNT_UNIT}

# alias (normalized: lowercased, trimmed, internal whitespace collapsed) ->
# (factor to convert 1 of this unit into the dimension's canonical unit, dimension)
_UNIT_ALIASES: dict[str, tuple[float, str]] = {
    # mass -> grams
    "lb": (453.592, MASS),
    "lbs": (453.592, MASS),
    "pound": (453.592, MASS),
    "pounds": (453.592, MASS),
    "oz": (28.3495, MASS),
    "ounce": (28.3495, MASS),
    "ounces": (28.3495, MASS),
    "kg": (1000.0, MASS),
    "g": (1.0, MASS),
    "gram": (1.0, MASS),
    "grams": (1.0, MASS),
    # volume -> millilitres
    "cup": (236.588, VOLUME),
    "cups": (236.588, VOLUME),
    "tbsp": (14.787, VOLUME),
    "tablespoon": (14.787, VOLUME),
    "tablespoons": (14.787, VOLUME),
    "tsp": (4.929, VOLUME),
    "teaspoon": (4.929, VOLUME),
    "teaspoons": (4.929, VOLUME),
    "fl oz": (29.574, VOLUME),
    "ml": (1.0, VOLUME),
    "l": (1000.0, VOLUME),
    "liter": (1000.0, VOLUME),
    "liters": (1000.0, VOLUME),
    "litre": (1000.0, VOLUME),
    "litres": (1000.0, VOLUME),
    "qt": (946.353, VOLUME),
    "quart": (946.353, VOLUME),
    "quarts": (946.353, VOLUME),
    "pt": (473.176, VOLUME),
    "pint": (473.176, VOLUME),
    "pints": (473.176, VOLUME),
    "gal": (3785.41, VOLUME),
    "gallon": (3785.41, VOLUME),
    "gallons": (3785.41, VOLUME),
    # count -> bare units
    "each": (1.0, COUNT),
    "unit": (1.0, COUNT),
    "units": (1.0, COUNT),
    "count": (1.0, COUNT),
    "piece": (1.0, COUNT),
    "pieces": (1.0, COUNT),
    "whole": (1.0, COUNT),
    "dozen": (12.0, COUNT),
}


def _normalize_unit_key(unit: str) -> str:
    """Lowercase, trim, and collapse internal whitespace for alias lookup."""
    return " ".join(unit.strip().lower().split())


def canonicalize(quantity: float | None, unit: str | None) -> tuple[float | None, str, str]:
    """Convert a quantity + free-text unit into a canonical dimension.

    Returns ``(canonical_quantity, dimension, canonical_unit)``:

    - ``dimension`` is ``"mass"``, ``"volume"``, ``"count"``, or
      ``"unknown:<normalized unit>"`` for units we don't recognize — the
      unit is folded into the dimension name so unrecognized units only
      ever aggregate with an exact match of themselves.
    - ``canonical_unit`` is the base display unit for the dimension
      (``"g"``, ``"ml"``, ``"count"``), or the normalized unit string itself
      for unknown dimensions.
    - A missing/blank unit is treated as ``count`` with factor 1.
    - ``quantity=None`` passes through as ``canonical_quantity=None``.
    """
    if unit is None or not unit.strip():
        factor, dimension = 1.0, COUNT
        canonical_unit = _COUNT_UNIT
    else:
        key = _normalize_unit_key(unit)
        alias = _UNIT_ALIASES.get(key)
        if alias is None:
            factor, dimension = 1.0, f"unknown:{key}"
            canonical_unit = key
        else:
            factor, dimension = alias
            canonical_unit = _DIMENSION_UNIT[dimension]

    canonical_quantity = quantity * factor if quantity is not None else None
    return canonical_quantity, dimension, canonical_unit


def format_display(qty: float | None, unit: str | None) -> tuple[float | None, str | None]:
    """Round a display quantity to at most 2 decimal places.

    Returns ``(rounded_qty, unit)``; ``unit`` passes through unchanged. An
    integral result collapses to a whole number (``2.0`` not ``2.00`` worth
    of float noise) so downstream formatting doesn't show spurious decimals.
    """
    if qty is None:
        return None, unit
    rounded = round(qty, 2)
    if rounded == int(rounded):
        rounded = float(int(rounded))
    return rounded, unit


@dataclass
class AggregatedIngredient:
    """One aggregated shopping-list line."""

    name: str
    quantity: float | None
    unit: str | None
    category: str


def aggregate(
    items: Iterable[tuple[str, float | None, str | None, str]],
) -> list[AggregatedIngredient]:
    """Aggregate ingredient rows by normalized name and unit dimension.

    ``items`` is an iterable of ``(name, quantity, unit, category)`` rows —
    typically every ``recipe_ingredients`` row pulled in for a shopping list,
    one row per day-recipe link so a recipe used on N days contributes N
    copies of its ingredients.

    Rows are grouped by ``(lower(trim(name)), dimension)`` and summed in the
    dimension's canonical unit. Two rows with the same name but different
    dimensions (e.g. "2 cups flour" and "300 g flour") are never merged —
    they surface as separate lines. Unknown units only merge with an exact
    match of themselves (see :func:`canonicalize`).

    Rows with ``quantity=None`` can't be summed, so they're grouped
    separately per exact unit string and counted instead; when more than one
    collapses together, the count is appended to the name (``"Salt — ×3"``)
    so the duplication isn't silently dropped.

    Display unit selection: every row contributing to a quantified group
    shares the same dimension by construction (that's the group key), so the
    summed canonical total is always converted back into the *first-seen*
    unit string of the group for display — e.g. "1 lb" + "8 oz" sums to
    "1.5 lb", not a canonical-grams figure. A row with no unit at all keeps
    the group unitless. The first-seen category wins for each group.
    """
    # Quantified groups: keyed by (name, dimension).
    quantified: dict[tuple[str, str], dict] = {}
    quantified_order: list[tuple[str, str]] = []

    # None-quantity groups: keyed by (name, dimension, normalized unit).
    none_groups: dict[tuple[str, str, str], dict] = {}
    none_order: list[tuple[str, str, str]] = []

    for name, quantity, unit, category in items:
        display_name = name.strip()
        key_name = display_name.lower()
        canonical_qty, dimension, _canonical_unit = canonicalize(quantity, unit)
        raw_unit_key = _normalize_unit_key(unit) if unit else ""

        if quantity is None:
            gkey = (key_name, dimension, raw_unit_key)
            group = none_groups.get(gkey)
            if group is None:
                group = {"name": display_name, "unit": unit, "category": category, "count": 0}
                none_groups[gkey] = group
                none_order.append(gkey)
            group["count"] += 1
            continue

        qkey = (key_name, dimension)
        group = quantified.get(qkey)
        if group is None:
            group = {
                "name": display_name,
                "category": category,
                "total": 0.0,
                "display_unit": unit,  # first-seen wins; converted back to at the end
            }
            quantified[qkey] = group
            quantified_order.append(qkey)
        group["total"] += canonical_qty

    results: list[AggregatedIngredient] = []

    for qkey in quantified_order:
        group = quantified[qkey]
        display_unit: str | None = group["display_unit"]
        if display_unit is None or not display_unit.strip():
            display_qty = group["total"]
            out_unit = None
        else:
            alias = _UNIT_ALIASES.get(_normalize_unit_key(display_unit))
            factor = alias[0] if alias else 1.0
            display_qty = group["total"] / factor
            out_unit = display_unit
        rounded_qty, rounded_unit = format_display(display_qty, out_unit)
        results.append(
            AggregatedIngredient(
                name=group["name"],
                quantity=rounded_qty,
                unit=rounded_unit,
                category=group["category"],
            )
        )

    for gkey in none_order:
        group = none_groups[gkey]
        name = group["name"]
        if group["count"] > 1:
            name = f"{name} — ×{group['count']}"
        results.append(
            AggregatedIngredient(
                name=name,
                quantity=None,
                unit=group["unit"],
                category=group["category"],
            )
        )

    return results
