"""Tests for feedback: CRUD, filters, weighted FTS, and tool formatting.

The DB-backed tests use the ``pool`` fixture (dev Postgres) and unique
``Ztest``-prefixed titles so they never collide with real demo entries.
"""

from __future__ import annotations

import uuid

import pytest

from cherryai_api.feedback import (
    FeedbackCreate,
    FeedbackSearchHit,
    FeedbackUpdate,
    create_entry,
    delete_entry,
    format_search_results,
    get_entry,
    list_entries,
    search_entries,
    update_entry,
)


def _unique_title(label: str) -> str:
    """A 'Ztest ...'-prefixed title that lands under the test namespace."""
    return f"Ztest {uuid.uuid4().hex[:8]} {label}"


# --- Pure functions (no database) --------------------------------------------


def test_format_search_results_empty() -> None:
    assert format_search_results([]) == "No feedback entries matched."


def test_format_search_results_strips_markup_and_links_by_number() -> None:
    hits = [
        FeedbackSearchHit(
            id=42,
            title="Cherry Picker Crashes",
            type="bug",
            status="open",
            priority="high",
            snippet="The <mark>picker</mark> crashes on empty baskets.",
            rank=0.5,
        )
    ]
    text = format_search_results(hits)
    assert "#42 Cherry Picker Crashes" in text
    assert "[bug/open/high]" in text
    assert "/feedback/42" in text
    assert "<mark>" not in text
    assert "The picker crashes on empty baskets." in text


# --- Database-backed CRUD ----------------------------------------------------


async def test_crud_round_trip(pool) -> None:
    title = _unique_title("Orchard Bug")
    created = await create_entry(
        pool,
        FeedbackCreate(
            title=title,
            type="bug",
            priority="high",
            tags=["cherry", "picker"],
            body="Picker crashes.",
            investigation="Repro'd on v2.",
            plan="Patch the null check.",
        ),
    )
    assert created.title == title
    assert created.type == "bug"
    assert created.status == "open"
    assert created.priority == "high"
    assert created.tags == ["cherry", "picker"]
    assert created.body == "Picker crashes."
    assert created.investigation == "Repro'd on v2."
    assert created.plan == "Patch the null check."

    fetched = await get_entry(pool, created.id)
    assert fetched is not None
    assert fetched.id == created.id

    ids = [item.id for item in await list_entries(pool)]
    assert created.id in ids

    updated = await update_entry(
        pool, created.id, FeedbackUpdate(status="in_progress", plan="Patched.")
    )
    assert updated is not None
    assert updated.status == "in_progress"
    assert updated.plan == "Patched."
    assert updated.title == title
    assert updated.updated_at >= created.updated_at

    assert await delete_entry(pool, created.id) is True
    assert await get_entry(pool, created.id) is None


async def test_create_status_always_open_regardless_of_input(pool) -> None:
    created = await create_entry(
        pool, FeedbackCreate(title=_unique_title("New Feature"), type="feature")
    )
    assert created.status == "open"


async def test_create_default_priority_is_medium(pool) -> None:
    created = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Default Priority"), type="feature")
    )
    assert created.priority == "medium"


async def test_get_unknown_returns_none(pool) -> None:
    assert await get_entry(pool, 2_147_483_647) is None


async def test_update_unknown_returns_none(pool) -> None:
    result = await update_entry(pool, 2_147_483_647, FeedbackUpdate(plan="nope"))
    assert result is None


async def test_delete_unknown_returns_false(pool) -> None:
    assert await delete_entry(pool, 2_147_483_647) is False


async def test_empty_title_raises(pool) -> None:
    with pytest.raises(ValueError):
        await create_entry(pool, FeedbackCreate(title="   ", type="bug"))


async def test_create_invalid_type_raises(pool) -> None:
    with pytest.raises(ValueError):
        await create_entry(
            pool, FeedbackCreate(title=_unique_title("Bad Type"), type="epic")
        )


async def test_create_invalid_priority_raises(pool) -> None:
    with pytest.raises(ValueError):
        await create_entry(
            pool,
            FeedbackCreate(
                title=_unique_title("Bad Priority"), type="bug", priority="urgent"
            ),
        )


async def test_update_invalid_status_raises(pool) -> None:
    created = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Invalid Status Update"), type="bug")
    )
    with pytest.raises(ValueError):
        await update_entry(pool, created.id, FeedbackUpdate(status="cancelled"))


async def test_update_never_changes_id(pool) -> None:
    created = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Stable Id"), type="bug")
    )
    updated = await update_entry(
        pool, created.id, FeedbackUpdate(title=_unique_title("Renamed Entirely"))
    )
    assert updated is not None
    assert updated.id == created.id


# --- List filters --------------------------------------------------------


async def test_list_filters_by_type(pool) -> None:
    bug = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Filter Bug"), type="bug")
    )
    feature = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Filter Feature"), type="feature")
    )
    ids = [item.id for item in await list_entries(pool, type="bug")]
    assert bug.id in ids
    assert feature.id not in ids


async def test_list_filters_by_status(pool) -> None:
    open_entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Stays Open"), type="bug")
    )
    resolved_entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Gets Resolved"), type="bug")
    )
    await update_entry(pool, resolved_entry.id, FeedbackUpdate(status="resolved"))

    ids = [item.id for item in await list_entries(pool, status="open")]
    assert open_entry.id in ids
    assert resolved_entry.id not in ids


async def test_list_filters_by_priority(pool) -> None:
    low = await create_entry(
        pool,
        FeedbackCreate(title=_unique_title("Low Priority"), type="bug", priority="low"),
    )
    critical = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Critical Priority"), type="bug", priority="critical"
        ),
    )
    ids = [item.id for item in await list_entries(pool, priority="critical")]
    assert critical.id in ids
    assert low.id not in ids


async def test_list_filters_combine_across_groups(pool) -> None:
    match = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Combined Match"),
            type="bug",
            priority="high",
        ),
    )
    wrong_priority = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Combined Wrong Priority"),
            type="bug",
            priority="low",
        ),
    )
    items = await list_entries(pool, type="bug", status="open", priority="high")
    ids = [item.id for item in items]
    assert match.id in ids
    assert wrong_priority.id not in ids


async def test_list_invalid_type_raises(pool) -> None:
    with pytest.raises(ValueError):
        await list_entries(pool, type="epic")


async def test_list_invalid_status_raises(pool) -> None:
    with pytest.raises(ValueError):
        await list_entries(pool, status="archived")


async def test_list_invalid_priority_raises(pool) -> None:
    with pytest.raises(ValueError):
        await list_entries(pool, priority="urgent")


# --- Weighted full-text search ------------------------------------------------


async def test_search_finds_seeded_entry(pool) -> None:
    marker = uuid.uuid4().hex[:8]
    created = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Basket Overflow"),
            type="bug",
            body=(
                f"Baskets overflow during peak harvest. Marker {marker} tracks "
                "this distinctive orchard entry for the search test."
            ),
        ),
    )
    hits = await search_entries(pool, "baskets overflow harvest")
    matched = [hit for hit in hits if hit.id == created.id]
    assert matched, "expected the seeded entry among search hits"
    hit = matched[0]
    assert hit.rank > 0
    assert hit.snippet


async def test_search_blank_query_returns_no_hits(pool) -> None:
    assert await search_entries(pool, "   ") == []


async def test_search_weighting_title_outranks_plan_only(pool) -> None:
    """A title match must rank above an entry where the term is only in the plan."""
    marker = uuid.uuid4().hex[:8]
    title_hit = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title(f"Trellisworth{marker} Needs Rework"),
            type="feature",
            body="Unrelated description text.",
        ),
    )
    plan_only_hit = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Unrelated Plan Entry"),
            type="feature",
            body="Unrelated description text.",
            plan=f"Eventually rework trellisworth{marker} once time allows.",
        ),
    )

    hits = await search_entries(pool, f"trellisworth{marker}")
    ids = [hit.id for hit in hits]
    assert title_hit.id in ids
    assert plan_only_hit.id in ids

    title_rank = next(hit.rank for hit in hits if hit.id == title_hit.id)
    plan_rank = next(hit.rank for hit in hits if hit.id == plan_only_hit.id)
    assert title_rank > plan_rank
    assert ids.index(title_hit.id) < ids.index(plan_only_hit.id)
