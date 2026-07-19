"""Tests for the wiki: slug derivation, CRUD, FTS, and tool formatting.

The DB-backed tests use the ``pool`` fixture (dev Postgres) and unique
``Ztest``-prefixed titles so they never collide with real demo pages.
"""

from __future__ import annotations

import uuid

import pytest

from cherryai_api.wiki import (
    SlugExists,
    WikiCreate,
    WikiSearchHit,
    WikiUpdate,
    create_entry,
    delete_entry,
    format_search_results,
    get_entry,
    list_entries,
    normalize_folder,
    search_entries,
    slugify,
    update_entry,
)


def _unique_title(label: str) -> str:
    """A 'Ztest ...'-prefixed title whose slug lands under the test namespace."""
    return f"Ztest {uuid.uuid4().hex[:8]} {label}"


# --- Pure functions (no database) --------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Hello World", "hello-world"),
        ("  Spaced  Out  ", "spaced-out"),
        ("Already-Hyphenated", "already-hyphenated"),
        ("Multiple   Spaces & Symbols!!!", "multiple-spaces-symbols"),
        ("C++ Tips", "c-tips"),
        ("--leading and trailing--", "leading-and-trailing"),
        ("MiXeD CaSe", "mixed-case"),
        ("!!!", ""),
    ],
)
def test_slugify(title: str, expected: str) -> None:
    assert slugify(title) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("research", "research"),
        ("Research / OCR ", "research/ocr"),
        ("/research/ocr/", "research/ocr"),
        ("research//ocr", "research/ocr"),
        ("../research", "research"),
        ("Research & Models", "research-models"),
        ("a/b/c", "a/b/c"),
    ],
)
def test_normalize_folder(raw: str, expected: str) -> None:
    assert normalize_folder(raw) == expected


def test_normalize_folder_rejects_excess_depth() -> None:
    with pytest.raises(ValueError, match="3 levels"):
        normalize_folder("a/b/c/d")


def test_normalize_folder_rejects_overlong_path() -> None:
    with pytest.raises(ValueError, match="200 characters"):
        normalize_folder("/".join(["x" * 90, "y" * 90, "z" * 90]))


def test_format_search_results_empty() -> None:
    assert format_search_results([]) == "No wiki pages matched."


def test_format_search_results_strips_markup_and_links_by_path() -> None:
    hits = [
        WikiSearchHit(
            slug="cherry-care",
            title="Cherry Care",
            tags=["orchard"],
            snippet="Water the <mark>cherry</mark> trees weekly.",
            rank=0.5,
        )
    ]
    text = format_search_results(hits)
    assert "Cherry Care" in text
    assert "/wiki/cherry-care" in text
    assert "<mark>" not in text
    assert "Water the cherry trees weekly." in text


# --- Database-backed CRUD ----------------------------------------------------


async def test_crud_round_trip(pool) -> None:
    title = _unique_title("Orchard Guide")
    created = await create_entry(
        pool, WikiCreate(title=title, tags=["cherry", "care"], body="Prune yearly.")
    )
    assert created.title == title
    assert created.tags == ["cherry", "care"]
    assert created.body == "Prune yearly."

    fetched = await get_entry(pool, created.slug)
    assert fetched is not None
    assert fetched.id == created.id

    slugs = [item.slug for item in await list_entries(pool)]
    assert created.slug in slugs

    updated = await update_entry(pool, created.slug, WikiUpdate(body="Prune twice."))
    assert updated is not None
    assert updated.body == "Prune twice."
    assert updated.title == title
    assert updated.updated_at >= created.updated_at

    assert await delete_entry(pool, created.slug) is True
    assert await get_entry(pool, created.slug) is None


async def test_duplicate_slug_raises_409_signal(pool) -> None:
    title = _unique_title("Duplicate Page")
    first = await create_entry(pool, WikiCreate(title=title))
    with pytest.raises(SlugExists) as exc_info:
        await create_entry(pool, WikiCreate(title=title))
    assert exc_info.value.slug == first.slug


async def test_get_unknown_returns_none(pool) -> None:
    assert await get_entry(pool, "ztest-does-not-exist") is None


async def test_update_unknown_returns_none(pool) -> None:
    result = await update_entry(pool, "ztest-does-not-exist", WikiUpdate(body="nope"))
    assert result is None


async def test_delete_unknown_returns_false(pool) -> None:
    assert await delete_entry(pool, "ztest-does-not-exist") is False


async def test_empty_title_raises(pool) -> None:
    with pytest.raises(ValueError):
        await create_entry(pool, WikiCreate(title="   "))


async def test_update_never_changes_slug(pool) -> None:
    created = await create_entry(pool, WikiCreate(title=_unique_title("Stable Slug")))
    original_slug = created.slug
    updated = await update_entry(
        pool, original_slug, WikiUpdate(title=_unique_title("Renamed Entirely"))
    )
    assert updated is not None
    assert updated.slug == original_slug


# --- Full-text search --------------------------------------------------------


async def test_search_finds_seeded_entry(pool) -> None:
    marker = uuid.uuid4().hex[:8]
    created = await create_entry(
        pool,
        WikiCreate(
            title=_unique_title("Pollination Notes"),
            body=(
                f"Bees carry pollen between blossoms. Marker {marker} tracks "
                "this distinctive orchard entry for the search test."
            ),
        ),
    )
    hits = await search_entries(pool, "pollen blossoms orchard")
    matched = [hit for hit in hits if hit.slug == created.slug]
    assert matched, "expected the seeded entry among search hits"
    hit = matched[0]
    assert hit.rank > 0
    assert hit.snippet


async def test_search_blank_query_returns_no_hits(pool) -> None:
    assert await search_entries(pool, "   ") == []
