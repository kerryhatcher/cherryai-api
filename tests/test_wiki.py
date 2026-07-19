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
    rename_folder,
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
            folder="research",
            snippet="Water the <mark>cherry</mark> trees weekly.",
            rank=0.5,
        )
    ]
    text = format_search_results(hits)
    assert "Cherry Care" in text
    assert "/wiki/cherry-care" in text
    assert "<mark>" not in text
    assert "Water the cherry trees weekly." in text


def test_format_search_results_includes_folder() -> None:
    hits = [
        WikiSearchHit(
            slug="ocr-survey",
            title="OCR Survey",
            tags=[],
            folder="research/ocr",
            snippet="AGPL-<mark>3.0</mark> poses risk",
            rank=0.9,
        )
    ]
    assert format_search_results(hits) == (
        "OCR Survey\n  research/ocr\n  /wiki/ocr-survey\n  AGPL-3.0 poses risk"
    )


def test_format_search_results_omits_folder_line_at_root() -> None:
    hits = [
        WikiSearchHit(
            slug="demo-notes",
            title="Demo Notes",
            tags=[],
            folder="",
            snippet="a snippet",
            rank=0.5,
        )
    ]
    assert format_search_results(hits) == "Demo Notes\n  /wiki/demo-notes\n  a snippet"


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


@pytest.mark.asyncio
async def test_create_entry_normalizes_folder(pool) -> None:
    entry = await create_entry(
        pool, WikiCreate(title=_unique_title("folder create"), folder="Research / OCR ")
    )
    try:
        assert entry.folder == "research/ocr"
        fetched = await get_entry(pool, entry.slug)
        assert fetched is not None and fetched.folder == "research/ocr"
    finally:
        await delete_entry(pool, entry.slug)


@pytest.mark.asyncio
async def test_create_entry_defaults_to_root(pool) -> None:
    entry = await create_entry(pool, WikiCreate(title=_unique_title("root page")))
    try:
        assert entry.folder == ""
    finally:
        await delete_entry(pool, entry.slug)


@pytest.mark.asyncio
async def test_update_entry_moves_and_clears_folder(pool) -> None:
    entry = await create_entry(
        pool, WikiCreate(title=_unique_title("folder move"), folder="research")
    )
    try:
        moved = await update_entry(pool, entry.slug, WikiUpdate(folder="ops/runbooks"))
        assert moved is not None and moved.folder == "ops/runbooks"

        # Omitting folder leaves it alone; "" explicitly moves the page to root.
        untouched = await update_entry(pool, entry.slug, WikiUpdate(title="Ztest renamed"))
        assert untouched is not None and untouched.folder == "ops/runbooks"

        rooted = await update_entry(pool, entry.slug, WikiUpdate(folder=""))
        assert rooted is not None and rooted.folder == ""
    finally:
        await delete_entry(pool, entry.slug)


@pytest.mark.asyncio
async def test_list_and_search_expose_folder(pool) -> None:
    title = _unique_title("folder visible")
    entry = await create_entry(
        pool, WikiCreate(title=title, folder="research/ocr", body="Zqqx unique marker body")
    )
    try:
        listed = [item for item in await list_entries(pool) if item.slug == entry.slug]
        assert listed and listed[0].folder == "research/ocr"

        hits = await search_entries(pool, "Zqqx")
        assert hits and hits[0].folder == "research/ocr"
    finally:
        await delete_entry(pool, entry.slug)


# --- Folder rename ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_folder_moves_folder_and_descendants(pool) -> None:
    parent = await create_entry(
        pool, WikiCreate(title=_unique_title("rename parent"), folder="zresearch")
    )
    child = await create_entry(
        pool, WikiCreate(title=_unique_title("rename child"), folder="zresearch/ocr")
    )
    outside = await create_entry(
        pool, WikiCreate(title=_unique_title("rename outside"), folder="zresearching")
    )
    try:
        moved = await rename_folder(pool, "zresearch", "znotes")
        assert moved == 2

        assert (await get_entry(pool, parent.slug)).folder == "znotes"
        assert (await get_entry(pool, child.slug)).folder == "znotes/ocr"
        # A sibling whose name merely starts with the source must not move.
        assert (await get_entry(pool, outside.slug)).folder == "zresearching"
    finally:
        for entry in (parent, child, outside):
            await delete_entry(pool, entry.slug)


@pytest.mark.asyncio
async def test_rename_folder_returns_zero_when_unmatched(pool) -> None:
    assert await rename_folder(pool, "znosuchfolder", "zwhatever") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        ("", "znotes", "Source folder"),
        ("zresearch", "", "Target folder"),
        ("zresearch", "zresearch", "must differ"),
        ("zresearch", "zresearch/ocr", "inside the source"),
    ],
)
async def test_rename_folder_rejects_invalid_pairs(pool, source, target, message) -> None:
    with pytest.raises(ValueError, match=message):
        await rename_folder(pool, source, target)


@pytest.mark.asyncio
async def test_rename_folder_rejects_result_exceeding_max_depth(pool) -> None:
    deep = await create_entry(
        pool, WikiCreate(title=_unique_title("deep page"), folder="zsrc/mid/leaf")
    )
    try:
        # zsrc -> za/zb would push zsrc/mid/leaf to za/zb/mid/leaf: 4 levels.
        with pytest.raises(ValueError, match="levels of nesting"):
            await rename_folder(pool, "zsrc", "za/zb")
    finally:
        await delete_entry(pool, deep.slug)


@pytest.mark.asyncio
async def test_rename_folder_rejecting_too_deep_leaves_shallow_page_untouched(pool) -> None:
    shallow = await create_entry(
        pool, WikiCreate(title=_unique_title("atomic shallow page"), folder="zatomic")
    )
    deep = await create_entry(
        pool, WikiCreate(title=_unique_title("atomic deep page"), folder="zatomic/mid/leaf")
    )
    try:
        # zatomic -> za/zb would push zatomic (1 level) to za/zb (fine) but
        # zatomic/mid/leaf to za/zb/mid/leaf (4 levels, over the limit). The
        # rename must reject the whole operation rather than moving the
        # shallow page and leaving the deep one behind.
        with pytest.raises(ValueError, match="levels of nesting"):
            await rename_folder(pool, "zatomic", "za/zb")

        assert (await get_entry(pool, shallow.slug)).folder == "zatomic"
        assert (await get_entry(pool, deep.slug)).folder == "zatomic/mid/leaf"
    finally:
        for entry in (shallow, deep):
            await delete_entry(pool, entry.slug)
