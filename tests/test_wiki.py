"""Tests for the wiki: slug derivation, CRUD, FTS, and tool formatting.

The DB-backed tests use the ``pool`` fixture (dev Postgres) and unique
``Ztest``-prefixed titles so they never collide with real demo pages. Every
DB-backed test also uses the ``owner`` fixture (a fresh user per test, via
``make_user``) since every wiki function is scoped to a capability.
"""

from __future__ import annotations

import uuid

import pytest

from cherryai_api.authz import Capability
from cherryai_api.families import create_family
from cherryai_api.orm import async_session_maker
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
    move_entry,
    normalize_folder,
    rename_folder,
    search_entries,
    slugify,
    update_entry,
)


def _cap(user_id, family_id=None, scopes=frozenset({"wiki:read", "wiki:write", "wiki:read:adult"})):
    """Build a Capability for a given user, defaulting to personal context with full wiki scopes."""
    return Capability(user_id=user_id, family_id=family_id, role=None, scopes=scopes)


async def _make_family(pool, organizer_id):
    """Create a family and return its id."""
    async with async_session_maker() as session:
        family = await create_family(
            session, name=f"Ztest {uuid.uuid4().hex[:8]}", organizer_id=organizer_id
        )
        return family.id


def _unique_title(label: str) -> str:
    """A 'Ztest ...'-prefixed title whose slug lands under the test namespace."""
    return f"Ztest {uuid.uuid4().hex[:8]} {label}"


@pytest.fixture
async def owner(make_user):
    """A fresh user id, created via ``make_user``, to own this test's entries."""
    user = await make_user(f"ztest-{uuid.uuid4().hex[:8]}@example.com")
    return user["id"]


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


async def test_crud_round_trip(pool, owner) -> None:
    cap = _cap(owner)
    title = _unique_title("Orchard Guide")
    created = await create_entry(
        pool, cap, WikiCreate(title=title, tags=["cherry", "care"], body="Prune yearly.")
    )
    assert created.title == title
    assert created.tags == ["cherry", "care"]
    assert created.body == "Prune yearly."

    fetched = await get_entry(pool, cap, created.slug)
    assert fetched is not None
    assert fetched.id == created.id

    slugs = [item.slug for item in await list_entries(pool, cap)]
    assert created.slug in slugs

    updated = await update_entry(pool, cap, created.slug, WikiUpdate(body="Prune twice."))
    assert updated is not None
    assert updated.body == "Prune twice."
    assert updated.title == title
    assert updated.updated_at >= created.updated_at

    assert await delete_entry(pool, cap, created.slug) is True
    assert await get_entry(pool, cap, created.slug) is None


async def test_duplicate_slug_raises_409_signal(pool, owner) -> None:
    cap = _cap(owner)
    title = _unique_title("Duplicate Page")
    first = await create_entry(pool, cap, WikiCreate(title=title))
    with pytest.raises(SlugExists) as exc_info:
        await create_entry(pool, cap, WikiCreate(title=title))
    assert exc_info.value.slug == first.slug


async def test_get_unknown_returns_none(pool, owner) -> None:
    cap = _cap(owner)
    assert await get_entry(pool, cap, "ztest-does-not-exist") is None


async def test_update_unknown_returns_none(pool, owner) -> None:
    cap = _cap(owner)
    result = await update_entry(pool, cap, "ztest-does-not-exist", WikiUpdate(body="nope"))
    assert result is None


async def test_delete_unknown_returns_false(pool, owner) -> None:
    cap = _cap(owner)
    assert await delete_entry(pool, cap, "ztest-does-not-exist") is False


async def test_empty_title_raises(pool, owner) -> None:
    cap = _cap(owner)
    with pytest.raises(ValueError):
        await create_entry(pool, cap, WikiCreate(title="   "))


async def test_update_never_changes_slug(pool, owner) -> None:
    cap = _cap(owner)
    created = await create_entry(pool, cap, WikiCreate(title=_unique_title("Stable Slug")))
    original_slug = created.slug
    updated = await update_entry(
        pool, cap, original_slug, WikiUpdate(title=_unique_title("Renamed Entirely"))
    )
    assert updated is not None
    assert updated.slug == original_slug


# --- Full-text search --------------------------------------------------------


async def test_search_finds_seeded_entry(pool, owner) -> None:
    cap = _cap(owner)
    marker = uuid.uuid4().hex[:8]
    created = await create_entry(
        pool,
        cap,
        WikiCreate(
            title=_unique_title("Pollination Notes"),
            body=(
                f"Bees carry pollen between blossoms. Marker {marker} tracks "
                "this distinctive orchard entry for the search test."
            ),
        ),
    )
    hits = await search_entries(pool, cap, "pollen blossoms orchard")
    matched = [hit for hit in hits if hit.slug == created.slug]
    assert matched, "expected the seeded entry among search hits"
    hit = matched[0]
    assert hit.rank > 0
    assert hit.snippet


async def test_search_blank_query_returns_no_hits(pool, owner) -> None:
    cap = _cap(owner)
    assert await search_entries(pool, cap, "   ") == []


@pytest.mark.asyncio
async def test_create_entry_normalizes_folder(pool, owner) -> None:
    cap = _cap(owner)
    entry = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("folder create"), folder="Research / OCR ")
    )
    try:
        assert entry.folder == "research/ocr"
        fetched = await get_entry(pool, cap, entry.slug)
        assert fetched is not None and fetched.folder == "research/ocr"
    finally:
        await delete_entry(pool, cap, entry.slug)


@pytest.mark.asyncio
async def test_create_entry_defaults_to_root(pool, owner) -> None:
    cap = _cap(owner)
    entry = await create_entry(pool, cap, WikiCreate(title=_unique_title("root page")))
    try:
        assert entry.folder == ""
    finally:
        await delete_entry(pool, cap, entry.slug)


@pytest.mark.asyncio
async def test_update_entry_moves_and_clears_folder(pool, owner) -> None:
    cap = _cap(owner)
    entry = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("folder move"), folder="research")
    )
    try:
        moved = await update_entry(pool, cap, entry.slug, WikiUpdate(folder="ops/runbooks"))
        assert moved is not None and moved.folder == "ops/runbooks"

        # Omitting folder leaves it alone; "" explicitly moves the page to root.
        untouched = await update_entry(pool, cap, entry.slug, WikiUpdate(title="Ztest renamed"))
        assert untouched is not None and untouched.folder == "ops/runbooks"

        rooted = await update_entry(pool, cap, entry.slug, WikiUpdate(folder=""))
        assert rooted is not None and rooted.folder == ""
    finally:
        await delete_entry(pool, cap, entry.slug)


@pytest.mark.asyncio
async def test_list_and_search_expose_folder(pool, owner) -> None:
    cap = _cap(owner)
    title = _unique_title("folder visible")
    entry = await create_entry(
        pool,
        cap,
        WikiCreate(title=title, folder="research/ocr", body="Zqqx unique marker body"),
    )
    try:
        listed = [item for item in await list_entries(pool, cap) if item.slug == entry.slug]
        assert listed and listed[0].folder == "research/ocr"

        hits = await search_entries(pool, cap, "Zqqx")
        assert hits and hits[0].folder == "research/ocr"
    finally:
        await delete_entry(pool, cap, entry.slug)


# --- Cross-owner isolation -----------------------------------------------------


@pytest.mark.asyncio
async def test_wiki_entries_invisible_across_users(pool, make_user):
    alice = await make_user("ztest-walice@example.com")
    bob = await make_user("ztest-wbob@example.com")
    alice_cap = _cap(alice["id"])
    bob_cap = _cap(bob["id"])
    entry = await create_entry(
        pool, alice_cap, WikiCreate(title="Ztest Private", body="secret pie recipe")
    )
    assert await get_entry(pool, bob_cap, entry.slug) is None
    hits = await search_entries(pool, bob_cap, "secret pie")
    assert all(h.slug != entry.slug for h in hits)


@pytest.mark.asyncio
async def test_same_slug_allowed_for_different_owners(pool, make_user):
    alice = await make_user("ztest-salice@example.com")
    bob = await make_user("ztest-sbob@example.com")
    alice_cap = _cap(alice["id"])
    bob_cap = _cap(bob["id"])
    a = await create_entry(pool, alice_cap, WikiCreate(title="Ztest Same", body=""))
    b = await create_entry(pool, bob_cap, WikiCreate(title="Ztest Same", body=""))
    assert a.slug == b.slug


# --- Folder rename ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_folder_moves_folder_and_descendants(pool, owner) -> None:
    cap = _cap(owner)
    parent = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("rename parent"), folder="zresearch")
    )
    child = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("rename child"), folder="zresearch/ocr")
    )
    outside = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("rename outside"), folder="zresearching")
    )
    try:
        moved = await rename_folder(pool, cap, "zresearch", "znotes")
        assert moved == 2

        assert (await get_entry(pool, cap, parent.slug)).folder == "znotes"
        assert (await get_entry(pool, cap, child.slug)).folder == "znotes/ocr"
        # A sibling whose name merely starts with the source must not move.
        assert (await get_entry(pool, cap, outside.slug)).folder == "zresearching"
    finally:
        for entry in (parent, child, outside):
            await delete_entry(pool, cap, entry.slug)


@pytest.mark.asyncio
async def test_rename_folder_returns_zero_when_unmatched(pool, owner) -> None:
    cap = _cap(owner)
    assert await rename_folder(pool, cap, "znosuchfolder", "zwhatever") == 0


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
async def test_rename_folder_rejects_invalid_pairs(pool, owner, source, target, message) -> None:
    cap = _cap(owner)
    with pytest.raises(ValueError, match=message):
        await rename_folder(pool, cap, source, target)


@pytest.mark.asyncio
async def test_rename_folder_rejects_result_exceeding_max_depth(pool, owner) -> None:
    cap = _cap(owner)
    deep = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("deep page"), folder="zsrc/mid/leaf")
    )
    try:
        # zsrc -> za/zb would push zsrc/mid/leaf to za/zb/mid/leaf: 4 levels.
        with pytest.raises(ValueError, match="levels of nesting"):
            await rename_folder(pool, cap, "zsrc", "za/zb")
    finally:
        await delete_entry(pool, cap, deep.slug)


@pytest.mark.asyncio
async def test_rename_folder_rejecting_too_deep_leaves_shallow_page_untouched(pool, owner) -> None:
    cap = _cap(owner)
    shallow = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("atomic shallow page"), folder="zatomic")
    )
    deep = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("atomic deep page"), folder="zatomic/mid/leaf")
    )
    try:
        # zatomic -> za/zb would push zatomic (1 level) to za/zb (fine) but
        # zatomic/mid/leaf to za/zb/mid/leaf (4 levels, over the limit). The
        # rename must reject the whole operation rather than moving the
        # shallow page and leaving the deep one behind.
        with pytest.raises(ValueError, match="levels of nesting"):
            await rename_folder(pool, cap, "zatomic", "za/zb")

        assert (await get_entry(pool, cap, shallow.slug)).folder == "zatomic"
        assert (await get_entry(pool, cap, deep.slug)).folder == "zatomic/mid/leaf"
    finally:
        for entry in (shallow, deep):
            await delete_entry(pool, cap, entry.slug)


# --- Family-scoped operations -------------------------------------------------


@pytest.mark.asyncio
async def test_family_scoped_list(pool, owner, make_user) -> None:
    """Entries created in a family context are only visible when listing with that family."""
    cap_personal = _cap(owner)
    family_id = await _make_family(pool, owner)
    cap_family = _cap(owner, family_id=family_id)

    personal_entry = await create_entry(
        pool, cap_personal, WikiCreate(title=_unique_title("personal page"), body="personal")
    )
    family_entry = await create_entry(
        pool, cap_family, WikiCreate(title=_unique_title("family page"), body="family")
    )
    try:
        personal_slugs = [e.slug for e in await list_entries(pool, cap_personal)]
        assert personal_entry.slug in personal_slugs
        assert family_entry.slug not in personal_slugs

        family_slugs = [e.slug for e in await list_entries(pool, cap_family)]
        assert family_entry.slug in family_slugs
        assert personal_entry.slug not in family_slugs
    finally:
        await delete_entry(pool, cap_personal, personal_entry.slug)
        await delete_entry(pool, cap_family, family_entry.slug)


@pytest.mark.asyncio
async def test_family_scoped_get(pool, owner) -> None:
    """Getting an entry in the wrong family context returns None."""
    cap_personal = _cap(owner)
    family_id = await _make_family(pool, owner)
    cap_family = _cap(owner, family_id=family_id)

    entry = await create_entry(
        pool, cap_family, WikiCreate(title=_unique_title("family only"), body="secret")
    )
    try:
        # Can get it with the family capability.
        assert await get_entry(pool, cap_family, entry.slug) is not None
        # Cannot get it with the personal capability.
        assert await get_entry(pool, cap_personal, entry.slug) is None
    finally:
        await delete_entry(pool, cap_family, entry.slug)


@pytest.mark.asyncio
async def test_cross_family_isolation(pool, owner) -> None:
    """Entries in family A are invisible in family B."""
    family_a = await _make_family(pool, owner)
    family_b = await _make_family(pool, owner)
    cap_a = _cap(owner, family_id=family_a)
    cap_b = _cap(owner, family_id=family_b)

    entry_a = await create_entry(
        pool, cap_a, WikiCreate(title=_unique_title("family A page"), body="for A")
    )
    try:
        assert await get_entry(pool, cap_b, entry_a.slug) is None
        a_slugs = [e.slug for e in await list_entries(pool, cap_a)]
        assert entry_a.slug in a_slugs
        b_slugs = [e.slug for e in await list_entries(pool, cap_b)]
        assert entry_a.slug not in b_slugs
    finally:
        await delete_entry(pool, cap_a, entry_a.slug)


# --- Audience filtering -------------------------------------------------------


@pytest.mark.asyncio
async def test_child_does_not_see_adult_audience_entries(pool, owner) -> None:
    """A capability without wiki:read:adult must not see adult-audience pages."""
    family_id = await _make_family(pool, owner)
    cap_adult = _cap(
        owner, family_id=family_id, scopes=frozenset({"wiki:read", "wiki:write", "wiki:read:adult"})
    )
    cap_child = _cap(owner, family_id=family_id, scopes=frozenset({"wiki:read", "wiki:write"}))

    adult_entry = await create_entry(
        pool, cap_adult, WikiCreate(title=_unique_title("adult page"), audience="adults")
    )
    family_entry = await create_entry(
        pool, cap_adult, WikiCreate(title=_unique_title("family page"), audience="family")
    )
    try:
        # Adult sees both.
        adult_list = [e.slug for e in await list_entries(pool, cap_adult)]
        assert adult_entry.slug in adult_list
        assert family_entry.slug in adult_list

        # Child sees only the family-audience page.
        child_list = [e.slug for e in await list_entries(pool, cap_child)]
        assert family_entry.slug in child_list
        assert adult_entry.slug not in child_list

        # Child gets None (not 404) for the adult page.
        assert await get_entry(pool, cap_child, adult_entry.slug) is None
    finally:
        await delete_entry(pool, cap_adult, adult_entry.slug)
        await delete_entry(pool, cap_adult, family_entry.slug)


@pytest.mark.asyncio
async def test_audience_filtering_only_applies_in_family_context(pool, owner) -> None:
    """Audience filtering only applies in family context; personal context sees all."""
    cap = _cap(owner)
    entry = await create_entry(
        pool, cap, WikiCreate(title=_unique_title("personal adult page"), audience="adults")
    )
    try:
        # In personal context, audience is ignored — the page is visible.
        assert await get_entry(pool, cap, entry.slug) is not None
    finally:
        await delete_entry(pool, cap, entry.slug)


# --- Move endpoint ------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_personal_to_family(pool, owner) -> None:
    """Moving a personal entry to a family context updates its family_id."""
    cap_personal = _cap(owner)
    family_id = await _make_family(pool, owner)
    cap_family = _cap(owner, family_id=family_id)

    entry = await create_entry(
        pool, cap_personal, WikiCreate(title=_unique_title("move to family"), body="moveme")
    )
    try:
        moved = await move_entry(pool, cap_personal, entry.slug, family_id)
        assert moved.family_id == family_id

        # Now visible in family context, not in personal.
        assert await get_entry(pool, cap_family, entry.slug) is not None
        assert await get_entry(pool, cap_personal, entry.slug) is None
    finally:
        await delete_entry(pool, cap_family, entry.slug)


@pytest.mark.asyncio
async def test_move_family_to_personal(pool, owner) -> None:
    """Moving a family entry to personal context clears its family_id."""
    family_id = await _make_family(pool, owner)
    cap_family = _cap(owner, family_id=family_id)
    cap_personal = _cap(owner)

    entry = await create_entry(
        pool, cap_family, WikiCreate(title=_unique_title("move to personal"), body="moveme")
    )
    try:
        moved = await move_entry(pool, cap_family, entry.slug, None)
        assert moved.family_id is None

        # Now visible in personal context, not in family.
        assert await get_entry(pool, cap_personal, entry.slug) is not None
        assert await get_entry(pool, cap_family, entry.slug) is None
    finally:
        await delete_entry(pool, cap_personal, entry.slug)


@pytest.mark.asyncio
async def test_move_slug_collision_raises(pool, owner) -> None:
    """Moving to a context where the slug already exists raises SlugExists."""
    cap_personal = _cap(owner)
    family_id = await _make_family(pool, owner)
    cap_family = _cap(owner, family_id=family_id)

    # Create an entry in the family with a known slug.
    collision_title = _unique_title("collision target")
    family_entry = await create_entry(
        pool, cap_family, WikiCreate(title=collision_title, body="target")
    )
    # Create a personal entry with the same slug (possible since scopes differ).
    personal_entry = await create_entry(
        pool, cap_personal, WikiCreate(title=collision_title, body="source")
    )
    try:
        # Moving the personal entry to the family should collide.
        with pytest.raises(SlugExists):
            await move_entry(pool, cap_personal, personal_entry.slug, family_id)
    finally:
        await delete_entry(pool, cap_family, family_entry.slug)
        await delete_entry(pool, cap_personal, personal_entry.slug)
