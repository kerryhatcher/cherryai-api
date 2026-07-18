"""Wiki entries: markdown pages with Postgres full-text search.

This module owns the wiki end to end: the pydantic models, the asyncpg data
access helpers, and the FastAPI router mounted under ``/api/wiki``. The chat
agent reuses :func:`search_entries` for its read-only ``search_wiki`` tool.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

# Generated tsvector + GIN index give us Postgres full-text search for free on
# every write. Created on startup alongside the chat tables (see db.connect()).
CREATE_WIKI_TABLE = """
CREATE TABLE IF NOT EXISTS wiki_entries (
    id UUID PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    body TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))
    ) STORED
);
CREATE INDEX IF NOT EXISTS wiki_entries_search_idx
    ON wiki_entries USING GIN (search);
"""

_SEARCH_LIMIT = 10
_HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=5, MaxWords=20, ShortWord=3"
)
_MARK_RE = re.compile(r"</?mark>")
_ENTRY_COLUMNS = "id, slug, title, tags, body, created_at, updated_at"


class SlugExists(Exception):
    """Raised when a derived slug collides with an existing entry."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(f"A wiki page with slug '{slug}' already exists")


class WikiEntry(BaseModel):
    """A full wiki entry row."""

    id: uuid.UUID
    slug: str
    title: str
    tags: list[str]
    body: str
    created_at: datetime
    updated_at: datetime


class WikiListItem(BaseModel):
    """A wiki entry in list views: no body."""

    id: uuid.UUID
    slug: str
    title: str
    tags: list[str]
    updated_at: datetime


class WikiSearchHit(BaseModel):
    """One full-text search result."""

    slug: str
    title: str
    tags: list[str]
    snippet: str
    rank: float


class WikiCreate(BaseModel):
    title: str
    tags: list[str] = []
    body: str = ""


class WikiUpdate(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    body: str | None = None


def slugify(title: str) -> str:
    """Derive a URL slug: lowercase, alphanumerics and single hyphens, trimmed.

    Any run of non-alphanumeric characters collapses to one hyphen, and leading
    or trailing hyphens are stripped. Returns "" when the title has no
    alphanumeric characters.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")


async def ensure_wiki_table(pool: asyncpg.Pool) -> None:
    """Create the wiki table and its search index if they do not exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_WIKI_TABLE)


async def list_entries(pool: asyncpg.Pool) -> list[WikiListItem]:
    """Return all entries, newest-updated first, without bodies."""
    rows = await pool.fetch(
        "SELECT id, slug, title, tags, updated_at FROM wiki_entries ORDER BY updated_at DESC"
    )
    return [WikiListItem(**dict(row)) for row in rows]


async def get_entry(pool: asyncpg.Pool, slug: str) -> WikiEntry | None:
    """Return the full entry for a slug, or None if it does not exist."""
    row = await pool.fetchrow(f"SELECT {_ENTRY_COLUMNS} FROM wiki_entries WHERE slug = $1", slug)
    return WikiEntry(**dict(row)) if row else None


async def create_entry(pool: asyncpg.Pool, data: WikiCreate) -> WikiEntry:
    """Insert a new entry with a server-derived slug.

    Raises :class:`ValueError` on an empty/slug-less title and
    :class:`SlugExists` when the derived slug collides.
    """
    title = data.title.strip()
    if not title:
        raise ValueError("Title must not be empty")
    slug = slugify(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")
    try:
        row = await pool.fetchrow(
            f"INSERT INTO wiki_entries (id, slug, title, tags, body) "
            f"VALUES ($1, $2, $3, $4, $5) RETURNING {_ENTRY_COLUMNS}",
            uuid.uuid4(),
            slug,
            title,
            list(data.tags),
            data.body,
        )
    except asyncpg.UniqueViolationError as error:
        raise SlugExists(slug) from error
    return WikiEntry(**dict(row))


async def update_entry(pool: asyncpg.Pool, slug: str, data: WikiUpdate) -> WikiEntry | None:
    """Update the provided fields of an entry, bumping updated_at.

    The slug never changes so wikilinks stay stable. Returns None if the slug
    does not exist; raises :class:`ValueError` if title is set to blank.
    """
    title = data.title.strip() if data.title is not None else None
    if data.title is not None and not title:
        raise ValueError("Title must not be empty")
    row = await pool.fetchrow(
        f"UPDATE wiki_entries SET "
        f"title = COALESCE($2, title), "
        f"tags = COALESCE($3, tags), "
        f"body = COALESCE($4, body), "
        f"updated_at = now() "
        f"WHERE slug = $1 RETURNING {_ENTRY_COLUMNS}",
        slug,
        title,
        list(data.tags) if data.tags is not None else None,
        data.body,
    )
    return WikiEntry(**dict(row)) if row else None


async def delete_entry(pool: asyncpg.Pool, slug: str) -> bool:
    """Delete an entry; return True if a row was removed."""
    result = await pool.execute("DELETE FROM wiki_entries WHERE slug = $1", slug)
    return result.endswith("1")


async def search_entries(pool: asyncpg.Pool, query: str) -> list[WikiSearchHit]:
    """Full-text search over title+body, top 10 by ts_rank with snippets.

    Uses ``websearch_to_tsquery`` so the query accepts natural phrasing. An
    empty or stop-word-only query yields no hits.
    """
    if not query.strip():
        return []
    rows = await pool.fetch(
        "SELECT slug, title, tags, "
        "ts_headline('english', title || ' ' || body, q, $2) AS snippet, "
        "ts_rank(search, q) AS rank "
        "FROM wiki_entries, websearch_to_tsquery('english', $1) AS q "
        "WHERE search @@ q "
        "ORDER BY rank DESC "
        f"LIMIT {_SEARCH_LIMIT}",
        query,
        _HEADLINE_OPTS,
    )
    return [WikiSearchHit(**dict(row)) for row in rows]


def format_search_results(hits: list[WikiSearchHit]) -> str:
    """Render search hits as compact text for the agent's search_wiki tool.

    One block per hit: title, the ``/wiki/{slug}`` path, and a plain-text
    snippet (highlight markup stripped). Returns a short "no matches" line when
    empty so the model always gets usable text.
    """
    if not hits:
        return "No wiki pages matched."
    lines: list[str] = []
    for hit in hits:
        snippet = " ".join(_MARK_RE.sub("", hit.snippet).split())
        lines.append(f"{hit.title} (/wiki/{hit.slug})\n{snippet}")
    return "\n\n".join(lines)


router = APIRouter(prefix="/api/wiki", tags=["wiki"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


# /search is declared before /{slug} so the literal path wins over the wildcard.
@router.get("/search")
async def search(request: Request, q: str) -> list[dict]:
    hits = await search_entries(_pool(request), q)
    return [hit.model_dump(mode="json") for hit in hits]


@router.get("")
async def list_wiki(request: Request) -> list[dict]:
    entries = await list_entries(_pool(request))
    return [entry.model_dump(mode="json") for entry in entries]


@router.post("", status_code=201)
async def create_wiki(request: Request, body: WikiCreate) -> dict:
    try:
        entry = await create_entry(_pool(request), body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except SlugExists as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return entry.model_dump(mode="json")


@router.get("/{slug}")
async def get_wiki(request: Request, slug: str) -> dict:
    entry = await get_entry(_pool(request), slug)
    if entry is None:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return entry.model_dump(mode="json")


@router.put("/{slug}")
async def update_wiki(request: Request, slug: str, body: WikiUpdate) -> dict:
    try:
        entry = await update_entry(_pool(request), slug, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if entry is None:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return entry.model_dump(mode="json")


@router.delete("/{slug}", status_code=204)
async def delete_wiki(request: Request, slug: str) -> None:
    if not await delete_entry(_pool(request), slug):
        raise HTTPException(status_code=404, detail="Wiki page not found")
