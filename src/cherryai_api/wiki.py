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
from pydantic import BaseModel, ConfigDict, Field

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
ALTER TABLE wiki_entries ADD COLUMN IF NOT EXISTS folder TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS wiki_entries_folder_idx ON wiki_entries (folder);
"""

_SEARCH_LIMIT = 10
_HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=5, MaxWords=20, ShortWord=3"
)
_MARK_RE = re.compile(r"</?mark>")
_ENTRY_COLUMNS = "id, slug, title, tags, body, folder, created_at, updated_at"


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
    folder: str
    created_at: datetime
    updated_at: datetime


class WikiListItem(BaseModel):
    """A wiki entry in list views: no body."""

    id: uuid.UUID
    slug: str
    title: str
    tags: list[str]
    folder: str
    updated_at: datetime


class WikiSearchHit(BaseModel):
    """One full-text search result."""

    slug: str
    title: str
    tags: list[str]
    folder: str
    snippet: str
    rank: float


class WikiCreate(BaseModel):
    title: str
    tags: list[str] = []
    body: str = ""
    folder: str = ""


class WikiUpdate(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    body: str | None = None
    folder: str | None = None


class FolderRename(BaseModel):
    """Rename payload; JSON uses ``from``/``to``, which are keywords in Python."""

    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(alias="from")
    target: str = Field(alias="to")


def slugify(title: str) -> str:
    """Derive a URL slug: lowercase, alphanumerics and single hyphens, trimmed.

    Any run of non-alphanumeric characters collapses to one hyphen, and leading
    or trailing hyphens are stripped. Returns "" when the title has no
    alphanumeric characters.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")


MAX_FOLDER_DEPTH = 3
MAX_FOLDER_LENGTH = 200


def normalize_folder(raw: str) -> str:
    """Normalize a folder path: slugify each segment, drop empties, rejoin.

    Returns "" for the root. Leading, trailing, and doubled slashes and ".."
    segments all normalize away rather than erroring. Raises :class:`ValueError`
    when the result is too deep or too long.
    """
    segments = [segment for segment in (slugify(part) for part in raw.split("/")) if segment]
    if not segments:
        return ""
    if len(segments) > MAX_FOLDER_DEPTH:
        raise ValueError(f"Folder path may be at most {MAX_FOLDER_DEPTH} levels deep")
    folder = "/".join(segments)
    if len(folder) > MAX_FOLDER_LENGTH:
        raise ValueError(f"Folder path may be at most {MAX_FOLDER_LENGTH} characters")
    return folder


async def ensure_wiki_table(pool: asyncpg.Pool) -> None:
    """Create the wiki table and its search index if they do not exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_WIKI_TABLE)


async def list_entries(pool: asyncpg.Pool) -> list[WikiListItem]:
    """Return all entries, newest-updated first, without bodies."""
    rows = await pool.fetch(
        "SELECT id, slug, title, tags, folder, updated_at "
        "FROM wiki_entries ORDER BY updated_at DESC"
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
    folder = normalize_folder(data.folder)
    try:
        row = await pool.fetchrow(
            f"INSERT INTO wiki_entries (id, slug, title, tags, body, folder) "
            f"VALUES ($1, $2, $3, $4, $5, $6) RETURNING {_ENTRY_COLUMNS}",
            uuid.uuid4(),
            slug,
            title,
            list(data.tags),
            data.body,
            folder,
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
    folder = normalize_folder(data.folder) if data.folder is not None else None
    row = await pool.fetchrow(
        f"UPDATE wiki_entries SET "
        f"title = COALESCE($2, title), "
        f"tags = COALESCE($3, tags), "
        f"body = COALESCE($4, body), "
        f"folder = COALESCE($5, folder), "
        f"updated_at = now() "
        f"WHERE slug = $1 RETURNING {_ENTRY_COLUMNS}",
        slug,
        title,
        list(data.tags) if data.tags is not None else None,
        data.body,
        folder,
    )
    return WikiEntry(**dict(row)) if row else None


async def delete_entry(pool: asyncpg.Pool, slug: str) -> bool:
    """Delete an entry; return True if a row was removed."""
    result = await pool.execute("DELETE FROM wiki_entries WHERE slug = $1", slug)
    return result.endswith("1")


async def rename_folder(pool: asyncpg.Pool, source: str, target: str) -> int:
    """Rewrite the folder prefix on a folder and every descendant, atomically.

    The depth check and the rewrite run as CTEs inside a single statement, so
    they see one consistent snapshot of the table: a page inserted under
    ``source`` between "check" and "write" can no longer slip past the depth
    check and still get rewritten into a too-deep folder. The rename is also
    all-or-nothing — if any matched page would exceed ``MAX_FOLDER_DEPTH``,
    the ``UPDATE`` touches zero rows rather than moving the conforming pages
    and skipping the rest.

    Returns the number of pages moved, or 0 when no page lives under ``source``.
    Raises :class:`ValueError` for an empty, identical, self-nesting, or
    too-deep rename.
    """
    src = normalize_folder(source)
    dst = normalize_folder(target)
    if not src:
        raise ValueError("Source folder must not be empty")
    if not dst:
        raise ValueError("Target folder must not be empty")
    if src == dst:
        raise ValueError("Target folder must differ from the source folder")
    if dst.startswith(f"{src}/"):
        raise ValueError("Target folder must not be inside the source folder")

    # LIKE with a '/' guard so 'zresearch' never matches 'zresearching'. The
    # UPDATE only fires when stats.too_deep = 0, so a single too-deep match
    # blocks the write for every matched row, not just its own.
    row = await pool.fetchrow(
        """
        WITH matched AS (
            SELECT id, $2 || substring(folder from length($1) + 1) AS new_folder
              FROM wiki_entries
             WHERE folder = $1 OR folder LIKE $1 || '/%'
        ),
        stats AS (
            SELECT count(*) AS total,
                   count(*) FILTER (
                       WHERE array_length(string_to_array(new_folder, '/'), 1) > $3
                   ) AS too_deep
              FROM matched
        ),
        updated AS (
            UPDATE wiki_entries e
               SET folder = m.new_folder, updated_at = now()
              FROM matched m, stats s
             WHERE e.id = m.id AND s.too_deep = 0
            RETURNING e.id
        )
        SELECT s.total, s.too_deep, (SELECT count(*) FROM updated) AS moved
          FROM stats s
        """,
        src,
        dst,
        MAX_FOLDER_DEPTH,
    )
    if row["total"] == 0:
        return 0
    if row["too_deep"] > 0:
        raise ValueError(f"Renaming would exceed {MAX_FOLDER_DEPTH} levels of nesting")
    return row["moved"]


async def search_entries(pool: asyncpg.Pool, query: str) -> list[WikiSearchHit]:
    """Full-text search over title+body, top 10 by ts_rank with snippets.

    Uses ``websearch_to_tsquery`` so the query accepts natural phrasing. An
    empty or stop-word-only query yields no hits.
    """
    if not query.strip():
        return []
    rows = await pool.fetch(
        "SELECT slug, title, tags, folder, "
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

    One block per hit: title, the folder path when the page is not at the root,
    the ``/wiki/{slug}`` path, and a plain-text snippet (highlight markup
    stripped). Returns a short "no matches" line when empty so the model always
    gets usable text.
    """
    if not hits:
        return "No wiki pages matched."
    lines: list[str] = []
    for hit in hits:
        snippet = " ".join(_MARK_RE.sub("", hit.snippet).split())
        folder = f"  {hit.folder}\n" if hit.folder else ""
        lines.append(f"{hit.title}\n{folder}  /wiki/{hit.slug}\n  {snippet}")
    return "\n\n".join(lines)


router = APIRouter(prefix="/api/wiki", tags=["wiki"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


# /search is declared before /{slug} so the literal path wins over the wildcard.
@router.get("/search")
async def search(request: Request, q: str) -> list[dict]:
    hits = await search_entries(_pool(request), q)
    return [hit.model_dump(mode="json") for hit in hits]


@router.post("/folders/rename")
async def rename_wiki_folder(request: Request, body: FolderRename) -> dict:
    try:
        moved = await rename_folder(_pool(request), body.source, body.target)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if moved == 0:
        raise HTTPException(status_code=404, detail="No pages found in that folder")
    return {"moved": moved}


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
