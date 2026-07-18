"""Feedback entries: an issue/feature/user-story tracker with weighted FTS.

This module owns feedback end to end: the pydantic models, the asyncpg data
access helpers, and the FastAPI router mounted under ``/api/feedback``. The
chat agent reuses :func:`search_entries` and :func:`create_entry` for its
``search_feedback`` and ``create_feedback`` tools. Mirrors ``wiki.py``; no
shared "document" abstraction between the two.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_VALID_TYPES = {"bug", "feature", "user_story"}
_VALID_STATUSES = {"open", "in_progress", "resolved", "closed", "wontfix"}
_VALID_PRIORITIES = {"low", "medium", "high", "critical"}

# Weighted generated tsvector: title ranks highest (A), the description body
# next (B), investigation and plan lowest (C) — so a title match always beats
# a plan-only match. Created on startup alongside the chat and wiki tables
# (see db.connect()).
CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback_entries (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    type TEXT NOT NULL CHECK (type IN ('bug', 'feature', 'user_story')),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'in_progress', 'resolved', 'closed', 'wontfix')),
    priority TEXT NOT NULL DEFAULT 'medium'
        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    body TEXT NOT NULL DEFAULT '',
    investigation TEXT NOT NULL DEFAULT '',
    plan TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(investigation, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(plan, '')), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS feedback_entries_search_idx
    ON feedback_entries USING GIN (search);
"""

_SEARCH_LIMIT = 10
_HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=5, MaxWords=20, ShortWord=3"
)
_MARK_RE = re.compile(r"</?mark>")
_ENTRY_COLUMNS = (
    "id, title, tags, type, status, priority, body, investigation, plan, created_at, "
    "updated_at, job_stage, job_status, job_id, job_error"
)
_LIST_COLUMNS = (
    "id, title, tags, type, status, priority, updated_at, job_stage, job_status, job_id, job_error"
)


class EntryLocked(Exception):
    """Raised when attempting to update or delete an entry with a running job."""

    def __init__(self, stage: str | None) -> None:
        self.stage = stage
        detail = f"Entry is locked by a running {stage} job" if stage else "Entry is locked"
        super().__init__(detail)


class FeedbackEntry(BaseModel):
    """A full feedback entry row."""

    id: int
    title: str
    tags: list[str]
    type: str
    status: str
    priority: str
    body: str
    investigation: str
    plan: str
    created_at: datetime
    updated_at: datetime
    job_stage: str | None = None
    job_status: str | None = None
    job_id: uuid.UUID | None = None
    job_error: str | None = None


class FeedbackListItem(BaseModel):
    """A feedback entry in list views: no markdown fields."""

    id: int
    title: str
    tags: list[str]
    type: str
    status: str
    priority: str
    updated_at: datetime
    job_stage: str | None = None
    job_status: str | None = None
    job_id: uuid.UUID | None = None
    job_error: str | None = None


class FeedbackSearchHit(BaseModel):
    """One full-text search result."""

    id: int
    title: str
    type: str
    status: str
    priority: str
    snippet: str
    rank: float


class FeedbackCreate(BaseModel):
    title: str
    type: str
    priority: str = "medium"
    tags: list[str] = []
    body: str = ""
    investigation: str = ""
    plan: str = ""


class FeedbackUpdate(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    type: str | None = None
    status: str | None = None
    priority: str | None = None
    body: str | None = None
    investigation: str | None = None
    plan: str | None = None


async def ensure_feedback_table(pool: asyncpg.Pool) -> None:
    """Create the feedback table and its search index if they do not exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_FEEDBACK_TABLE)


async def list_entries(
    pool: asyncpg.Pool,
    *,
    type: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> list[FeedbackListItem]:
    """Return entries newest-updated first, optionally filtered exactly.

    Each of ``type``/``status``/``priority`` is optional and combinable.
    Raises :class:`ValueError` on an unrecognized value.
    """
    if type is not None and type not in _VALID_TYPES:
        raise ValueError(f"Invalid type '{type}'")
    if status is not None and status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'")
    if priority is not None and priority not in _VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'")

    clauses: list[str] = []
    params: list[str] = []
    for column, value in (("type", type), ("status", status), ("priority", priority)):
        if value is not None:
            params.append(value)
            clauses.append(f"{column} = ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = await pool.fetch(
        f"SELECT {_LIST_COLUMNS} FROM feedback_entries {where} ORDER BY updated_at DESC",
        *params,
    )
    return [FeedbackListItem(**dict(row)) for row in rows]


async def get_entry(pool: asyncpg.Pool, id: int) -> FeedbackEntry | None:
    """Return the full entry for an id, or None if it does not exist."""
    row = await pool.fetchrow(f"SELECT {_ENTRY_COLUMNS} FROM feedback_entries WHERE id = $1", id)
    return FeedbackEntry(**dict(row)) if row else None


async def create_entry(pool: asyncpg.Pool, data: FeedbackCreate) -> FeedbackEntry:
    """Insert a new entry; status is always 'open' regardless of input.

    Raises :class:`ValueError` on an empty title or an invalid type/priority.
    """
    title = data.title.strip()
    if not title:
        raise ValueError("Title must not be empty")
    if data.type not in _VALID_TYPES:
        raise ValueError(f"Invalid type '{data.type}'")
    if data.priority not in _VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{data.priority}'")
    row = await pool.fetchrow(
        "INSERT INTO feedback_entries "
        "(title, tags, type, status, priority, body, investigation, plan) "
        f"VALUES ($1, $2, $3, 'open', $4, $5, $6, $7) RETURNING {_ENTRY_COLUMNS}",
        title,
        list(data.tags),
        data.type,
        data.priority,
        data.body,
        data.investigation,
        data.plan,
    )
    return FeedbackEntry(**dict(row))


async def update_entry(pool: asyncpg.Pool, id: int, data: FeedbackUpdate) -> FeedbackEntry | None:
    """Update the provided fields of an entry, bumping updated_at.

    Returns None if the id does not exist; raises :class:`ValueError` on a
    blank title or an invalid type/status/priority, and :class:`EntryLocked`
    if a workflow job is currently running against this entry.
    """
    title = data.title.strip() if data.title is not None else None
    if data.title is not None and not title:
        raise ValueError("Title must not be empty")
    if data.type is not None and data.type not in _VALID_TYPES:
        raise ValueError(f"Invalid type '{data.type}'")
    if data.status is not None and data.status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status '{data.status}'")
    if data.priority is not None and data.priority not in _VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{data.priority}'")
    current = await pool.fetchrow(
        "SELECT job_status, job_stage FROM feedback_entries WHERE id = $1", id
    )
    if current is None:
        return None
    if current["job_status"] == "running":
        raise EntryLocked(current["job_stage"])
    row = await pool.fetchrow(
        "UPDATE feedback_entries SET "
        "title = COALESCE($2, title), "
        "tags = COALESCE($3, tags), "
        "type = COALESCE($4, type), "
        "status = COALESCE($5, status), "
        "priority = COALESCE($6, priority), "
        "body = COALESCE($7, body), "
        "investigation = COALESCE($8, investigation), "
        "plan = COALESCE($9, plan), "
        "updated_at = now() "
        f"WHERE id = $1 RETURNING {_ENTRY_COLUMNS}",
        id,
        title,
        list(data.tags) if data.tags is not None else None,
        data.type,
        data.status,
        data.priority,
        data.body,
        data.investigation,
        data.plan,
    )
    return FeedbackEntry(**dict(row)) if row else None


async def delete_entry(pool: asyncpg.Pool, id: int) -> bool:
    """Delete an entry; return True if a row was removed.

    Raises :class:`EntryLocked` if a workflow job is currently running
    against this entry.
    """
    current = await pool.fetchrow(
        "SELECT job_status, job_stage FROM feedback_entries WHERE id = $1", id
    )
    if current is None:
        return False
    if current["job_status"] == "running":
        raise EntryLocked(current["job_stage"])
    result = await pool.execute("DELETE FROM feedback_entries WHERE id = $1", id)
    return result.endswith("1")


async def search_entries(pool: asyncpg.Pool, query: str) -> list[FeedbackSearchHit]:
    """Weighted full-text search: top 10 by ts_rank with snippets.

    Uses ``websearch_to_tsquery`` so the query accepts natural phrasing. The
    generated ``search`` column weights title above body above
    investigation/plan, so a title match always outranks a plan-only match.
    An empty or stop-word-only query yields no hits.
    """
    if not query.strip():
        return []
    rows = await pool.fetch(
        "SELECT id, title, type, status, priority, "
        "ts_headline('english', body || ' ' || investigation || ' ' || plan, q, $2) "
        "AS snippet, "
        "ts_rank(search, q) AS rank "
        "FROM feedback_entries, websearch_to_tsquery('english', $1) AS q "
        "WHERE search @@ q "
        "ORDER BY rank DESC "
        f"LIMIT {_SEARCH_LIMIT}",
        query,
        _HEADLINE_OPTS,
    )
    return [FeedbackSearchHit(**dict(row)) for row in rows]


def format_search_results(hits: list[FeedbackSearchHit]) -> str:
    """Render search hits as compact text for the agent's search_feedback tool.

    One line per hit: "#N title [type/status/priority] — snippet —
    /feedback/N" (highlight markup stripped). Returns a short "no matches"
    line when empty so the model always gets usable text.
    """
    if not hits:
        return "No feedback entries matched."
    lines: list[str] = []
    for hit in hits:
        snippet = " ".join(_MARK_RE.sub("", hit.snippet).split())
        lines.append(
            f"#{hit.id} {hit.title} [{hit.type}/{hit.status}/{hit.priority}] "
            f"— {snippet} — /feedback/{hit.id}"
        )
    return "\n\n".join(lines)


router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


# /search is declared before /{id} so the literal path wins over the wildcard
# (id is also int-typed, so a "search" segment could never match it anyway).
@router.get("/search")
async def search(request: Request, q: str) -> list[dict]:
    hits = await search_entries(_pool(request), q)
    return [hit.model_dump(mode="json") for hit in hits]


@router.get("")
async def list_feedback(
    request: Request,
    type: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    try:
        entries = await list_entries(_pool(request), type=type, status=status, priority=priority)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return [entry.model_dump(mode="json") for entry in entries]


@router.post("", status_code=201)
async def create_feedback(request: Request, body: FeedbackCreate) -> dict:
    try:
        entry = await create_entry(_pool(request), body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _trigger_auto_triage(request, entry.id)
    return entry.model_dump(mode="json")


def _trigger_auto_triage(request: Request, id: int) -> None:
    """Fire-and-forget auto-triage right after a successful create.

    Imported locally: workflows.py imports search_entries/format_search_results
    from this module, so a top-level import here would be circular.
    """
    from cherryai_api.workflows import fire_and_forget_triage

    workflows = getattr(request.app.state, "workflows", None)
    if workflows is None:
        return  # not configured (e.g. a minimal test app) — best-effort only
    fire_and_forget_triage(workflows, _pool(request), id)


@router.get("/{id}")
async def get_feedback(request: Request, id: int) -> dict:
    entry = await get_entry(_pool(request), id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Feedback entry not found")
    return entry.model_dump(mode="json")


@router.put("/{id}")
async def update_feedback(request: Request, id: int, body: FeedbackUpdate) -> dict:
    try:
        entry = await update_entry(_pool(request), id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EntryLocked as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if entry is None:
        raise HTTPException(status_code=404, detail="Feedback entry not found")
    return entry.model_dump(mode="json")


@router.delete("/{id}", status_code=204)
async def delete_feedback(request: Request, id: int) -> None:
    try:
        deleted = await delete_entry(_pool(request), id)
    except EntryLocked as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Feedback entry not found")
