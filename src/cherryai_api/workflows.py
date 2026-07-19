"""AI-driven workflows for feedback entries: triage, investigate, plan.

Three pydantic-ai agents run on Ollama cloud (via pydantic-ai's native Ollama
provider on the OpenAI-compatible chat model class) and write their results
back onto a ``feedback_entries`` row. While a job runs, the row is locked:
``feedback.update_entry``/``delete_entry`` refuse to touch it (see
``EntryLocked`` there). Mirrors the ``wiki.py``/``feedback.py`` module
pattern: models + logic + router in one file.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

import asyncpg
import httpx
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from cherryai_api.db import Database
from cherryai_api.feedback import FeedbackEntry, get_entry
from cherryai_api.feedback import format_search_results as format_feedback_results
from cherryai_api.feedback import search_entries as search_feedback_entries
from cherryai_api.memory import CogneeMemory
from cherryai_api.settings import Settings
from cherryai_api.wiki import format_search_results as format_wiki_results
from cherryai_api.wiki import search_all_entries as search_wiki_entries

_STAGES = ("triage", "investigate", "plan")

# Job-state columns added to the existing feedback_entries table (see feedback.py
# for CREATE_FEEDBACK_TABLE). Added at startup next to table creation; the
# stale-'running' cleanup below runs right after, recovering from a crashed
# process.
ALTER_FEEDBACK_JOB_COLUMNS = """
ALTER TABLE feedback_entries ADD COLUMN IF NOT EXISTS job_stage TEXT NULL;
ALTER TABLE feedback_entries ADD COLUMN IF NOT EXISTS job_status TEXT NULL;
ALTER TABLE feedback_entries ADD COLUMN IF NOT EXISTS job_id UUID NULL;
ALTER TABLE feedback_entries ADD COLUMN IF NOT EXISTS job_error TEXT NULL;
"""

_STALE_JOB_ERROR = "Job interrupted by server restart"

CLEANUP_STALE_JOBS_SQL = """
UPDATE feedback_entries SET job_status = 'failed', job_error = $1
WHERE job_status = 'running';
"""

# Questions from the triage agent live in a marker-delimited section appended
# to the body; reruns replace this section (or append/remove it) and never
# touch the human-written text around it. Reporters often type replies inline
# inside that section, so reruns must first consolidate anything they wrote
# into a second, persistent "answered" section instead of discarding it.
TRIAGE_MARKER_START = "<!-- cherryai:triage -->"
TRIAGE_MARKER_END = "<!-- /cherryai:triage -->"
_TRIAGE_HEADER = "## Questions for the reporter"
ANSWERED_MARKER_START = "<!-- cherryai:triage-answered -->"
ANSWERED_MARKER_END = "<!-- /cherryai:triage-answered -->"
_ANSWERED_HEADER = "## Answered by the reporter"


def _section_re(start: str, end: str) -> re.Pattern[str]:
    return re.compile(r"\n*" + re.escape(start) + r"(.*?)" + re.escape(end) + r"\n*", re.DOTALL)


_TRIAGE_SECTION_RE = _section_re(TRIAGE_MARKER_START, TRIAGE_MARKER_END)
_ANSWERED_SECTION_RE = _section_re(ANSWERED_MARKER_START, ANSWERED_MARKER_END)


class TriageResult(BaseModel):
    """Structured output of the triage agent."""

    type: Literal["bug", "feature", "user_story"]
    priority: Literal["low", "medium", "high", "critical"]
    tags: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    answered: list[str] = Field(default_factory=list)


@dataclass
class WorkflowRuntime:
    """The three workflow agents plus the settings and tasks they need at runtime."""

    settings: Settings
    triage_agent: Agent[None, TriageResult]
    investigate_agent: Agent[None, str]
    plan_agent: Agent[None, str]
    background_tasks: set[asyncio.Task] = field(default_factory=set)


async def ensure_workflow_columns(pool: asyncpg.Pool) -> None:
    """Add job-state columns if missing, then clear any stale 'running' jobs."""
    async with pool.acquire() as conn:
        await conn.execute(ALTER_FEEDBACK_JOB_COLUMNS)
        await conn.execute(CLEANUP_STALE_JOBS_SQL, _STALE_JOB_ERROR)


# --- Triage marker section -----------------------------------------------------


def _strip_triage_section(body: str) -> str:
    """Return body with the triage marker section removed; human text untouched."""
    return _TRIAGE_SECTION_RE.sub("", body).rstrip()


def _split_section(body: str, section_re: re.Pattern[str], header: str) -> tuple[str, str]:
    """Remove a marker section from `body`; return (rest, section payload).

    The payload is the section's inner text with the markers and `header`
    stripped — for the questions section that is the generated bullets plus
    any replies the reporter typed inline.
    """
    match = section_re.search(body)
    if match is None:
        return body, ""
    content = match.group(1).strip()
    if content.startswith(header):
        content = content[len(header) :].strip()
    return section_re.sub("", body).rstrip(), content


def _has_reporter_content(questions_payload: str) -> bool:
    """True if a questions-section payload contains reporter-typed lines.

    A pristine generated section is only '- question' bullets; any other
    non-blank line means the reporter wrote answers inline.
    """
    return any(
        line.strip() and not line.lstrip().startswith("- ")
        for line in questions_payload.splitlines()
    )


def _apply_answered_section(human_body: str, answered_text: str) -> str:
    """Append the persistent answered-info section, or nothing if it is empty."""
    if not answered_text:
        return human_body
    section = f"{ANSWERED_MARKER_START}\n{_ANSWERED_HEADER}\n{answered_text}\n{ANSWERED_MARKER_END}"
    return f"{human_body}\n\n{section}" if human_body else section


def _apply_triage_section(human_body: str, questions: list[str]) -> str:
    """Append a fresh marker section with `questions`, or none if there are none.

    `human_body` must already have any prior marker section stripped (see
    `_strip_triage_section`) so reruns replace rather than duplicate it.
    """
    if not questions:
        return human_body
    lines = "\n".join(f"- {question}" for question in questions)
    section = f"{TRIAGE_MARKER_START}\n{_TRIAGE_HEADER}\n{lines}\n{TRIAGE_MARKER_END}"
    return f"{human_body}\n\n{section}" if human_body else section


# --- Agent construction ---------------------------------------------------------


def _ollama_model(settings: Settings, model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name,
        provider=OllamaProvider(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key),
    )


def _register_search_tools(agent: Agent, database: Database, memory: CogneeMemory) -> None:
    """Register the three read-only search tools shared by investigate and plan."""

    @agent.tool_plain
    async def search_wiki(query: str) -> str:
        """Search this workspace's wiki. Returns matching pages as compact text."""
        try:
            # TODO(task 9): scope to the feedback entry's / triggering user's owner id
            hits = await search_wiki_entries(database.pool, query)
        except Exception as error:
            logger.bind(query=query).warning(f"search_wiki failed: {error}")
            return f"search_wiki failed: {error}"
        return format_wiki_results(hits)

    @agent.tool_plain
    async def search_feedback(query: str) -> str:
        """Search tracked bugs, features, and user stories. Returns matches as text."""
        try:
            hits = await search_feedback_entries(database.pool, query)
        except Exception as error:
            logger.bind(query=query).warning(f"search_feedback failed: {error}")
            return f"search_feedback failed: {error}"
        return format_feedback_results(hits)

    @agent.tool_plain
    async def search_memory(query: str) -> str:
        """Recall relevant details from prior conversations and stored knowledge."""
        try:
            return await memory.recall(query)
        except Exception as error:
            logger.bind(query=query).warning(f"search_memory failed: {error}")
            return f"search_memory failed: {error}"


TRIAGE_SYSTEM_PROMPT = (
    "You triage one feedback entry (a bug, feature request, or user story) for "
    "an internal tracker. Re-evaluate `type`, `priority`, and `tags` from "
    "scratch based on the current content below — do not just repeat the "
    "existing values unless they still fit. The reporter may have replied to "
    "earlier questions, either inline inside the previous questions section or "
    "in the already-answered summary. Consolidate EVERY piece of information "
    "the reporter has provided into `answered` (one 'question — answer' item "
    "each, preserving the reporter's meaning; carry forward all prior answered "
    "items — never drop any). If anything is still ambiguous or missing, list "
    "only genuinely unanswered clarifying questions in `questions`; otherwise "
    "return an empty list."
)

INVESTIGATE_SYSTEM_PROMPT = (
    "You investigate one feedback entry (a bug, feature request, or user "
    "story) for an internal tracker. Use search_wiki, search_feedback, and "
    "search_memory to gather relevant context, then write a concise markdown "
    "investigation covering likely root cause or context, related entries or "
    "pages you found, and open questions. If prior investigation notes are "
    "given, incorporate and refine them rather than starting over."
)

PLAN_SYSTEM_PROMPT = (
    "You write an implementation plan for one feedback entry (a bug, feature "
    "request, or user story) for an internal tracker. Use search_wiki, "
    "search_feedback, and search_memory to ground the plan in this "
    "workspace's existing conventions and prior work, then write a concise "
    "markdown plan covering steps, affected areas, and risks. If a prior plan "
    "is given, incorporate and refine it rather than starting over."
)


def build_triage_agent(settings: Settings) -> Agent[None, TriageResult]:
    """Build the triage agent: structured output, no tools."""
    return Agent(
        _ollama_model(settings, settings.workflow_triage_model),
        output_type=TriageResult,
        instructions=TRIAGE_SYSTEM_PROMPT,
    )


def build_investigate_agent(
    settings: Settings, database: Database, memory: CogneeMemory
) -> Agent[None, str]:
    """Build the investigate agent: markdown output, read-only search tools."""
    agent: Agent[None, str] = Agent(
        _ollama_model(settings, settings.workflow_investigate_model),
        instructions=INVESTIGATE_SYSTEM_PROMPT,
    )
    _register_search_tools(agent, database, memory)
    return agent


def build_plan_agent(
    settings: Settings, database: Database, memory: CogneeMemory
) -> Agent[None, str]:
    """Build the plan agent: markdown output, read-only search tools."""
    agent: Agent[None, str] = Agent(
        _ollama_model(settings, settings.workflow_plan_model),
        instructions=PLAN_SYSTEM_PROMPT,
    )
    _register_search_tools(agent, database, memory)
    return agent


def build_workflow_runtime(
    settings: Settings, database: Database, memory: CogneeMemory
) -> WorkflowRuntime:
    """Build the three workflow agents once, for the app's lifetime."""
    return WorkflowRuntime(
        settings=settings,
        triage_agent=build_triage_agent(settings),
        investigate_agent=build_investigate_agent(settings, database, memory),
        plan_agent=build_plan_agent(settings, database, memory),
    )


# --- Prompts for a specific entry -----------------------------------------------


def _triage_prompt(
    entry: FeedbackEntry, human_body: str, answered: str, prior_questions: str
) -> str:
    return (
        f"Title: {entry.title}\n"
        f"Current type: {entry.type}\n"
        f"Current priority: {entry.priority}\n"
        f"Current tags: {', '.join(entry.tags) or '(none)'}\n\n"
        f"Body:\n{human_body or '(empty)'}\n\n"
        f"Already answered by the reporter:\n{answered or '(none)'}\n\n"
        f"Previous questions to the reporter (may contain inline replies):\n"
        f"{prior_questions or '(none)'}\n\n"
        f"Investigation notes:\n{entry.investigation or '(none)'}\n\n"
        f"Plan:\n{entry.plan or '(none)'}"
    )


def _investigate_prompt(entry: FeedbackEntry) -> str:
    prior = (
        f"\n\nPrior investigation notes (revise/incorporate, don't just repeat):\n"
        f"{entry.investigation}"
        if entry.investigation.strip()
        else ""
    )
    return (
        f"Title: {entry.title}\n"
        f"Type: {entry.type} | Priority: {entry.priority} | Status: {entry.status}\n"
        f"Tags: {', '.join(entry.tags) or '(none)'}\n\n"
        f"Body:\n{_strip_triage_section(entry.body) or '(empty)'}"
        f"{prior}"
    )


def _plan_prompt(entry: FeedbackEntry) -> str:
    prior = (
        f"\n\nPrior plan (revise/incorporate, don't just repeat):\n{entry.plan}"
        if entry.plan.strip()
        else ""
    )
    return (
        f"Title: {entry.title}\n"
        f"Type: {entry.type} | Priority: {entry.priority} | Status: {entry.status}\n"
        f"Tags: {', '.join(entry.tags) or '(none)'}\n\n"
        f"Body:\n{_strip_triage_section(entry.body) or '(empty)'}\n\n"
        f"Investigation:\n{entry.investigation or '(none)'}"
        f"{prior}"
    )


# --- Model preflight -------------------------------------------------------------


async def _fetch_available_models(settings: Settings) -> set[str]:
    """Return the model ids/names the configured Ollama endpoint currently serves."""
    headers = (
        {"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {}
    )
    url = f"{settings.ollama_base_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    entries = data.get("data") or data.get("models") or []
    names = (entry.get("id") or entry.get("name") for entry in entries)
    return {name for name in names if name}


def _model_for_stage(settings: Settings, stage: str) -> str:
    return {
        "triage": settings.workflow_triage_model,
        "investigate": settings.workflow_investigate_model,
        "plan": settings.workflow_plan_model,
    }[stage]


async def _preflight_model(settings: Settings, model_name: str) -> None:
    """Raise RuntimeError with a clear message if `model_name` isn't live-served.

    Runs before any agent call so a missing model fails the job fast instead of
    hanging or surfacing an opaque provider error.
    """
    available = await _fetch_available_models(settings)
    if model_name not in available:
        raise RuntimeError(f"Model '{model_name}' is not available from Ollama cloud")


# --- Job lock: race-safe claim, run, clear/fail ---------------------------------


async def start_job(
    pool: asyncpg.Pool, runtime: WorkflowRuntime, id: int, stage: str
) -> uuid.UUID | None:
    """Race-safely claim the job lock and launch the stage runner as a tracked task.

    Returns the new job id, or None if the entry does not exist or a job is
    already running — callers distinguish 404 vs 409 by re-checking the row.
    """
    job_id = uuid.uuid4()
    row = await pool.fetchrow(
        "UPDATE feedback_entries SET job_stage = $2, job_status = 'running', "
        "job_id = $3, job_error = NULL "
        "WHERE id = $1 AND (job_status IS NULL OR job_status <> 'running') "
        "RETURNING id",
        id,
        stage,
        job_id,
    )
    if row is None:
        return None
    task = asyncio.create_task(_run_job(pool, runtime, id, stage, job_id))
    runtime.background_tasks.add(task)
    task.add_done_callback(runtime.background_tasks.discard)
    return job_id


async def _clear_job(pool: asyncpg.Pool, id: int) -> None:
    await pool.execute(
        "UPDATE feedback_entries SET job_stage = NULL, job_status = NULL, "
        "job_id = NULL, job_error = NULL WHERE id = $1",
        id,
    )


async def _fail_job(pool: asyncpg.Pool, id: int, error: str) -> None:
    """Persist the failure; job_stage/job_id are left as-is so the UI can show them."""
    await pool.execute(
        "UPDATE feedback_entries SET job_status = 'failed', job_error = $2 WHERE id = $1",
        id,
        error,
    )


async def _run_triage(pool: asyncpg.Pool, runtime: WorkflowRuntime, entry: FeedbackEntry) -> None:
    human_body, prior_answered = _split_section(entry.body, _ANSWERED_SECTION_RE, _ANSWERED_HEADER)
    human_body, prior_questions = _split_section(human_body, _TRIAGE_SECTION_RE, _TRIAGE_HEADER)
    result = await runtime.triage_agent.run(
        _triage_prompt(entry, human_body, prior_answered, prior_questions)
    )
    triage: TriageResult = result.output
    if triage.answered:
        answered_text = "\n".join(f"- {item}" for item in triage.answered)
    else:
        # Safety net: the agent reported nothing answered, but reporter-provided
        # text must never be dropped — carry it forward verbatim instead.
        preserved = [prior_answered] if prior_answered else []
        if _has_reporter_content(prior_questions):
            preserved.append(prior_questions)
        answered_text = "\n\n".join(preserved)
        if preserved:
            logger.bind(feedback_id=entry.id).info(
                "Triage preserved reporter content verbatim (agent returned no answered items)"
            )
    new_body = _apply_triage_section(
        _apply_answered_section(human_body, answered_text), triage.questions
    )
    await pool.execute(
        "UPDATE feedback_entries SET type = $2, priority = $3, tags = $4, body = $5, "
        "updated_at = now() WHERE id = $1",
        entry.id,
        triage.type,
        triage.priority,
        list(triage.tags),
        new_body,
    )


async def _run_investigate(
    pool: asyncpg.Pool, runtime: WorkflowRuntime, entry: FeedbackEntry
) -> None:
    result = await runtime.investigate_agent.run(_investigate_prompt(entry))
    await pool.execute(
        "UPDATE feedback_entries SET investigation = $2, updated_at = now() WHERE id = $1",
        entry.id,
        result.output,
    )


async def _run_plan(pool: asyncpg.Pool, runtime: WorkflowRuntime, entry: FeedbackEntry) -> None:
    result = await runtime.plan_agent.run(_plan_prompt(entry))
    await pool.execute(
        "UPDATE feedback_entries SET plan = $2, updated_at = now() WHERE id = $1",
        entry.id,
        result.output,
    )


async def _run_job(
    pool: asyncpg.Pool, runtime: WorkflowRuntime, id: int, stage: str, job_id: uuid.UUID
) -> None:
    try:
        await _preflight_model(runtime.settings, _model_for_stage(runtime.settings, stage))
        entry = await get_entry(pool, id)
        if entry is None:
            logger.bind(feedback_id=id, stage=stage).warning("Entry vanished before job ran")
            return
        if stage == "triage":
            await _run_triage(pool, runtime, entry)
        elif stage == "investigate":
            await _run_investigate(pool, runtime, entry)
        else:
            await _run_plan(pool, runtime, entry)
        await _clear_job(pool, id)
    except Exception as error:
        logger.bind(feedback_id=id, stage=stage, job_id=str(job_id)).warning(
            f"Workflow job failed: {error}"
        )
        await _fail_job(pool, id, str(error))


def fire_and_forget_triage(runtime: WorkflowRuntime, pool: asyncpg.Pool, id: int) -> None:
    """Kick off auto-triage after a successful create without blocking the caller.

    Shared by the HTTP create route and the chat agent's create_feedback tool.
    Any failure to start is logged and swallowed — auto-triage must never delay
    or fail entry creation.
    """

    async def _run() -> None:
        try:
            job_id = await start_job(pool, runtime, id, "triage")
            if job_id is None:
                logger.bind(feedback_id=id).warning("Auto-triage could not start")
        except Exception as error:
            logger.bind(feedback_id=id).warning(f"Auto-triage failed to start: {error}")

    task = asyncio.create_task(_run())
    runtime.background_tasks.add(task)
    task.add_done_callback(runtime.background_tasks.discard)


# --- Router ----------------------------------------------------------------------

router = APIRouter(prefix="/api/feedback", tags=["feedback-workflows"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


def _runtime(request: Request) -> WorkflowRuntime:
    return request.app.state.workflows


async def _trigger(request: Request, id: int, stage: str) -> dict:
    pool = _pool(request)
    job_id = await start_job(pool, _runtime(request), id, stage)
    if job_id is None:
        entry = await get_entry(pool, id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Feedback entry not found")
        raise HTTPException(status_code=409, detail=f"A {entry.job_stage} job is already running")
    return {"job_id": str(job_id), "stage": stage}


@router.post("/{id}/triage", status_code=202)
async def trigger_triage(request: Request, id: int) -> dict:
    return await _trigger(request, id, "triage")


@router.post("/{id}/investigate", status_code=202)
async def trigger_investigate(request: Request, id: int) -> dict:
    return await _trigger(request, id, "investigate")


@router.post("/{id}/plan", status_code=202)
async def trigger_plan(request: Request, id: int) -> dict:
    return await _trigger(request, id, "plan")
