"""FastAPI HTTP surface for CherryAI (cookie-authenticated, multi-user)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from cherryai_api.admin import router as admin_router
from cherryai_api.agent import AgentDeps, build_agent, run_turn, stream_turn, strip_leaked_reasoning
from cherryai_api.agent_audit import log_tool_call
from cherryai_api.auth import auth_backend, current_verified_user, fastapi_users_app, require_chat
from cherryai_api.authz import Capability, assert_rls_enforced, get_capability
from cherryai_api.calendar import router as calendar_router
from cherryai_api.contacts import router as contacts_router
from cherryai_api.db import build_database, make_session_title
from cherryai_api.db_migrations import run_migrations_to_head
from cherryai_api.email import router as email_router
from cherryai_api.facts import build_extractor_agent, build_judge_agent, extract_and_save_facts
from cherryai_api.families import families_router
from cherryai_api.family_context import FamilyContextMiddleware
from cherryai_api.feedback import router as feedback_router
from cherryai_api.frontend_errors import (
    FRONTEND_ERROR_RETENTION_DAYS,
    FrontendError,
    prune_frontend_errors,
)
from cherryai_api.integrations import router as integrations_router
from cherryai_api.logging_setup import setup_file_logging
from cherryai_api.meals import router as meals_router
from cherryai_api.memory import build_memory
from cherryai_api.orm import async_session_maker, get_async_session
from cherryai_api.planner import router as planner_router
from cherryai_api.settings import get_settings
from cherryai_api.telemetry import configure_telemetry, instrument_dependencies
from cherryai_api.users import User, UserCreate, UserRead, UserUpdate
from cherryai_api.wiki import router as wiki_router
from cherryai_api.workflows import build_workflow_runtime
from cherryai_api.workflows import router as workflows_router

# Per-field caps for POST /api/log/error. Authentication (see `log_error`'s
# `current_verified_user` dependency) is now the primary abuse control, but
# it does not make these caps redundant: a logged-in client can still send an
# oversized stack trace, and a buggy or compromised client can still loop,
# so unbounded input from a verified caller remains a storage-abuse vector.
# Values are generous enough to hold a real browser stack trace in full
# (errorLogger.ts already caps its own `stack` at 4096 chars, so 16384
# leaves headroom for other/future callers) while still bounding worst-case
# row size to a few tens of KB. Oversized input is truncated rather than
# rejected with a 422 — see the field_validator below and the docstring on
# `log_error` for why.
_MAX_MESSAGE_LEN = 4096
_MAX_SOURCE_LEN = 2048
_MAX_STACK_LEN = 16384
_MAX_URL_LEN = 2048
_MAX_USER_AGENT_LEN = 512
_MAX_TIMESTAMP_LEN = 64
_MAX_CONTEXT_LEN = 256
# `lineno`/`colno` land in PG `integer` columns; anything outside int4 range
# raises a DataError at commit time and would silently drop the whole report,
# so clamp instead.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str


class LogErrorRequest(BaseModel):
    message: str
    source: str | None = None
    lineno: int | None = None
    colno: int | None = None
    stack: str | None = None
    url: str | None = None
    user_agent: str | None = None
    timestamp: str | None = None
    context: str | None = None

    @field_validator("message")
    @classmethod
    def _truncate_message(cls, value: str) -> str:
        return value[:_MAX_MESSAGE_LEN]

    @field_validator("source")
    @classmethod
    def _truncate_source(cls, value: str | None) -> str | None:
        return value[:_MAX_SOURCE_LEN] if value is not None else None

    @field_validator("lineno", "colno")
    @classmethod
    def _clamp_position(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return max(_INT32_MIN, min(_INT32_MAX, value))

    @field_validator("stack")
    @classmethod
    def _truncate_stack(cls, value: str | None) -> str | None:
        return value[:_MAX_STACK_LEN] if value is not None else None

    @field_validator("url")
    @classmethod
    def _truncate_url(cls, value: str | None) -> str | None:
        return value[:_MAX_URL_LEN] if value is not None else None

    @field_validator("user_agent")
    @classmethod
    def _truncate_user_agent(cls, value: str | None) -> str | None:
        return value[:_MAX_USER_AGENT_LEN] if value is not None else None

    @field_validator("timestamp")
    @classmethod
    def _truncate_timestamp(cls, value: str | None) -> str | None:
        return value[:_MAX_TIMESTAMP_LEN] if value is not None else None

    @field_validator("context")
    @classmethod
    def _truncate_context(cls, value: str | None) -> str | None:
        return value[:_MAX_CONTEXT_LEN] if value is not None else None


# How often the background prune loop below wakes up. DO App Platform's job
# kinds (PRE_DEPLOY / POST_DEPLOY / FAILED_DEPLOY) have no scheduled/cron
# option, and the API runs with instance_count: 1, so an in-process asyncio
# loop is the retention mechanism rather than an external scheduler. The
# retention window itself lives in `FRONTEND_ERROR_RETENTION_DAYS`
# (frontend_errors.py) — not duplicated here.
_FRONTEND_ERROR_PRUNE_INTERVAL_SECONDS = 24 * 60 * 60


async def _prune_frontend_errors_periodically() -> None:
    """Delete old `frontend_errors` rows once at startup, then roughly daily, forever.

    Runs as a long-lived background task for the life of the process (see
    `lifespan`), which cancels it on shutdown. Each pass opens its own
    session rather than sharing one across the whole task lifetime, and is
    wrapped in a broad except: a prune failure (e.g. a transient database
    blip) must never crash the process — matching how `log_error` already
    tolerates database failure. It simply logs and tries again next tick.
    """
    while True:
        try:
            async with async_session_maker() as session:
                deleted = await prune_frontend_errors(session)
            logger.info(
                f"Pruned {deleted} frontend_errors row(s) older than "
                f"{FRONTEND_ERROR_RETENTION_DAYS}d"
            )
        except Exception as exc:
            logger.warning(f"Frontend error prune pass failed: {exc}")
        await asyncio.sleep(_FRONTEND_ERROR_PRUNE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the database pool and build the agent once per process."""
    settings = get_settings()
    setup_file_logging(settings.log_dir)
    # FastAPI/ASGI instrumentation is NOT done here — it must happen at import
    # time or it silently produces no spans. See configure_telemetry's docstring
    # and its call site below the app definition.
    instrument_dependencies()
    if settings.run_migrations_on_startup:
        await asyncio.to_thread(run_migrations_to_head)
    database = build_database()
    await database.connect()
    await assert_rls_enforced(database.pool)
    # Kept solely for the workflow runtime, which is a workspace-level
    # background pipeline not tied to a requesting user. Per-request chat
    # traffic builds its own memory in send_message.
    default_memory = build_memory()
    workflows = build_workflow_runtime(settings, database, default_memory)
    agent = build_agent(settings, database=database, workflows=workflows)
    app.state.settings = settings
    app.state.db = database
    app.state.workflows = workflows
    app.state.agent = agent
    if settings.fact_extraction_enabled:
        app.state.fact_extractor_agent = build_extractor_agent(settings)
        app.state.fact_judge_agent = build_judge_agent(settings)
    else:
        app.state.fact_extractor_agent = None
        app.state.fact_judge_agent = None
    prune_task = asyncio.create_task(_prune_frontend_errors_periodically())
    logger.info("CherryAI API started")
    try:
        yield
    finally:
        prune_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await prune_task
        await database.close()
        logger.info("CherryAI API stopped")


app = FastAPI(title="CherryAI API", lifespan=lifespan)
app.add_middleware(FamilyContextMiddleware)
# Must stay at module scope, before the app ever serves an ASGI scope. Moving
# this into the lifespan silently disables all HTTP span collection — Starlette
# caches its middleware stack during the "lifespan" scope, before the lifespan
# body runs, so instrumenting from there patches a method that is never called
# again. No error is raised when this is wrong. See telemetry.configure_telemetry.
configure_telemetry(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(wiki_router)
app.include_router(feedback_router)
app.include_router(workflows_router)
app.include_router(calendar_router)
app.include_router(contacts_router)
app.include_router(email_router)
app.include_router(integrations_router)
app.include_router(meals_router)
app.include_router(planner_router)
app.include_router(families_router)
app.include_router(fastapi_users_app.get_auth_router(auth_backend), prefix="/auth", tags=["auth"])
app.include_router(
    fastapi_users_app.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users_app.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# Middleware: request ID for correlating frontend error reports with API logs
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique request ID to every request for log correlation."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Exception handler: log unhandled exceptions with full stack traces
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    logger.opt(exception=True).error(
        f"Unhandled exception [{rid}] {request.method} {request.url.path}: {exc}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": rid},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/log/error")
async def log_error(
    body: LogErrorRequest,
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> dict:
    """Accept a frontend error report and persist it to Postgres.

    Requires an authenticated, verified session — errors occurring before
    login (the login page itself, unauthenticated bootstrap) are
    intentionally dropped rather than accepted anonymously; that tradeoff
    was made explicitly to close off an unauthenticated write path into
    Postgres. A persistence failure is logged and swallowed rather than
    surfaced to the caller: `errorLogger.ts` fires this request with
    `keepalive: true` and ignores the response entirely, so failing loudly
    here would only lose the report with no compensating benefit — a 500
    would additionally fill the API's own logs with noise during the exact
    incident the report is meant to help diagnose.
    """
    url = body.url or request.headers.get("referer", "")
    user_agent = body.user_agent or request.headers.get("user-agent", "")

    row = FrontendError(
        user_id=user.id,
        message=body.message,
        source=body.source,
        lineno=body.lineno,
        colno=body.colno,
        stack=body.stack,
        url=url[:_MAX_URL_LEN],
        user_agent=user_agent[:_MAX_USER_AGENT_LEN],
        client_ip=request.client.host if request.client else "",
        client_timestamp=body.timestamp,
        context=body.context,
    )

    try:
        session.add(row)
        await session.commit()
    except Exception as exc:
        # Deliberately broad: no failure mode of this write may escape. Not
        # every failure is a SQLAlchemyError — when Postgres is unreachable,
        # asyncio's socket layer raises ConnectionRefusedError (an OSError)
        # before asyncpg produces anything SQLAlchemy can wrap, and that is
        # precisely the incident the frontend is trying to report.
        # CancelledError is a BaseException, so client disconnects still
        # cancel normally.
        #
        # The exception text stays server-side: SQLAlchemy renders the failing
        # statement and its bound parameters into str(exc), and connection
        # errors render the database host and port. The caller is now a
        # verified user rather than an anonymous one, but that grants no
        # standing to see INSERT statements, bound parameters, or the
        # database host/port, so the detail stays opaque regardless.
        logger.warning(f"Failed to persist frontend error report: {exc}")
        return {"status": "error", "detail": "failed to persist error report"}

    return {"status": "ok"}


@app.get("/api/health")
async def health() -> dict:
    """Report liveness and dependency reachability."""
    db_ok = False
    with contextlib.suppress(Exception):
        db_ok = await app.state.db.ping()
    status = "ok" if db_ok else "degraded"
    return {"status": status, "postgres": db_ok}


@app.get("/api/sessions")
async def list_sessions(user: User = Depends(require_chat)) -> list[dict]:  # noqa: B008
    sessions = await app.state.db.list_sessions(user.id)
    return [s.model_dump(mode="json") for s in sessions]


@app.post("/api/sessions", status_code=201)
async def create_session(
    body: CreateSessionRequest | None = None,
    user: User = Depends(require_chat),  # noqa: B008
) -> dict:
    title = body.title if body and body.title else "New chat"
    session = await app.state.db.create_session(title, user.id)
    return session.model_dump(mode="json")


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(
    session_id: uuid.UUID,
    user: User = Depends(require_chat),  # noqa: B008
) -> list[dict]:
    session = await app.state.db.get_session(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await app.state.db.list_messages(session_id)
    return [m.model_dump(mode="json") for m in messages]


def _remember_turn_in_background(memory, prompt: str, answer: str) -> None:
    """Persist a turn to Cognee without blocking the HTTP response."""

    async def _run() -> None:
        try:
            await memory.remember_turn(prompt, answer)
        except Exception as error:
            logger.warning(f"Cognee remember_turn failed: {error}")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


_background_tasks: set[asyncio.Task] = set()


def _extract_facts_in_background(extractor_agent, judge_agent, memory, message: str) -> None:
    """Extract and save durable facts from a user message without blocking the reply.

    A no-op when fact extraction is disabled (agents are None). Any failure is
    logged and swallowed by `extract_and_save_facts` itself; this wrapper only
    guards against the task-spawning step failing.
    """
    if extractor_agent is None or judge_agent is None:
        return

    async def _run() -> None:
        try:
            await extract_and_save_facts(extractor_agent, judge_agent, memory, message)
        except Exception as error:
            logger.warning(f"Fact extraction pipeline failed: {error}")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.post("/api/sessions/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID,
    body: SendMessageRequest,
    user: User = Depends(require_chat),  # noqa: B008
    capability: Capability = Depends(get_capability),  # noqa: B008
):
    """Persist the user message and stream the assistant reply as SSE."""
    # Chat gate: child sessions with chat disabled get 403.
    if not capability.has("chat"):
        raise HTTPException(
            status_code=403,
            detail={"code": "module_disabled", "module": "chat"},
        )
    db = app.state.db
    agent = app.state.agent

    session = await db.get_session(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    prompt = body.content.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Message content is empty")

    memory = build_memory(user.memory_dataset, str(session_id))
    deps = AgentDeps(memory=memory, user_id=user.id, capability=capability)

    # Audit: log agent activity for child sessions.
    is_child = capability.role == "child" if capability.role else False
    if is_child:
        await log_tool_call(db.pool, session_id, user.id, "chat.message", body.content[:100])

    was_empty = await db.is_session_empty(session_id)
    await db.add_message(session_id, "user", prompt)
    if was_empty:
        await db.set_title(session_id, make_session_title(prompt))
    _extract_facts_in_background(
        app.state.fact_extractor_agent, app.state.fact_judge_agent, memory, prompt
    )

    history = await _load_history(db, session_id, exclude_last_user=prompt)

    async def event_stream() -> AsyncIterator[dict]:
        collected: list[str] = []
        try:
            async for kind, payload in stream_turn(agent, prompt, history, deps=deps):
                if kind == "token":
                    collected.append(payload)
                    yield {"event": "token", "data": payload}
                elif kind == "done":
                    final = (payload or "".join(collected)).strip()
                    if not final:
                        # Some models occasionally emit a whitespace-only
                        # answer; one non-streamed retry usually recovers.
                        logger.warning("Empty assistant reply; retrying turn")
                        retry = await run_turn(agent, prompt, history, deps=deps)
                        final = strip_leaked_reasoning(retry.output or "")
                        if final:
                            yield {"event": "token", "data": final}
                    if not final:
                        final = "The model returned an empty response — please try again."
                    await db.add_message(session_id, "assistant", final)
                    _remember_turn_in_background(memory, prompt, final)
                    yield {"event": "done", "data": json.dumps({"content": final})}
        except Exception as error:
            logger.exception("Agent stream failed")
            answer = "".join(collected)
            if answer:
                await db.add_message(session_id, "assistant", answer)
            yield {"event": "error", "data": json.dumps({"detail": str(error)})}

    return EventSourceResponse(event_stream())


async def _load_history(db, session_id: uuid.UUID, exclude_last_user: str):
    """Build pydantic-ai message history from stored messages.

    The just-inserted user message is passed to the agent separately as the
    prompt, so it is dropped from the reconstructed history.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    rows = await db.list_messages(session_id)
    history: list = []
    for row in rows[:-1] if rows else []:
        if row.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=row.content)]))
        elif row.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=row.content)]))
    return history
