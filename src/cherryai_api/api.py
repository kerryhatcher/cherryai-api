"""FastAPI HTTP surface for the CherryAI demo (no auth, by design)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cherryai_api.agent import build_agent, run_turn, stream_turn
from cherryai_api.db import build_database, make_session_title
from cherryai_api.facts import build_extractor_agent, build_judge_agent, extract_and_save_facts
from cherryai_api.feedback import router as feedback_router
from cherryai_api.logging_setup import setup_file_logging
from cherryai_api.memory import build_memory
from cherryai_api.settings import get_settings
from cherryai_api.telemetry import setup_telemetry
from cherryai_api.wiki import router as wiki_router
from cherryai_api.workflows import build_workflow_runtime
from cherryai_api.workflows import router as workflows_router


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the database pool and build the agent once per process."""
    settings = get_settings()
    setup_file_logging(settings.log_dir)
    setup_telemetry(app)
    database = build_database()
    await database.connect()
    memory = build_memory()
    workflows = build_workflow_runtime(settings, database, memory)
    agent = build_agent(settings, memory=memory, database=database, workflows=workflows)
    app.state.settings = settings
    app.state.db = database
    app.state.memory = memory
    app.state.workflows = workflows
    app.state.agent = agent
    if settings.fact_extraction_enabled:
        app.state.fact_extractor_agent = build_extractor_agent(settings)
        app.state.fact_judge_agent = build_judge_agent(settings)
    else:
        app.state.fact_extractor_agent = None
        app.state.fact_judge_agent = None
    logger.info("CherryAI API started")
    try:
        yield
    finally:
        await database.close()
        logger.info("CherryAI API stopped")


app = FastAPI(title="CherryAI API", lifespan=lifespan)
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


async def _neo4j_reachable() -> bool:
    """Verify the Neo4j server answers before reporting it healthy."""
    settings = get_settings()
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            await driver.verify_connectivity()
            return True
        finally:
            await driver.close()
    except Exception as error:
        logger.warning(f"Neo4j health check failed: {error}")
        return False


@app.get("/api/health")
async def health() -> dict:
    """Report liveness and dependency reachability."""
    db_ok = False
    with contextlib.suppress(Exception):
        db_ok = await app.state.db.ping()
    neo4j_ok = await _neo4j_reachable()
    status = "ok" if db_ok and neo4j_ok else "degraded"
    return {"status": status, "postgres": db_ok, "neo4j": neo4j_ok}


@app.get("/api/sessions")
async def list_sessions() -> list[dict]:
    sessions = await app.state.db.list_sessions()
    return [s.model_dump(mode="json") for s in sessions]


@app.post("/api/sessions", status_code=201)
async def create_session(body: CreateSessionRequest | None = None) -> dict:
    title = body.title if body and body.title else "New chat"
    session = await app.state.db.create_session(title)
    return session.model_dump(mode="json")


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: uuid.UUID) -> list[dict]:
    session = await app.state.db.get_session(session_id)
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
async def send_message(session_id: uuid.UUID, body: SendMessageRequest):
    """Persist the user message and stream the assistant reply as SSE."""
    db = app.state.db
    memory = app.state.memory
    agent = app.state.agent

    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    prompt = body.content.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Message content is empty")

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
            async for kind, payload in stream_turn(agent, prompt, history):
                if kind == "token":
                    collected.append(payload)
                    yield {"event": "token", "data": payload}
                elif kind == "done":
                    final = (payload or "".join(collected)).strip()
                    if not final:
                        # openrouter/free sometimes emits a whitespace-only
                        # answer; one non-streamed retry usually recovers.
                        logger.warning("Empty assistant reply; retrying turn")
                        retry = await run_turn(agent, prompt, history)
                        final = (retry.output or "").strip()
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
