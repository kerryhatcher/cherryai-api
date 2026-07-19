"""Tests for agent streaming behavior.

Reproduces the "Stephanie Hatcher" bug: when the model narrates text
alongside a tool call ("Let me check the wiki:"), the streamed turn must
still run the tool, feed its result back to the model, and finish with the
model's real answer — not end early on the narration.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from cherryai_api.agent import AgentDeps, stream_turn


class _FakeMemory:
    """A stand-in for CogneeMemory: no real Cognee calls."""

    async def recall(self, query: str) -> str:  # pragma: no cover - unused by these tests
        return ""


def _deps() -> AgentDeps:
    return AgentDeps(memory=_FakeMemory(), user_id=uuid.uuid4())


NARRATION = "I don't have anything on that person. Let me check the wiki:"
WIKI_RESULT = "Stephanie Hatcher (/wiki/stephanie-hatcher) Wife of Kerry Hatcher."
FINAL_ANSWER = "Stephanie Hatcher is the wife of Kerry Hatcher."


def _saw_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


def _narrate_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _saw_tool_return(messages):
        return ModelResponse(parts=[TextPart(FINAL_ANSWER)])
    return ModelResponse(
        parts=[
            TextPart(NARRATION),
            ToolCallPart(tool_name="search_wiki", args={"query": "Stephanie Hatcher"}),
        ]
    )


async def _stream_narrate_then_answer(messages: list[ModelMessage], info: AgentInfo):
    if _saw_tool_return(messages):
        yield FINAL_ANSWER
        return
    yield NARRATION
    yield {1: DeltaToolCall(name="search_wiki", json_args='{"query": "Stephanie Hatcher"}')}


def _build_narrating_agent() -> tuple[Agent[None, str], list[str]]:
    """An agent whose model narrates before calling its only tool."""
    model = FunctionModel(
        function=_narrate_then_answer,
        stream_function=_stream_narrate_then_answer,
    )
    agent: Agent[None, str] = Agent(model)
    tool_calls: list[str] = []

    @agent.tool_plain
    async def search_wiki(query: str) -> str:
        tool_calls.append(query)
        return WIKI_RESULT

    return agent, tool_calls


async def test_stream_turn_answers_from_tool_results_despite_narration():
    agent, tool_calls = _build_narrating_agent()

    events = [event async for event in stream_turn(agent, "who is Stephanie Hatcher", deps=_deps())]

    kind, final = events[-1]
    assert kind == "done"
    assert final == FINAL_ANSWER
    assert tool_calls == ["Stephanie Hatcher"]


def _build_echo_agent(reply: str) -> Agent[None, str]:
    """An agent whose model always answers with ``reply`` verbatim."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(reply)])

    async def stream_respond(messages: list[ModelMessage], info: AgentInfo):
        yield reply

    return Agent(FunctionModel(function=respond, stream_function=stream_respond))


async def _final_payload(agent: Agent[None, str]) -> str:
    events = [event async for event in stream_turn(agent, "who is Stephanie Hatcher", deps=_deps())]
    kind, final = events[-1]
    assert kind == "done"
    return final


async def test_stream_turn_strips_leaked_thought_header():
    agent = _build_echo_agent("thought\nBased on the wiki, Stephanie is Kerry's wife.")
    assert await _final_payload(agent) == "Based on the wiki, Stephanie is Kerry's wife."


async def test_stream_turn_strips_leaked_think_block():
    agent = _build_echo_agent(
        "<think>The user wants the wiki entry.</think>\nStephanie is Kerry's wife."
    )
    assert await _final_payload(agent) == "Stephanie is Kerry's wife."


async def test_stream_turn_keeps_ordinary_answers_untouched():
    agent = _build_echo_agent("Thoughtful gardens need thought and care.")
    assert await _final_payload(agent) == "Thoughtful gardens need thought and care."


class _DatabaseStub:
    """A stand-in for Database: only the `.pool` attribute is read."""

    def __init__(self, pool) -> None:
        self.pool = pool


def _respond_search_wiki(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _saw_tool_return(messages):
        return ModelResponse(parts=[TextPart("done")])
    return ModelResponse(parts=[ToolCallPart(tool_name="search_wiki", args={"query": "anything"})])


@pytest.mark.asyncio
async def test_search_wiki_tool_scopes_to_deps_user(pool, make_user, monkeypatch):
    """The wiki tool must pass the deps user id into search_entries."""
    from cherryai_api import agent as agent_mod
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    seen: dict = {}

    async def fake_search(pool_arg, owner_id, query):
        seen["owner_id"] = owner_id
        return []

    monkeypatch.setattr(agent_mod, "search_entries", fake_search)

    user = await make_user("ztest-search-wiki-deps@example.com")
    agent = build_agent(
        Settings(ollama_api_key="x"),
        database=_DatabaseStub(pool),
    )
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    model = FunctionModel(function=_respond_search_wiki)
    await agent.run("who is in the wiki?", deps=deps, model=model)

    assert seen["owner_id"] == deps.user_id
