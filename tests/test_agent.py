"""Tests for agent streaming behavior.

Reproduces the "Stephanie Hatcher" bug: when the model narrates text
alongside a tool call ("Let me check the wiki:"), the streamed turn must
still run the tool, feed its result back to the model, and finish with the
model's real answer — not end early on the narration.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from cherryai_api.agent import stream_turn

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

    events = [event async for event in stream_turn(agent, "who is Stephanie Hatcher")]

    kind, final = events[-1]
    assert kind == "done"
    assert final == FINAL_ANSWER
    assert tool_calls == ["Stephanie Hatcher"]
