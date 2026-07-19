"""Automatic chat fact extraction: local Ollama model -> Cognee's permanent graph.

Wired from ``api.py``'s ``send_message`` as a fire-and-forget background task
that runs after the user's message is persisted (see ``_remember_turn_in_background``
there for the same pattern). Every stage is best-effort: an unreachable local
Ollama, a missing model, malformed structured output, or a Cognee error is
logged at warning level and the pipeline stops there — it must never affect
the chat reply. There are no retries.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from cherryai_api.memory import CogneeMemory
from cherryai_api.settings import Settings


class ExtractedFacts(BaseModel):
    """Structured output of the extractor agent."""

    facts: list[str] = Field(default_factory=list)


class FactVerdict(BaseModel):
    """Structured output of the dedup judge agent."""

    verdict: Literal["new", "duplicate", "supersedes"]


EXTRACT_SYSTEM_PROMPT = (
    "From ONE user chat message, extract only durable personal facts stated "
    "by the user: identity, location, possessions, preferences, and "
    "relationships. Rewrite each as a concise third-person statement, for "
    "example 'The user lives in Macon, GA.' Questions, requests, task "
    "instructions, opinions about the current task, and other ephemeral "
    "context are not facts — return an empty list for those."
)

JUDGE_SYSTEM_PROMPT = (
    "Decide whether a candidate fact about the user is new information, a "
    "duplicate of something already known, or supersedes (updates or "
    "contradicts) an existing fact. Given the candidate fact and any similar "
    "facts already recalled from memory, respond with exactly one verdict: "
    "'new' if nothing similar is known, 'duplicate' if it is already known "
    "with no new information, or 'supersedes' if it updates or contradicts a "
    "prior fact."
)


def _local_ollama_model(settings: Settings) -> OpenAIChatModel:
    """Build the OpenAI-compatible model pointed at the local Ollama instance."""
    return OpenAIChatModel(
        settings.fact_extraction_model,
        provider=OllamaProvider(base_url=settings.ollama_local_base_url),
    )


def build_extractor_agent(settings: Settings) -> Agent[None, ExtractedFacts]:
    """Build the fact-extraction agent: structured output, no tools."""
    return Agent(
        _local_ollama_model(settings),
        output_type=ExtractedFacts,
        instructions=EXTRACT_SYSTEM_PROMPT,
        # qwen3:8b occasionally emits reasoning text instead of the output
        # tool call; a couple of validation retries absorbs that flakiness.
        retries=3,
    )


def build_judge_agent(settings: Settings) -> Agent[None, FactVerdict]:
    """Build the dedup judge agent: structured output, no tools."""
    return Agent(
        _local_ollama_model(settings),
        output_type=FactVerdict,
        instructions=JUDGE_SYSTEM_PROMPT,
        # Same flakiness margin as the extractor agent.
        retries=3,
    )


def _judge_prompt(fact: str, recalled: str) -> str:
    return f"Candidate fact:\n{fact}\n\nSimilar facts already known:\n{recalled}"


async def _dedup_and_save(
    judge_agent: Agent[None, FactVerdict], memory: CogneeMemory, fact: str
) -> None:
    """Recall similar facts, judge the candidate, and save unless it's a duplicate.

    Every decision is logged at info level (fact text + verdict) for
    auditability, per the design spec.
    """
    recalled = await memory.recall_facts(fact)
    result = await judge_agent.run(_judge_prompt(fact, recalled))
    verdict = result.output.verdict
    logger.bind(fact=fact, verdict=verdict).info("Fact extraction decision")
    if verdict == "duplicate":
        return
    await memory.remember_fact(fact)


async def extract_and_save_facts(
    extractor_agent: Agent[None, ExtractedFacts],
    judge_agent: Agent[None, FactVerdict],
    memory: CogneeMemory,
    message: str,
) -> None:
    """Run the full extract -> dedup -> save pipeline for one chat message.

    Best-effort throughout: any exception (from either agent or from Cognee)
    is logged at warning level and swallowed, per fact, so one bad fact never
    stops the rest, and the whole pipeline never raises into its caller.
    """
    try:
        result = await extractor_agent.run(message)
    except Exception as error:
        logger.warning(f"Fact extraction failed: {error}")
        return
    for fact in result.output.facts:
        try:
            await _dedup_and_save(judge_agent, memory, fact)
        except Exception as error:
            logger.bind(fact=fact).warning(f"Fact dedup/save failed: {error}")
