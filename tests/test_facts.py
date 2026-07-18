"""Tests for the chat fact-extraction pipeline: extract -> dedup -> save.

Both agents and the memory object are always fakes here — no real network or
LLM traffic. `_FakeAgent`/`_FailingAgent` mirror the pattern in
test_workflows.py.
"""

from __future__ import annotations

from cherryai_api.api import _extract_facts_in_background
from cherryai_api.facts import (
    ExtractedFacts,
    FactVerdict,
    extract_and_save_facts,
)


class _FakeAgent:
    """A stand-in for a pydantic-ai Agent: records prompts, returns fixed output."""

    def __init__(self, output) -> None:
        self.output = output
        self.prompts: list[str] = []

    async def run(self, prompt: str):
        self.prompts.append(prompt)
        return _FakeResult(self.output)


class _FakeResult:
    def __init__(self, output) -> None:
        self.output = output


class _FailingAgent:
    """A stand-in agent whose run() always raises, for failure-path tests."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.prompts: list[str] = []

    async def run(self, prompt: str):
        self.prompts.append(prompt)
        raise RuntimeError(self.message)


class _VerdictQueueAgent:
    """A stand-in judge agent returning a different verdict on each call."""

    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []

    async def run(self, prompt: str):
        self.prompts.append(prompt)
        return _FakeResult(FactVerdict(verdict=self._verdicts.pop(0)))


class _FakeMemory:
    """A stand-in for CogneeMemory: records remember_fact calls, fixed recall."""

    def __init__(self, recall_result: str = "No similar facts were found.") -> None:
        self.recall_result = recall_result
        self.recalled_queries: list[str] = []
        self.remembered_facts: list[str] = []

    async def recall_facts(self, query: str) -> str:
        self.recalled_queries.append(query)
        return self.recall_result

    async def remember_fact(self, fact: str) -> None:
        self.remembered_facts.append(fact)


# --- Statement message: facts extracted and saved -------------------------------


async def test_statement_message_extracts_and_saves_facts() -> None:
    extractor = _FakeAgent(ExtractedFacts(facts=["The user lives in Macon, GA."]))
    judge = _FakeAgent(FactVerdict(verdict="new"))
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "I live in Macon, GA.")

    assert extractor.prompts == ["I live in Macon, GA."]
    assert memory.remembered_facts == ["The user lives in Macon, GA."]


# --- Question / chit-chat: zero saves --------------------------------------------


async def test_question_message_yields_zero_saves() -> None:
    extractor = _FakeAgent(ExtractedFacts(facts=[]))
    judge = _FailingAgent("judge should never be called")
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "What's the weather like?")

    assert memory.remembered_facts == []
    assert judge.prompts == []


# --- Judge verdicts: new saves, duplicate skips, supersedes saves ---------------


async def test_judge_verdict_new_saves_the_fact() -> None:
    extractor = _FakeAgent(ExtractedFacts(facts=["The user owns a Jeep Gladiator."]))
    judge = _FakeAgent(FactVerdict(verdict="new"))
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "I drive a Jeep Gladiator.")

    assert memory.remembered_facts == ["The user owns a Jeep Gladiator."]


async def test_judge_verdict_duplicate_skips_the_fact() -> None:
    extractor = _FakeAgent(ExtractedFacts(facts=["The user owns a Jeep Gladiator."]))
    judge = _FakeAgent(FactVerdict(verdict="duplicate"))
    memory = _FakeMemory(recall_result="The user owns a Jeep Gladiator.")

    await extract_and_save_facts(extractor, judge, memory, "I drive a Jeep Gladiator.")

    assert memory.remembered_facts == []
    assert memory.recalled_queries == ["The user owns a Jeep Gladiator."]


async def test_judge_verdict_supersedes_saves_the_fact() -> None:
    extractor = _FakeAgent(ExtractedFacts(facts=["The user now lives in Atlanta, GA."]))
    judge = _FakeAgent(FactVerdict(verdict="supersedes"))
    memory = _FakeMemory(recall_result="The user lives in Macon, GA.")

    await extract_and_save_facts(extractor, judge, memory, "I moved to Atlanta.")

    assert memory.remembered_facts == ["The user now lives in Atlanta, GA."]


# --- Multiple facts: each judged and saved/skipped independently ----------------


async def test_multiple_facts_are_each_judged_independently() -> None:
    extractor = _FakeAgent(
        ExtractedFacts(facts=["The user lives in Macon, GA.", "The user owns a Jeep Gladiator."])
    )
    judge = _VerdictQueueAgent(["duplicate", "new"])
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "I live in Macon and drive a Jeep.")

    assert memory.remembered_facts == ["The user owns a Jeep Gladiator."]


# --- Failure isolation: extractor/judge/save errors never escape ---------------


async def test_extractor_raising_does_not_escape() -> None:
    extractor = _FailingAgent("ollama unreachable")
    judge = _FailingAgent("judge should never be called")
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "I live in Macon, GA.")  # must not raise

    assert memory.remembered_facts == []
    assert judge.prompts == []


async def test_one_fact_failing_does_not_block_the_rest() -> None:
    extractor = _FakeAgent(
        ExtractedFacts(facts=["Bad fact that breaks the judge.", "The user owns a Jeep Gladiator."])
    )

    class _JudgeFailsOnce:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, prompt: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("judge exploded")
            return _FakeResult(FactVerdict(verdict="new"))

    judge = _JudgeFailsOnce()
    memory = _FakeMemory()

    await extract_and_save_facts(extractor, judge, memory, "message")  # must not raise

    assert memory.remembered_facts == ["The user owns a Jeep Gladiator."]
    assert judge.calls == 2


async def test_fact_extraction_disabled_is_a_noop_when_agents_are_none() -> None:
    """api.py sets both agents to None when fact_extraction_enabled=False."""
    memory = _FakeMemory()

    _extract_facts_in_background(None, None, memory, "I live in Macon, GA.")

    # No background task was spawned (nothing to await), and memory untouched.
    assert memory.remembered_facts == []
    assert memory.recalled_queries == []


async def test_memory_save_failure_for_one_fact_does_not_block_the_rest() -> None:
    extractor = _FakeAgent(
        ExtractedFacts(facts=["Fact that fails to save.", "The user owns a Jeep Gladiator."])
    )
    judge = _FakeAgent(FactVerdict(verdict="new"))

    class _MemorySavesOnceThenFails(_FakeMemory):
        async def remember_fact(self, fact: str) -> None:
            if fact == "Fact that fails to save.":
                raise RuntimeError("cognee down")
            await super().remember_fact(fact)

    memory = _MemorySavesOnceThenFails()

    await extract_and_save_facts(extractor, judge, memory, "message")  # must not raise

    assert memory.remembered_facts == ["The user owns a Jeep Gladiator."]
