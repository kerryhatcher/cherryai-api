"""Tests for CogneeMemory.remember_fact/recall_facts: args shape, first-run fallback.

`cognee.remember`/`cognee.recall` are monkeypatched directly on the imported
`cognee` module — no real network or LLM traffic, no real graph store.
"""

from __future__ import annotations

import cherryai_api.memory as memory_module
from cherryai_api.memory import (
    CogneeMemory,
    DatasetNotFoundError,
    PermissionDeniedError,
)


def _memory() -> CogneeMemory:
    return CogneeMemory("test_dataset", "test-session", top_k=3)


async def test_remember_fact_saves_with_no_session_id_and_user_facts_node_set(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    async def fake_remember(data, dataset_name=None, **kwargs):
        calls.append({"data": data, "dataset_name": dataset_name, **kwargs})

    monkeypatch.setattr(memory_module.cognee, "remember", fake_remember)
    memory = _memory()

    await memory.remember_fact("The user lives in Macon, GA.")

    assert len(calls) == 1
    call = calls[0]
    assert call["data"] == "The user lives in Macon, GA."
    assert call["dataset_name"] == "test_dataset"
    assert call["node_set"] == ["user_facts"]
    assert "session_id" not in call


async def test_recall_facts_queries_graph_scope_only_no_session(monkeypatch) -> None:
    captured: dict = {}

    async def fake_recall(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return []

    monkeypatch.setattr(memory_module.cognee, "recall", fake_recall)
    memory = _memory()

    result = await memory.recall_facts("Where does the user live?")

    assert captured["query"] == "Where does the user live?"
    assert captured["datasets"] == ["test_dataset"]
    assert captured["scope"] == ["graph"]
    assert "session_id" not in captured
    assert result == "No similar facts were found."


async def test_recall_facts_formats_results_when_found(monkeypatch) -> None:
    class _Result:
        content = "The user lives in Macon, GA."

    async def fake_recall(query, **kwargs):
        return [_Result()]

    monkeypatch.setattr(memory_module.cognee, "recall", fake_recall)
    memory = _memory()

    result = await memory.recall_facts("Where does the user live?")

    assert result == "The user lives in Macon, GA."


async def test_recall_facts_tolerates_dataset_not_found_on_first_run(monkeypatch) -> None:
    async def fake_recall(query, **kwargs):
        raise DatasetNotFoundError()

    monkeypatch.setattr(memory_module.cognee, "recall", fake_recall)
    memory = _memory()

    result = await memory.recall_facts("Where does the user live?")

    assert result == "No similar facts were found."


async def test_recall_facts_tolerates_permission_denied_on_first_run(monkeypatch) -> None:
    async def fake_recall(query, **kwargs):
        raise PermissionDeniedError()

    monkeypatch.setattr(memory_module.cognee, "recall", fake_recall)
    memory = _memory()

    result = await memory.recall_facts("Where does the user live?")

    assert result == "No similar facts were found."


def test_build_memory_accepts_per_user_values():
    from cherryai_api.memory import build_memory

    m = build_memory("user-abc", "session-123")
    assert m.dataset == "user-abc"
    assert m.session_id == "session-123"


def test_build_memory_defaults_to_settings():
    from cherryai_api.memory import build_memory
    from cherryai_api.settings import get_settings

    m = build_memory()
    assert m.dataset == get_settings().cognee_dataset
