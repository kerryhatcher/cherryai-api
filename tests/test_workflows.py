"""Tests for feedback AI workflows: triage marker section, job lock, preflight.

Model calls are always mocked (fake agents recording prompts / returning fixed
output) — no real Ollama traffic. DB-backed tests use the ``pool`` fixture (dev
Postgres) and unique ``Ztest``-prefixed titles, same as test_feedback.py.
"""

from __future__ import annotations

import uuid

import pytest

import cherryai_api.workflows as workflows
from cherryai_api.feedback import FeedbackCreate, create_entry, get_entry
from cherryai_api.settings import Settings
from cherryai_api.workflows import TriageResult, WorkflowRuntime


def _unique_title(label: str) -> str:
    """A 'Ztest ...'-prefixed title that lands under the test namespace."""
    return f"Ztest {uuid.uuid4().hex[:8]} {label}"


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


async def _noop_preflight(settings: Settings, model_name: str) -> None:
    """Stand-in for _preflight_model that always succeeds."""


def _make_runtime(
    *,
    triage_output: TriageResult | None = None,
    investigate_output: str = "investigation markdown",
    plan_output: str = "plan markdown",
) -> WorkflowRuntime:
    settings = Settings(openrouter_api_key="x")
    triage = triage_output or TriageResult(type="bug", priority="medium", tags=[], questions=[])
    return WorkflowRuntime(
        settings=settings,
        triage_agent=_FakeAgent(triage),
        investigate_agent=_FakeAgent(investigate_output),
        plan_agent=_FakeAgent(plan_output),
    )


async def _drain_background_tasks(runtime: WorkflowRuntime) -> None:
    """Await every task in runtime.background_tasks, including ones spawned

    while awaiting (e.g. fire_and_forget_triage's wrapper spawning the job
    task itself only once it runs).
    """
    while runtime.background_tasks:
        for task in list(runtime.background_tasks):
            await task


# --- Triage marker section: pure functions ------------------------------------


def test_strip_triage_section_removes_marker_keeps_human_text() -> None:
    body = (
        "Human text.\n\n"
        f"{workflows.TRIAGE_MARKER_START}\n## Questions for the reporter\n- Q1?\n"
        f"{workflows.TRIAGE_MARKER_END}"
    )
    assert workflows._strip_triage_section(body) == "Human text."


def test_strip_triage_section_is_noop_when_absent() -> None:
    assert workflows._strip_triage_section("Just human text.") == "Just human text."


def test_apply_triage_section_appends_questions() -> None:
    result = workflows._apply_triage_section("Human text.", ["Q1?", "Q2?"])
    assert result.count(workflows.TRIAGE_MARKER_START) == 1
    assert "Q1?" in result
    assert "Q2?" in result
    assert result.startswith("Human text.")


def test_apply_triage_section_no_questions_yields_bare_body() -> None:
    result = workflows._apply_triage_section("Human text.", [])
    assert result == "Human text."
    assert workflows.TRIAGE_MARKER_START not in result


def test_apply_then_strip_round_trips_to_original_body() -> None:
    applied = workflows._apply_triage_section("Human text.", ["Q1?"])
    assert workflows._strip_triage_section(applied) == "Human text."


def test_rerun_replaces_not_duplicates_section() -> None:
    first = workflows._apply_triage_section("Human text.", ["Old question?"])
    second = workflows._apply_triage_section(workflows._strip_triage_section(first), ["New?"])
    assert second.count(workflows.TRIAGE_MARKER_START) == 1
    assert "Old question?" not in second
    assert "New?" in second
    assert second.startswith("Human text.")


def test_split_triage_section_returns_inline_reporter_replies() -> None:
    body = (
        "Human text.\n\n"
        f"{workflows.TRIAGE_MARKER_START}\n## Questions for the reporter\n"
        "- Q1?\nMy answer to Q1.\n- Q2?\n"
        f"{workflows.TRIAGE_MARKER_END}"
    )
    rest, content = workflows._split_section(
        body, workflows._TRIAGE_SECTION_RE, workflows._TRIAGE_HEADER
    )
    assert rest == "Human text."
    assert "My answer to Q1." in content
    assert "## Questions for the reporter" not in content


def test_has_reporter_content_false_for_pristine_questions() -> None:
    assert not workflows._has_reporter_content("- Q1?\n- Q2?")
    assert not workflows._has_reporter_content("")


def test_has_reporter_content_true_for_inline_answers() -> None:
    assert workflows._has_reporter_content("- Q1?\nMy answer.\n- Q2?")


def test_apply_answered_section_round_trips() -> None:
    applied = workflows._apply_answered_section("Human text.", "- Q1? A: yes")
    assert applied.startswith("Human text.")
    assert workflows.ANSWERED_MARKER_START in applied
    rest, content = workflows._split_section(
        applied, workflows._ANSWERED_SECTION_RE, workflows._ANSWERED_HEADER
    )
    assert rest == "Human text."
    assert content == "- Q1? A: yes"


def test_apply_answered_section_empty_is_noop() -> None:
    assert workflows._apply_answered_section("Human text.", "") == "Human text."


# --- Model preflight -----------------------------------------------------------


async def test_preflight_model_missing_raises(monkeypatch) -> None:
    async def fake_fetch(settings: Settings) -> set[str]:
        return {"some-other-model"}

    monkeypatch.setattr(workflows, "_fetch_available_models", fake_fetch)
    settings = Settings(openrouter_api_key="x")
    with pytest.raises(RuntimeError, match="not available"):
        await workflows._preflight_model(settings, "gpt-oss:20b")


async def test_preflight_model_present_does_not_raise(monkeypatch) -> None:
    async def fake_fetch(settings: Settings) -> set[str]:
        return {"gpt-oss:20b", "kimi-k2.7-code"}

    monkeypatch.setattr(workflows, "_fetch_available_models", fake_fetch)
    settings = Settings(openrouter_api_key="x")
    await workflows._preflight_model(settings, "gpt-oss:20b")  # must not raise


# --- Job lock: race-safe claim, success, failure, startup cleanup -------------


async def test_start_job_unknown_id_returns_none_and_spawns_nothing(pool) -> None:
    runtime = _make_runtime()
    job_id = await workflows.start_job(pool, runtime, 2_147_483_647, "triage")
    assert job_id is None
    assert not runtime.background_tasks


async def test_start_job_second_call_is_race_safe_409(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(pool, FeedbackCreate(title=_unique_title("Race"), type="bug"))
    runtime = _make_runtime()

    first = await workflows.start_job(pool, runtime, entry.id, "triage")
    second = await workflows.start_job(pool, runtime, entry.id, "triage")

    assert first is not None
    assert second is None
    await _drain_background_tasks(runtime)


async def test_run_job_success_clears_job_fields_and_writes_back(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Investigate Me"), type="bug")
    )
    runtime = _make_runtime(investigate_output="## Root cause\nSomething broke.")

    job_id = await workflows.start_job(pool, runtime, entry.id, "investigate")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.job_stage is None
    assert updated.job_status is None
    assert updated.job_id is None
    assert updated.job_error is None
    assert updated.investigation == "## Root cause\nSomething broke."
    assert job_id is not None


async def test_run_job_failure_persists_job_error_and_stage(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(pool, FeedbackCreate(title=_unique_title("Fails"), type="bug"))
    runtime = _make_runtime()
    runtime.plan_agent = _FailingAgent("boom")

    job_id = await workflows.start_job(pool, runtime, entry.id, "plan")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.job_status == "failed"
    assert "boom" in updated.job_error
    assert updated.job_stage == "plan"
    assert updated.job_id == job_id


async def test_run_job_preflight_failure_fails_fast_with_clear_error(pool, monkeypatch) -> None:
    async def no_models(settings: Settings) -> set[str]:
        return set()

    monkeypatch.setattr(workflows, "_fetch_available_models", no_models)
    entry = await create_entry(pool, FeedbackCreate(title=_unique_title("Preflight"), type="bug"))
    runtime = _make_runtime()

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.job_status == "failed"
    assert "not available" in updated.job_error
    # The fake agent must never have been called — preflight failed first.
    assert runtime.triage_agent.prompts == []


async def test_ensure_workflow_columns_clears_stale_running(pool) -> None:
    entry = await create_entry(pool, FeedbackCreate(title=_unique_title("Stale"), type="bug"))
    await pool.execute(
        "UPDATE feedback_entries SET job_stage = 'investigate', job_status = 'running', "
        "job_id = $2 WHERE id = $1",
        entry.id,
        uuid.uuid4(),
    )

    await workflows.ensure_workflow_columns(pool)

    updated = await get_entry(pool, entry.id)
    assert updated.job_status == "failed"
    assert updated.job_error == workflows._STALE_JOB_ERROR
    assert updated.job_stage == "investigate"  # left in place; only status/error changed


async def test_ensure_workflow_columns_is_idempotent(pool) -> None:
    await workflows.ensure_workflow_columns(pool)
    await workflows.ensure_workflow_columns(pool)  # must not raise


# --- Triage write-back: full re-evaluate, prompt sees split sections -----------


async def test_triage_prompt_strips_markers_but_shows_prior_questions(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    body = (
        f"Human text.\n\n{workflows.TRIAGE_MARKER_START}\nOld question?\n"
        f"{workflows.TRIAGE_MARKER_END}"
    )
    entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Prompt Body"), type="bug", body=body)
    )
    runtime = _make_runtime()

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    prompt = runtime.triage_agent.prompts[0]
    assert workflows.TRIAGE_MARKER_START not in prompt
    assert "Old question?" in prompt  # shown so the agent can spot inline replies
    assert "Human text." in prompt


async def test_rerun_triage_replaces_marker_and_reevaluates_fields(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Rerun Triage"), type="bug", body="Human text.")
    )
    runtime = _make_runtime(
        triage_output=TriageResult(type="bug", priority="low", tags=["a"], questions=["First?"])
    )
    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    runtime.triage_agent = _FakeAgent(
        TriageResult(type="feature", priority="high", tags=["b"], questions=["Second?"])
    )
    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.body.count(workflows.TRIAGE_MARKER_START) == 1
    assert "First?" not in updated.body
    assert "Second?" in updated.body
    assert updated.body.startswith("Human text.")
    assert updated.type == "feature"
    assert updated.priority == "high"
    assert updated.tags == ["b"]


async def test_triage_with_no_questions_removes_marker_section(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("No Questions"), type="bug", body="Human text.")
    )
    runtime = _make_runtime(
        triage_output=TriageResult(type="bug", priority="low", tags=[], questions=["Q?"])
    )
    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    runtime.triage_agent = _FakeAgent(
        TriageResult(type="bug", priority="low", tags=[], questions=[])
    )
    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert workflows.TRIAGE_MARKER_START not in updated.body
    assert updated.body == "Human text."


def _body_with_inline_answers() -> str:
    return (
        "Human text.\n\n"
        f"{workflows.TRIAGE_MARKER_START}\n## Questions for the reporter\n"
        "- Which iOS version?\niOS 19.2 on an iPhone 15 Pro.\n- Which browser?\n"
        f"{workflows.TRIAGE_MARKER_END}"
    )


async def test_triage_prompt_includes_inline_reporter_replies(pool, monkeypatch) -> None:
    """The agent must SEE inline answers — the pre-fix code stripped them away."""
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Sees Answers"), type="bug", body=_body_with_inline_answers()
        ),
    )
    runtime = _make_runtime()

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    prompt = runtime.triage_agent.prompts[0]
    assert "iOS 19.2 on an iPhone 15 Pro." in prompt


async def test_rerun_triage_moves_answers_into_answered_section(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Answers Kept"), type="bug", body=_body_with_inline_answers()
        ),
    )
    runtime = _make_runtime(
        triage_output=TriageResult(
            type="bug",
            priority="high",
            tags=[],
            questions=["Does it happen on wifi only?"],
            answered=["Which iOS version? — iOS 19.2 on an iPhone 15 Pro."],
        )
    )

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.body.startswith("Human text.")
    assert updated.body.count(workflows.ANSWERED_MARKER_START) == 1
    assert "iOS 19.2 on an iPhone 15 Pro." in updated.body
    assert "Does it happen on wifi only?" in updated.body
    assert "- Which iOS version?\niOS 19.2" not in updated.body  # old Q&A block replaced


async def test_rerun_triage_preserves_raw_replies_when_agent_reports_none(
    pool, monkeypatch
) -> None:
    """Safety net: reporter text is never dropped even if the agent returns no answered items."""
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(
        pool,
        FeedbackCreate(
            title=_unique_title("Raw Preserved"), type="bug", body=_body_with_inline_answers()
        ),
    )
    runtime = _make_runtime(
        triage_output=TriageResult(type="bug", priority="low", tags=[], questions=["New?"])
    )

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert "iOS 19.2 on an iPhone 15 Pro." in updated.body
    assert updated.body.count(workflows.ANSWERED_MARKER_START) == 1
    assert "New?" in updated.body


async def test_rerun_triage_keeps_prior_answered_section_on_fallback(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    body = (
        "Human text.\n\n"
        f"{workflows.ANSWERED_MARKER_START}\n## Answered by the reporter\n"
        "- Old answer stays.\n"
        f"{workflows.ANSWERED_MARKER_END}"
    )
    entry = await create_entry(
        pool, FeedbackCreate(title=_unique_title("Answered Kept"), type="bug", body=body)
    )
    runtime = _make_runtime(
        triage_output=TriageResult(type="bug", priority="low", tags=[], questions=[])
    )

    await workflows.start_job(pool, runtime, entry.id, "triage")
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert "Old answer stays." in updated.body
    assert updated.body.count(workflows.ANSWERED_MARKER_START) == 1


# --- Auto-triage fire-and-forget ----------------------------------------------


async def test_fire_and_forget_triage_starts_and_completes_a_job(pool, monkeypatch) -> None:
    monkeypatch.setattr(workflows, "_preflight_model", _noop_preflight)
    entry = await create_entry(pool, FeedbackCreate(title=_unique_title("Auto Triage"), type="bug"))
    runtime = _make_runtime(
        triage_output=TriageResult(type="bug", priority="medium", tags=[], questions=[])
    )

    workflows.fire_and_forget_triage(runtime, pool, entry.id)
    await _drain_background_tasks(runtime)

    updated = await get_entry(pool, entry.id)
    assert updated.job_status is None
    assert updated.type == "bug"
    assert updated.priority == "medium"


async def test_fire_and_forget_triage_swallows_start_failures(pool, monkeypatch) -> None:
    """A DB error (or any exception) during start must be logged, not raised."""

    async def broken_start_job(*args, **kwargs):
        raise RuntimeError("db hiccup")

    monkeypatch.setattr(workflows, "start_job", broken_start_job)
    runtime = _make_runtime()

    workflows.fire_and_forget_triage(runtime, pool, 1)  # must not raise
    await _drain_background_tasks(runtime)
