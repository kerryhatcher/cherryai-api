# Agent Phase 2 Report — search_wiki tool updated

## Changes made

### `src/cherryai_api/agent.py`
- Added `from cherryai_api.authz import Capability` import
- Added `capability: Capability | None = None` field to `AgentDeps` dataclass
- Updated `search_wiki` tool to use `ctx.deps.capability` when available, falling back to constructing a personal `Capability(user_id=ctx.deps.user_id, family_id=None, role=None, scopes=frozenset())`
- Passes the capability to `search_entries(database.pool, cap, query)` instead of `search_entries(database.pool, ctx.deps.user_id, query)`

### `src/cherryai_api/api.py`
- Added `from cherryai_api.authz import Capability` import
- Updated `AgentDeps` construction in the chat endpoint to include a personal `Capability` (user_id, family_id=None, role=None, scopes=frozenset())

### `src/cherryai_api/workflows.py`
- No change needed — uses `search_all_entries` (unscoped, intentional for background workflows)

## Verification
- `uv run ruff check src/cherryai_api/agent.py src/cherryai_api/workflows.py src/cherryai_api/api.py` — all checks passed
- No staged files

## Residual risks
- `tests/test_agent.py::test_search_wiki_tool_scopes_to_deps_user` patches the old `(pool, owner_id, query)` signature and will need updating to match the new `(pool, capability, query)` signature
- The `capability` field on `AgentDeps` is optional (`None` by default) for backward compatibility with any code that constructs `AgentDeps` without it

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Updated search_wiki tool in agent.py to use Capability-based search_entries. Added capability field to AgentDeps. Updated api.py to construct personal Capability. Ruff clean. No scope widening."
    }
  ],
  "changedFiles": [
    "src/cherryai_api/agent.py",
    "src/cherryai_api/api.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "uv run ruff check src/cherryai_api/agent.py src/cherryai_api/workflows.py src/cherryai_api/api.py",
      "result": "passed",
      "summary": "All checks passed"
    }
  ],
  "validationOutput": [
    "ruff: All checks passed",
    "workflows.py: no change needed (uses search_all_entries, unscoped)"
  ],
  "residualRisks": [
    "tests/test_agent.py::test_search_wiki_tool_scopes_to_deps_user patches old (pool, owner_id, query) signature and needs updating"
  ],
  "noStagedFiles": true,
  "diffSummary": "agent.py: added Capability import, capability field to AgentDeps, updated search_wiki tool. api.py: added Capability import, passes personal Capability to AgentDeps.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "The test file test_agent.py has a test that patches the old search_entries signature - it will need updating by the test-writing subagent."
}
```
