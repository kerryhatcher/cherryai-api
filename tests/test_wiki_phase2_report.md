# Phase 2 Wiki Family Scoping — Implementation Report

## Summary

Updated the wiki module to use the authz capability model for family scoping and audience filtering. All 404 tests pass.

## Changes

### `src/cherryai_api/wiki.py`
- Added `audience` and `family_id` to `WikiEntry` model and `_ENTRY_COLUMNS`
- Added `audience` to `WikiCreate` and `WikiUpdate` models
- Added `MoveBody` model and `move_entry` function
- Changed all query functions from `(pool, owner_id, ...)` to `(pool, capability, ...)`
- All queries now use `scoped_connection` + `wiki_visibility_sql`/`scope_sql`
- All routes now use `require_permission("wiki", "view")` or `require_permission("wiki", "edit")`
- Added `POST /api/wiki/{slug}/move` endpoint for personal↔family moves
- Fixed parameter numbering in `scope_sql`/`wiki_visibility_sql` calls (start values)
- Fixed `rename_folder` to use separate scope clauses for matched/updated CTEs

### `tests/test_wiki.py`
- Added `_cap()` helper to build Capability from user_id
- Added `_make_family()` helper to create families in DB
- Updated all DB-backed tests to pass Capability instead of owner_id
- Added 7 new tests:
  - `test_family_scoped_list` — family entries only visible in family context
  - `test_family_scoped_get` — cross-context get returns None
  - `test_cross_family_isolation` — family A entries invisible in family B
  - `test_child_does_not_see_adult_audience_entries` — audience filtering
  - `test_audience_filtering_only_applies_in_family_context` — personal context ignores audience
  - `test_move_personal_to_family` — move endpoint
  - `test_move_family_to_personal` — move endpoint
  - `test_move_slug_collision_raises` — slug collision on move

### `tests/test_agent.py`
- Updated `test_search_wiki_tool_scopes_to_deps_user` to expect Capability instead of UUID

## Test Results
- 404 passed, 0 failed, 14 warnings (all pre-existing cognee/pydantic deprecation warnings)

## Residual Risks
- RLS not yet enabled on wiki_entries (Task 8) — requires adding `"wiki_entries"` to `RLS_TABLES` and running RLS migration
- Agent `search_wiki` tool already passes Capability (verified working)
- Workflow runtime's `search_wiki` uses `search_all_entries` (unscoped) — correct for workspace-level search
