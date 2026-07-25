# Agent Instructions — cherryai-api

This repository is an **independent git repository** with its own GitHub remote. The parent folder (`../`) contains only high-level planning and research; this repo holds the FastAPI backend implementation.

## Repository Structure

- **Parent folder** (`../`) — Planning repo (independent), holds project requirements and research. Never `git add` files from here into this repo's git history.
- **This folder** — Application code for the CherryAI FastAPI backend. All implementation belongs here.
- Other independent repos at `../cherryai-web/` — React SPA frontend (separate repo, separate git history).

When working in this folder, all git operations (status, commit, push) apply to **this** repo only. Verify with `git rev-parse --show-toplevel` before committing.

## Git Workflow

- **Commit regularly.** Commit after each completed task, story, or phase of work. Do not let changes pile up into one large commit.
- **Use Conventional Commits** for every commit message:

  ```
  <type>[optional scope]: <description>
  ```

  Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `build`, `ci`. Most commits will be `feat:` (new features), `fix:` (bug fixes), or `docs:` (documentation).

  - Description in present tense, imperative mood ("add", not "added"), under 72 characters.
  - Mark breaking changes with `!` after the type/scope or a `BREAKING CHANGE:` footer.

## Development Standards

### Python and Dependencies

- **Use `uv` exclusively.** Never call system `python` or `python3` directly. All commands must be prefixed with `uv run`.
- **Never use pyenv.** Do not rely on pyenv shims or `.python-version` interpreters; `python3` must only ever be invoked via `uv run`.
- **Managing packages:** Use `uv add <package>` to add dependencies and `uv add --dev <package>` for dev dependencies. Use `uv remove <package>` to remove.

### Code Quality

- **Always lint before committing.** Run `uv run ruff check` and fix issues with `uv run ruff check --fix`.
- **Run tests before committing.** Execute `uv run pytest` to ensure all tests pass.
- **Follow project conventions.** Refer to `src/cherryai_api/` for established patterns in modules like `api.py`, `agent.py`, `db.py`, etc.

### Testing

Write and maintain tests for:
- Database operations (session/message CRUD)
- Tool integrations (web_search, web_fetch, search_memory)
- Error handling and fallback logic

Tests live in `tests/` and use pytest + pytest-asyncio. Run them before every commit.

## Subagents and Model Economy

When delegating work to subagents, always pick the cheapest model that can do the task well — cheaper models are usually faster as well as cheaper, so this saves both money and time:

- **Haiku** — mechanical, low-judgment work: writing docs from a known outline, renames and file moves, boilerplate, simple lookups and summaries.
- **Sonnet** — standard implementation against a clear spec or contract: CRUD endpoints, tests, routine refactors.
- **Opus (or stronger)** — only high-judgment work: architecture, tricky integrations (e.g. Cognee/Pydantic AI wiring), debugging unknowns.

Match the model to the task, not the project's importance. Prefer running independent tasks as parallel subagents.

## Secrets and Environment

- **Never commit `.env` or `.env.local`** — these contain API keys and database credentials.
- Environment configuration is loaded via Pydantic Settings from `.env` at startup.
- Document required env var names in README and setup instructions; never commit actual values.

## Commits and Collaboration

- Each commit should be a self-contained unit of work (one feature, one fix, one refactor).
- Use clear, descriptive commit messages so the git log reads like a history of decisions.
- If a commit breaks something, it's easier to revert a small commit than a large one.
- Push to the remote after each logical group of commits (e.g., after completing a feature).

## Related Documentation

- **[README.md](./README.md)** — Setup, API endpoints, code structure, deployment notes
- **[CherryAI Planning Repo README](../README.md)** — Project requirements and architecture decisions
- **[cherryai-web README](../cherryai-web/README.md)** — React frontend that consumes this API
- **[Demo Design Spec](../docs/superpowers/specs/2026-07-18-cherryai-demo-design.md)** — Technical design and implementation plan

## Frontend Error Logging

The API accepts error reports from the frontend at `POST /api/log/error` and persists them to a Postgres table. This requires an authenticated, verified session — errors occurring before login are intentionally dropped rather than logged anonymously.

- **Endpoint:** `POST /api/log/error` — defined in `src/cherryai_api/api.py` (search for `log_error`); requires `current_verified_user` dependency
- **Storage:** `frontend_errors` table in Postgres
- **Model:** `src/cherryai_api/frontend_errors.py` — fields include `id`, `user_id`, `message`, `source`, `lineno`, `colno`, `stack`, `url`, `user_agent`, `client_ip`, `client_timestamp`, `context`, `created_at`
- **Migrations:** Alembic migrations `0004` and `0005` created the table and added the schema

### Reading the logs

Use the CLI commands to list and inspect frontend errors:

```bash
# List recent errors, newest first (default 50 rows)
cherryai errors list

# List with a custom limit
cherryai errors list --limit 100

# Show one error in full (accepts any unique prefix of the id)
cherryai errors show 12345678
cherryai errors show abc123  # if abc123 uniquely identifies one row
```

### Testing the logger

Log in to the app (the endpoint requires authentication), then open the browser console and throw a test error:

```js
throw new Error('test log from console')
```

Then list the errors to verify it was recorded:

```bash
cherryai errors list
```

### Retention and cleanup

Retention/cleanup for the `frontend_errors` table is an **unresolved open question** — the table grows without bound. However, requiring authentication means the write rate is now bounded by the verified user count rather than by the open internet. A future cleanup policy should consider: log retention period (e.g., 30/60/90 days), archival strategy, and automated deletion of old rows.

## GitHub Actions

- **Always pin actions to full commit SHAs, never git tags** (tags are mutable and can be repointed at malicious code; SHAs are immutable). Keep the human-readable version in a trailing comment:

  ```yaml
  - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5
  ```

  To resolve a tag to its SHA: `gh api repos/<owner>/<repo>/git/ref/tags/<tag> --jq '.object.sha'` (if `.object.type` is `tag` rather than `commit`, dereference the annotated tag object first).
