"""Per-call-site model/provider configuration, overridable from Postgres.

The ``model_configs`` table stores one row per ``call_site`` (e.g. ``chat``,
``workflow_triage``, ``_default``). Each row can override some or all of
provider, base_url, api_key, and model_name.

Resolution order (innermost wins):
  1. ``_default`` row in the DB
  2. call-site-specific row in the DB
  3. env-var defaults from ``Settings``

Any field left blank in a more-specific row inherits from the next fallback.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from cherryai_api.settings import Settings

# ── Well-known call sites ────────────────────────────────────────────────
# ``_default`` is the fallback row used when no specific call-site row exists.
CALL_SITE_DEFAULT = "_default"
CALL_SITE_CHAT = "chat"
CALL_SITE_WORKFLOW_TRIAGE = "workflow_triage"
CALL_SITE_WORKFLOW_INVESTIGATE = "workflow_investigate"
CALL_SITE_WORKFLOW_PLAN = "workflow_plan"
CALL_SITE_FACT_EXTRACTION = "fact_extraction"
CALL_SITE_COGNEE_LLM = "cognee_llm"

ALL_CALL_SITES = (
    CALL_SITE_DEFAULT,
    CALL_SITE_CHAT,
    CALL_SITE_WORKFLOW_TRIAGE,
    CALL_SITE_WORKFLOW_INVESTIGATE,
    CALL_SITE_WORKFLOW_PLAN,
    CALL_SITE_FACT_EXTRACTION,
    CALL_SITE_COGNEE_LLM,
)

# ── DDL ───────────────────────────────────────────────────────────────────

CREATE_MODEL_CONFIGS_TABLE = """
CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY,
    call_site TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_model_configs_call_site
    ON model_configs (call_site);
"""

# ── Row model ─────────────────────────────────────────────────────────────


@dataclass
class ModelConfigRow:
    """A single row from the ``model_configs`` table."""

    id: uuid.UUID
    call_site: str
    provider: str
    base_url: str
    api_key: str
    model_name: str
    created_at: datetime
    updated_at: datetime


def _row_from_record(row: asyncpg.Record) -> ModelConfigRow:
    return ModelConfigRow(**dict(row))


# ── Resolved config ───────────────────────────────────────────────────────


@dataclass
class ResolvedModelConfig:
    """Fully-resolved model configuration ready to pass to a model constructor.

    Every field is guaranteed non-empty (falls back through _default → env).
    """

    call_site: str
    provider: str
    base_url: str
    api_key: str
    model_name: str


# ── Data access ───────────────────────────────────────────────────────────


async def list_configs(pool: asyncpg.Pool) -> list[ModelConfigRow]:
    """Return every row in model_configs, newest first."""
    rows = await pool.fetch(
        "SELECT id, call_site, provider, base_url, api_key, model_name, "
        "created_at, updated_at "
        "FROM model_configs ORDER BY call_site"
    )
    return [_row_from_record(r) for r in rows]


async def get_config(pool: asyncpg.Pool, call_site: str) -> ModelConfigRow | None:
    """Return the row for *call_site*, or None."""
    row = await pool.fetchrow(
        "SELECT id, call_site, provider, base_url, api_key, model_name, "
        "created_at, updated_at "
        "FROM model_configs WHERE call_site = $1",
        call_site,
    )
    return _row_from_record(row) if row else None


async def upsert_config(
    pool: asyncpg.Pool,
    call_site: str,
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
    model_name: str = "",
) -> ModelConfigRow:
    """Insert or replace the row for *call_site*.

    Empty-string fields are stored as-is (the resolver treats them as
    "inherit from fallback"). Returns the new row.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO model_configs (id, call_site, provider, base_url, api_key, model_name)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (call_site) DO UPDATE SET
            provider = EXCLUDED.provider,
            base_url = EXCLUDED.base_url,
            api_key = EXCLUDED.api_key,
            model_name = EXCLUDED.model_name,
            updated_at = now()
        RETURNING id, call_site, provider, base_url, api_key, model_name,
                  created_at, updated_at
        """,
        uuid.uuid4(),
        call_site,
        provider,
        base_url,
        api_key,
        model_name,
    )
    return _row_from_record(row)


async def delete_config(pool: asyncpg.Pool, call_site: str) -> bool:
    """Delete the row for *call_site*. Returns True if a row was removed."""
    result = await pool.execute("DELETE FROM model_configs WHERE call_site = $1", call_site)
    return result.endswith("1")


# ── Resolver ──────────────────────────────────────────────────────────────


def _env_defaults(settings: Settings) -> dict[str, ResolvedModelConfig]:
    """Return the env-var-based defaults for every call site.

    The returned configs serve as the ultimate fallback: any field left blank
    in both the DB site-specific row AND the DB _default row will use the
    env value.
    """
    return {
        CALL_SITE_DEFAULT: ResolvedModelConfig(
            call_site=CALL_SITE_DEFAULT,
            provider="ollama",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model_name=settings.chat_model,
        ),
        CALL_SITE_CHAT: ResolvedModelConfig(
            call_site=CALL_SITE_CHAT,
            provider="ollama",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model_name=settings.chat_model,
        ),
        CALL_SITE_WORKFLOW_TRIAGE: ResolvedModelConfig(
            call_site=CALL_SITE_WORKFLOW_TRIAGE,
            provider="ollama",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model_name=settings.workflow_triage_model,
        ),
        CALL_SITE_WORKFLOW_INVESTIGATE: ResolvedModelConfig(
            call_site=CALL_SITE_WORKFLOW_INVESTIGATE,
            provider="ollama",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model_name=settings.workflow_investigate_model,
        ),
        CALL_SITE_WORKFLOW_PLAN: ResolvedModelConfig(
            call_site=CALL_SITE_WORKFLOW_PLAN,
            provider="ollama",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model_name=settings.workflow_plan_model,
        ),
        CALL_SITE_FACT_EXTRACTION: ResolvedModelConfig(
            call_site=CALL_SITE_FACT_EXTRACTION,
            provider="ollama",
            base_url=settings.ollama_local_base_url,
            api_key="",
            model_name=settings.fact_extraction_model,
        ),
        CALL_SITE_COGNEE_LLM: ResolvedModelConfig(
            call_site=CALL_SITE_COGNEE_LLM,
            provider="ollama",
            base_url=settings.cognee_llm_endpoint,
            api_key=settings.cognee_llm_api_key,
            model_name=settings.cognee_llm_model,
        ),
    }


def _merge(
    env: ResolvedModelConfig,
    default_row: ModelConfigRow | None,
    site_row: ModelConfigRow | None,
) -> ResolvedModelConfig:
    """Merge three layers, each non-empty field overriding the one below.

    site_row (most specific) > default_row > env (env var fallback).
    """

    def _first(*values: str) -> str:
        for v in values:
            if v:
                return v
        return ""

    return ResolvedModelConfig(
        call_site=env.call_site,
        provider=_first(
            site_row.provider if site_row else "",
            default_row.provider if default_row else "",
            env.provider,
        ),
        base_url=_first(
            site_row.base_url if site_row else "",
            default_row.base_url if default_row else "",
            env.base_url,
        ),
        api_key=_first(
            site_row.api_key if site_row else "",
            default_row.api_key if default_row else "",
            env.api_key,
        ),
        model_name=_first(
            site_row.model_name if site_row else "",
            default_row.model_name if default_row else "",
            env.model_name,
        ),
    )


async def resolve_config(
    pool: asyncpg.Pool,
    call_site: str,
    settings: Settings | None = None,
) -> ResolvedModelConfig:
    """Resolve the model config for *call_site*, merging DB overrides over env vars.

    Resolution order (most-specific wins):
      1. A ``_default`` row in the DB  (fallback for every site)
      2. A call-site-specific row in the DB
      3. Environment variable defaults from ``Settings``

    Any field left blank at a given level inherits from the next fallback.
    This function is safe to call from both the lifespan (before the pool
    is fully set up — no DB rows exist, so env defaults are returned) and
    from runtime (DB rows may exist for some or all sites).

    If the pool is None or the query fails, env defaults are returned so
    the app never crashes from config resolution.
    """
    from cherryai_api.settings import get_settings

    settings = settings or get_settings()
    env_map = _env_defaults(settings)
    env = env_map.get(call_site) or env_map[CALL_SITE_DEFAULT]

    if pool is None:
        return env

    try:
        default_row = await get_config(pool, CALL_SITE_DEFAULT)
        site_row = await get_config(pool, call_site) if call_site != CALL_SITE_DEFAULT else None
    except Exception:
        # Cannot reach the DB — return env defaults so the app stays up.
        return env

    return _merge(env, default_row, site_row)
