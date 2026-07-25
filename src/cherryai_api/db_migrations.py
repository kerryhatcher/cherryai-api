"""Run Alembic migrations to head at application startup.

The DigitalOcean App Platform PRE_DEPLOY job that used to run
``alembic upgrade head`` was proven to silently no-op on every deployment
(production stayed on revision 0003 across several deploys while the image
already contained 0004/0005). Rather than trust a job runner that reports
success without doing anything, the API migrates its own schema on boot,
before it opens any connection pool or builds the agent.

This module uses Alembic's Python API directly (``alembic.command``) rather
than shelling out to ``uv run alembic``. Spawning a subprocess would re-run
this project's whole import chain from scratch, including Cognee's
module-level ``dotenv.load_dotenv(override=True)`` (see ``memory.py``),
which clobbers ``DATABASE_URL`` from a local ``.env`` file before the
subprocess's own settings are read. Calling into Alembic in-process avoids
that: ``cherryai_api.settings.get_settings()`` is already cached by the time
Cognee is ever imported (``build_memory``/``build_agent`` run after this).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg
from alembic import command
from alembic.config import Config
from loguru import logger

from cherryai_api.settings import get_settings

# alembic.ini lives at the repo root (container WORKDIR /app). This file is
# src/cherryai_api/db_migrations.py, so two parents up from its directory is
# the repo root — resolved from __file__, never from the process CWD, since
# tests and the CLI can run from elsewhere.
_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


async def _current_revision(dsn: str) -> str | None:
    """Return the ``alembic_version`` row, or None on an unversioned database."""
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval("SELECT version_num FROM alembic_version")
    except asyncpg.UndefinedTableError:
        return None
    finally:
        await conn.close()


def run_migrations_to_head() -> None:
    """Upgrade the database to the latest Alembic revision.

    Synchronous and blocking — call via ``asyncio.to_thread`` from async
    code. Alembic's own ``migrations/env.py`` opens an async engine and
    drives it with ``asyncio.run()``, which cannot be called from a
    thread that already has a running event loop; a plain worker thread
    has none, so this must not be awaited directly.

    Raises whatever Alembic (or the underlying connection) raises — a
    failed migration must fail startup rather than serve against a stale
    or partially-migrated schema.
    """
    if not _ALEMBIC_INI.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {_ALEMBIC_INI}")

    dsn = get_settings().asyncpg_dsn
    before = asyncio.run(_current_revision(dsn))
    logger.info(f"Running Alembic migrations: {before or '(unversioned)'} -> head")

    config = Config(str(_ALEMBIC_INI))
    command.upgrade(config, "head")

    after = asyncio.run(_current_revision(dsn))
    logger.info(f"Alembic migrations complete: {before or '(unversioned)'} -> {after}")
