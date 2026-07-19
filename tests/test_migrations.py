"""Alembic migrations run against a scratch database, never the dev one."""

import asyncio

import asyncpg
import pytest
from alembic import command
from alembic.config import Config

from cherryai_api.settings import get_settings

_SCRATCH_DB = "ztest_migrations"


def _scratch_url() -> str:
    base = get_settings().asyncpg_dsn
    root, _, _ = base.rpartition("/")
    return f"{root}/{_SCRATCH_DB}"


@pytest.fixture
async def scratch_db():
    """Create an empty scratch database and yield its asyncpg DSN."""
    admin = await asyncpg.connect(get_settings().asyncpg_dsn)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} (FORCE)")
        await admin.execute(f"CREATE DATABASE {_SCRATCH_DB}")
    finally:
        await admin.close()
    yield _scratch_url()
    admin = await asyncpg.connect(get_settings().asyncpg_dsn)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} (FORCE)")
    finally:
        await admin.close()


def alembic_config(dsn: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", dsn.replace("postgresql://", "postgresql+asyncpg://", 1))
    return cfg


def upgrade(dsn: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(dsn), revision)


@pytest.mark.asyncio
async def test_0001_creates_identity_tables(scratch_db):
    await asyncio.to_thread(upgrade, scratch_db, "0001")
    conn = await asyncpg.connect(scratch_db)
    try:
        tables = {
            r["tablename"]
            for r in await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        }
    finally:
        await conn.close()
    assert {"user", "accesstoken"} <= tables
