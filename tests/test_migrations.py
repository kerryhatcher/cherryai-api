"""Alembic migrations run against a scratch database, never the dev one."""

import asyncio
import uuid

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


async def _seed_legacy(dsn: str) -> None:
    """Simulate a pre-auth production DB: legacy tables with orphan rows."""
    from cherryai_api.db import _CREATE_TABLES
    from cherryai_api.feedback import CREATE_FEEDBACK_TABLE
    from cherryai_api.wiki import CREATE_WIKI_TABLE

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(_CREATE_TABLES)
        await conn.execute(CREATE_WIKI_TABLE)
        await conn.execute(CREATE_FEEDBACK_TABLE)
        await conn.execute(
            "INSERT INTO sessions (id, title) VALUES ($1, 'Ztest legacy chat')",
            uuid.uuid4(),
        )
        await conn.execute(
            "INSERT INTO wiki_entries (id, slug, title, body) "
            "VALUES ($1, 'ztest-legacy', 'Ztest Legacy', 'body')",
            uuid.uuid4(),
        )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_0002_fresh_database_needs_no_admin(scratch_db, monkeypatch):
    monkeypatch.delenv("CHERRYAI_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("CHERRYAI_ADMIN_PASSWORD", raising=False)
    await asyncio.to_thread(upgrade, scratch_db)  # head; zero rows to backfill
    conn = await asyncpg.connect(scratch_db)
    try:
        cols = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='sessions'"
            )
        }
        assert "user_id" in cols
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_0002_backfills_orphans_to_env_admin(scratch_db, monkeypatch):
    await asyncio.to_thread(upgrade, scratch_db, "0001")
    await _seed_legacy(scratch_db)
    monkeypatch.setenv("CHERRYAI_ADMIN_EMAIL", "ztest-boot@example.com")
    monkeypatch.setenv("CHERRYAI_ADMIN_PASSWORD", "pw-ztest-boot")
    await asyncio.to_thread(upgrade, scratch_db)
    conn = await asyncpg.connect(scratch_db)
    try:
        admin = await conn.fetchrow(
            'SELECT id, role, is_verified, memory_dataset FROM "user" '
            "WHERE email = 'ztest-boot@example.com'"
        )
        assert admin["role"] == "admin" and admin["is_verified"] is True
        from cherryai_api.settings import get_settings

        assert admin["memory_dataset"] == get_settings().cognee_dataset
        orphan_sessions = await conn.fetchval("SELECT count(*) FROM sessions WHERE user_id IS NULL")
        assert orphan_sessions == 0
        owner = await conn.fetchval("SELECT owner_id FROM wiki_entries WHERE slug = 'ztest-legacy'")
        assert owner == admin["id"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_0002_fails_loudly_without_admin_when_orphans_exist(scratch_db, monkeypatch):
    await asyncio.to_thread(upgrade, scratch_db, "0001")
    await _seed_legacy(scratch_db)
    monkeypatch.delenv("CHERRYAI_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("CHERRYAI_ADMIN_PASSWORD", raising=False)
    with pytest.raises(Exception, match="bootstrap"):
        await asyncio.to_thread(upgrade, scratch_db)


@pytest.mark.asyncio
async def test_0002_wiki_slug_unique_per_owner(scratch_db, monkeypatch):
    monkeypatch.setenv("CHERRYAI_ADMIN_EMAIL", "ztest-boot@example.com")
    monkeypatch.setenv("CHERRYAI_ADMIN_PASSWORD", "pw-ztest-boot")
    await asyncio.to_thread(upgrade, scratch_db)
    conn = await asyncpg.connect(scratch_db)
    try:
        a, b = uuid.uuid4(), uuid.uuid4()
        from fastapi_users.password import PasswordHelper

        hashed = PasswordHelper().hash("x")
        for uid, email in ((a, "ztest-a@example.com"), (b, "ztest-b@example.com")):
            await conn.execute(
                'INSERT INTO "user" (id, email, hashed_password, is_active, '
                "is_superuser, is_verified, role, display_name, memory_dataset) "
                "VALUES ($1, $2, $3, true, false, true, 'chat', '', '')",
                uid,
                email,
                hashed,
            )
        # Same slug under two different owners must both insert fine.
        for uid in (a, b):
            await conn.execute(
                "INSERT INTO wiki_entries (id, slug, title, body, owner_id) "
                "VALUES ($1, 'ztest-dup', 'Ztest', '', $2)",
                uuid.uuid4(),
                uid,
            )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                "INSERT INTO wiki_entries (id, slug, title, body, owner_id) "
                "VALUES ($1, 'ztest-dup', 'Ztest', '', $2)",
                uuid.uuid4(),
                a,
            )
    finally:
        await conn.close()
