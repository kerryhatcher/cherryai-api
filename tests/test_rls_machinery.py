"""RLS machinery: GUCs, policy template, isolation on a scratch table.

The dev `cherryai` Postgres role (the one `pool`/DATABASE_URL connect as
everywhere else in this suite) is SUPERUSER + BYPASSRLS, and it is also the
owner of every table. Postgres unconditionally exempts superuser/BYPASSRLS
roles from row security -- FORCE ROW LEVEL SECURITY only ever subjects the
*table owner* to its policies, never a superuser -- so RLS enforcement is
invisible on that connection no matter what the policy SQL says. This is a
known gap: docs/research/2026-07-25-family-authz-multitenancy.md's role
hygiene section calls for a non-superuser, non-owner application role and
says to never grant BYPASSRLS, but the dev/test environment has never had
that separation (one role, shared by migrations, the app, and every test).
Fixing that for real means touching shared infra used by the whole test
suite and every other concurrent worktree, which is out of scope here --
see task-9-report.md's "Phase 2 prerequisite" note.

So the tests that actually exercise enforcement connect through a throwaway,
unprivileged role (`unprivileged_pool` fixture below) created and dropped
around each test, instead of the shared superuser `pool`.
"""

import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from cherryai_api.authz import (
    Capability,
    assert_rls_enforced,
    rls_policy_sql,
    scoped_connection,
)
from cherryai_api.settings import get_settings

UID_A, UID_B, FAM_A, FAM_B = (uuid.uuid4() for _ in range(4))

_RLS_TEST_ROLE = "ztest_rls_user"
_RLS_TEST_PASSWORD = "ztest-rls"


def cap(user, family=None):
    return Capability(user, family, None, frozenset())


def _unprivileged_dsn() -> str:
    """The app's DSN with credentials swapped for the temp test role.

    ``asyncpg_dsn``, not ``database_url``: asyncpg rejects the
    ``postgresql+asyncpg://`` scheme the raw settings field may carry.
    """
    parts = urlsplit(get_settings().asyncpg_dsn)
    netloc = f"{_RLS_TEST_ROLE}:{_RLS_TEST_PASSWORD}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit(parts._replace(netloc=netloc))


@pytest.fixture()
async def scratch_table(pool):
    name = f"ztest_rls_{uuid.uuid4().hex[:8]}"
    await pool.execute(
        f"CREATE TABLE {name} (id UUID PRIMARY KEY, owner_id UUID NOT NULL, "
        f"family_id UUID, body TEXT NOT NULL)"
    )
    for statement in rls_policy_sql(name):
        await pool.execute(statement)
    # seed: A-personal, A-family, B-family rows
    await pool.execute(
        f"INSERT INTO {name} VALUES ($1,$2,NULL,'a-personal'), "
        f"($3,$2,$4,'fam-a'), ($5,$6,$7,'fam-b')",
        uuid.uuid4(),
        UID_A,
        uuid.uuid4(),
        FAM_A,
        uuid.uuid4(),
        UID_B,
        FAM_B,
    )
    yield name
    await pool.execute(f"DROP TABLE {name}")


@pytest.fixture()
async def unprivileged_pool(pool, scratch_table):
    """A temp non-superuser/non-BYPASSRLS role, scoped to this test's table.

    Built via the shared superuser `pool` (idempotent create, then teardown
    via DROP OWNED BY + DROP ROLE) so RLS behavior is observable at all --
    see the module docstring for why the shared `pool` itself can't show it.
    """
    await pool.execute(f"DROP ROLE IF EXISTS {_RLS_TEST_ROLE}")
    await pool.execute(
        f"CREATE ROLE {_RLS_TEST_ROLE} LOGIN PASSWORD '{_RLS_TEST_PASSWORD}' "
        "NOSUPERUSER NOBYPASSRLS"
    )
    await pool.execute(f"GRANT SELECT, INSERT ON {scratch_table} TO {_RLS_TEST_ROLE}")
    temp_pool = await asyncpg.create_pool(_unprivileged_dsn(), min_size=1, max_size=2)
    try:
        yield temp_pool
    finally:
        await temp_pool.close()
        await pool.execute(f"DROP OWNED BY {_RLS_TEST_ROLE}")
        await pool.execute(f"DROP ROLE {_RLS_TEST_ROLE}")


@pytest.mark.asyncio
async def test_personal_scope_sees_only_own_personal_rows(unprivileged_pool, scratch_table):
    async with scoped_connection(unprivileged_pool, cap(UID_A)) as conn:
        rows = await conn.fetch(f"SELECT body FROM {scratch_table}")
    assert [r["body"] for r in rows] == ["a-personal"]


@pytest.mark.asyncio
async def test_family_scope_sees_only_that_family(unprivileged_pool, scratch_table):
    async with scoped_connection(unprivileged_pool, cap(UID_A, FAM_A)) as conn:
        rows = await conn.fetch(f"SELECT body FROM {scratch_table} ORDER BY body")
    assert [r["body"] for r in rows] == ["a-personal", "fam-a"]


@pytest.mark.asyncio
async def test_cross_family_write_rejected(unprivileged_pool, scratch_table):
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with scoped_connection(unprivileged_pool, cap(UID_A, FAM_A)) as conn:
            await conn.execute(
                f"INSERT INTO {scratch_table} VALUES ($1,$2,$3,'evil')", uuid.uuid4(), UID_A, FAM_B
            )


@pytest.mark.asyncio
async def test_unscoped_connection_sees_nothing(unprivileged_pool, scratch_table):
    rows = await unprivileged_pool.fetch(f"SELECT body FROM {scratch_table}")
    assert rows == []  # fail-closed without GUCs (spec §7)


@pytest.mark.asyncio
async def test_assert_rls_enforced_passes_on_empty_registry(pool):
    await assert_rls_enforced(pool)  # RLS_TABLES is () in phase 1


@pytest.mark.asyncio
async def test_assert_rls_enforced_catches_disable_row_level_security(
    pool, scratch_table, monkeypatch
):
    """DISABLE ROW LEVEL SECURITY clears relrowsecurity but leaves the FORCE
    bit set — a predicate that only checks relforcerowsecurity would miss it.
    """
    import cherryai_api.authz as authz_module

    monkeypatch.setattr(authz_module, "RLS_TABLES", (scratch_table,))
    await assert_rls_enforced(pool)  # policies applied by the fixture → passes

    await pool.execute(f"ALTER TABLE {scratch_table} DISABLE ROW LEVEL SECURITY")
    with pytest.raises(RuntimeError, match=scratch_table):
        await assert_rls_enforced(pool)
