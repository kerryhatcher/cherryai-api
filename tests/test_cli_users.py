"""ensure_admin service function and CLI users commands."""

import pytest

from cherryai_api.orm import async_session_maker
from cherryai_api.users import ensure_admin


@pytest.mark.asyncio
async def test_ensure_admin_is_idempotent(pool):
    async with async_session_maker() as session:
        user, created = await ensure_admin(session, "ztest-cliboot@example.com", "pw-ztest-cli")
        assert created is True
        assert user.role == "admin" and user.is_verified and user.is_superuser
    async with async_session_maker() as session:
        again, created = await ensure_admin(session, "ztest-cliboot@example.com", "pw-ztest-cli")
        assert created is False
        assert again.id == user.id


def test_users_list_runs(pool):
    from typer.testing import CliRunner

    from cherryai_api.cli import app

    result = CliRunner().invoke(app, ["users", "list"])
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_users_deactivate_is_atomic(pool, make_user):
    """Deactivation and token revocation must occur in one session.

    Regression test for: if pass 2 fails after pass 1 commits,
    deactivated account keeps live tokens.
    """
    import subprocess
    import sys

    user = await make_user("ztest-cli-deact@example.com")

    # Insert an AccessToken directly via pool
    token_value = "a" * 43  # 43-char token
    await pool.execute(
        "INSERT INTO accesstoken (token, user_id, created_at) VALUES ($1, $2, now())",
        token_value,
        user["id"],
    )

    # Verify token exists before deactivation
    token_count_before = await pool.fetchval(
        "SELECT COUNT(*) FROM accesstoken WHERE user_id = $1",
        user["id"],
    )
    assert token_count_before == 1

    # Invoke deactivate command in subprocess (avoids asyncio.run() conflict)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "users",
            "deactivate",
            "ztest-cli-deact@example.com",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "OK:" in result.stdout

    # Verify both deactivation and token revocation occurred atomically
    is_active = await pool.fetchval(
        'SELECT is_active FROM "user" WHERE id = $1',
        user["id"],
    )
    assert is_active is False

    token_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM accesstoken WHERE user_id = $1",
        user["id"],
    )
    assert token_count_after == 0
