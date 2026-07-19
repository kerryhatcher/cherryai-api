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
