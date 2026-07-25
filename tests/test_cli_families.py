"""CLI tests for families commands (create, list, show, add-member, set-perm, transfer)."""

import subprocess
import sys
import uuid

import pytest

from cherryai_api.orm import async_session_maker
from tests.test_families_models import _mk_user


@pytest.mark.asyncio
async def test_families_create_list_setperm(pool):
    """Test families CLI: create, list, and set-perm commands."""
    email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    async with async_session_maker() as session:
        await _mk_user(session, email)
        await session.commit()

    # Use subprocess.run to avoid asyncio.run() conflict with pytest-asyncio
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "create",
            "Ztest CLI Fam",
            "--organizer-email",
            email,
        ],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, f"Command failed: {created.stderr}"

    listed = subprocess.run(
        [sys.executable, "-m", "cherryai_api.cli", "families", "list"],
        capture_output=True,
        text=True,
    )
    assert "Ztest CLI Fam" in listed.stdout

    # organizer perms are implicit; set-perm on the organizer errors politely
    fam_id = created.stdout.strip().splitlines()[-1].split()[-1]
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "set-perm",
            fam_id,
            email,
            "wiki",
            "view",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
