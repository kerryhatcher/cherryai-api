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
    fam_id = created.stdout.strip().split()[-1].split("=")[-1]
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


@pytest.mark.asyncio
async def test_families_add_member_validates_role(pool):
    """Test add-member rejects invalid roles."""
    organizer_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    member_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    async with async_session_maker() as session:
        await _mk_user(session, organizer_email)
        await _mk_user(session, member_email)
        await session.commit()

    # Create family
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "create",
            "Ztest Validation",
            "--organizer-email",
            organizer_email,
        ],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0
    fam_id = created.stdout.strip().split()[-1].split("=")[-1]

    # Test invalid role
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "add-member",
            fam_id,
            member_email,
            "--role",
            "invalid_role",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "role must be one of" in r.stderr

    # Test organizer role (not allowed)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "add-member",
            fam_id,
            member_email,
            "--role",
            "organizer",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "role must be one of" in r.stderr


@pytest.mark.asyncio
async def test_families_add_member_happy_path(pool):
    """Test add-member with valid role succeeds and creates membership."""
    from sqlalchemy import select

    from cherryai_api.families import FamilyMembership

    organizer_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    member_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    member_id = None
    async with async_session_maker() as session:
        await _mk_user(session, organizer_email)
        member = await _mk_user(session, member_email)
        member_id = member.id
        await session.commit()

    # Create family
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "create",
            "Ztest Happy Path",
            "--organizer-email",
            organizer_email,
        ],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0
    fam_id = created.stdout.strip().split()[-1].split("=")[-1]

    # Add member with valid role (adult)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "add-member",
            fam_id,
            member_email,
            "--role",
            "adult",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"add-member failed: {r.stderr}"
    assert "added" in r.stdout

    # Verify membership was created
    async with async_session_maker() as session:
        memberships = await session.execute(
            select(FamilyMembership).where(
                FamilyMembership.family_id == uuid.UUID(fam_id),
                FamilyMembership.user_id == member_id,
            )
        )
        found = memberships.scalar_one_or_none()
        assert found is not None
        assert found.role == "adult"


@pytest.mark.asyncio
async def test_families_add_member_duplicate_rejected(pool):
    """Re-adding a member who's already in the family exits non-zero with a
    clear message instead of crashing on the uq_membership_family_user
    IntegrityError."""
    organizer_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    member_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    async with async_session_maker() as session:
        await _mk_user(session, organizer_email)
        await _mk_user(session, member_email)
        await session.commit()

    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "create",
            "Ztest Dup Member",
            "--organizer-email",
            organizer_email,
        ],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0
    fam_id = created.stdout.strip().split()[-1].split("=")[-1]

    add_args = [
        sys.executable,
        "-m",
        "cherryai_api.cli",
        "families",
        "add-member",
        fam_id,
        member_email,
        "--role",
        "adult",
    ]
    first = subprocess.run(add_args, capture_output=True, text=True)
    assert first.returncode == 0, f"add-member failed: {first.stderr}"

    second = subprocess.run(add_args, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already a member" in second.stderr


@pytest.mark.asyncio
async def test_families_transfer_validates_target_role(pool):
    """Test transfer rejects non-admin targets."""
    from sqlalchemy import select

    from cherryai_api.families import add_member, create_family
    from cherryai_api.users import User

    organizer_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    adult_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"
    admin_email = f"ztest-{uuid.uuid4().hex[:8]}@example.com"

    async with async_session_maker() as session:
        await _mk_user(session, organizer_email)
        await _mk_user(session, adult_email)
        await _mk_user(session, admin_email)
        await session.commit()

    # Create family and add adult + admin members
    async with async_session_maker() as session:
        organizer = (
            await session.execute(select(User).where(User.email == organizer_email))
        ).scalar_one()
        family = await create_family(session, name="Ztest Transfer", organizer_id=organizer.id)
        fam_id = str(family.id)
        await add_member(session, family_id=family.id, email=adult_email, role="adult")
        await add_member(session, family_id=family.id, email=admin_email, role="admin")
        await session.commit()

    # Test transfer to adult (should fail)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "transfer",
            fam_id,
            adult_email,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "transfer target must be an admin" in r.stderr

    # Test transfer to admin (should succeed)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cherryai_api.cli",
            "families",
            "transfer",
            fam_id,
            admin_email,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "ok" in r.stdout
