"""get_capability against the dev Postgres: membership resolution + fallback."""

import uuid

import pytest

from cherryai_api.authz import get_capability
from cherryai_api.families import (
    FAMILY_ROLE_CHILD,
    FAMILY_ROLE_ORGANIZER,
    PERM_EDIT,
    PERM_NONE,
    PERM_VIEW,
    Family,
    FamilyMembership,
)
from cherryai_api.family_context import active_family_var
from cherryai_api.orm import async_session_maker
from cherryai_api.users import User


async def _mk_user(session, email: str) -> User:
    user = User(
        email=email,
        hashed_password="x",
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role="chat",
        display_name="Ztest",
        memory_dataset="ztest",
    )
    session.add(user)
    await session.flush()
    return user


async def _fixture(session):
    """One family: organizer + child (wiki=view, chat off). Returns (org, kid, fam)."""
    org = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
    kid = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
    fam = Family(name="Ztest Family")
    session.add(fam)
    await session.flush()
    session.add(
        FamilyMembership(
            family_id=fam.id,
            user_id=org.id,
            role=FAMILY_ROLE_ORGANIZER,
            perm_wiki=PERM_EDIT,
            perm_meals=PERM_EDIT,
            perm_planner=PERM_EDIT,
        )
    )
    session.add(
        FamilyMembership(
            family_id=fam.id,
            user_id=kid.id,
            role=FAMILY_ROLE_CHILD,
            perm_wiki=PERM_VIEW,
            perm_meals=PERM_NONE,
            perm_planner=PERM_NONE,
            chat_enabled=False,
        )
    )
    await session.commit()
    return org, kid, fam


@pytest.mark.asyncio
async def test_active_family_capability(pool):
    async with async_session_maker() as session:
        org, kid, fam = await _fixture(session)
        active_family_var.set(fam.id)
        cap = await get_capability(user=kid, session=session)
        assert cap.family_id == fam.id and cap.role == FAMILY_ROLE_CHILD
        assert cap.has("wiki:read") and not cap.has("wiki:write")
        assert not cap.has("meals:read")
        assert not cap.has("chat")  # gate off
        assert not cap.has("wiki:read:adult")  # invariant
        assert not cap.has("email")  # invariant


@pytest.mark.asyncio
async def test_stale_family_falls_back_to_personal(pool):
    async with async_session_maker() as session:
        org, kid, fam = await _fixture(session)
        active_family_var.set(uuid.uuid4())  # not a membership
        cap = await get_capability(user=org, session=session)
        assert cap.family_id is None and cap.role is None
        assert cap.has("wiki:write")  # personal context, own data


@pytest.mark.asyncio
async def test_organizer_capability(pool):
    async with async_session_maker() as session:
        org, kid, fam = await _fixture(session)
        active_family_var.set(fam.id)
        cap = await get_capability(user=org, session=session)
        assert cap.has("family:own") and cap.has("family:manage")
        assert cap.has("wiki:read:adult") and cap.has("email")
