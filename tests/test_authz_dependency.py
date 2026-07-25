"""get_capability against the dev Postgres: membership resolution + fallback."""

import uuid

import pytest
from sqlalchemy import text

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


async def _current_guc(session, name: str) -> str:
    row = (await session.execute(text("SELECT current_setting(:n, true)"), {"n": name})).scalar()
    return row or ""


@pytest.mark.asyncio
async def test_bogus_active_family_does_not_leak_into_guc(pool):
    """The ORM's after_begin GUC listener stamps app.family_id straight from
    the request's active_family_var when a transaction begins — which
    typically happens before get_capability has validated that id against
    the caller's memberships. get_capability must restamp the
    already-open transaction with the validated value once it knows better,
    not leave the unvalidated/bogus request id sitting in the GUC."""
    async with async_session_maker() as session:
        org, kid, fam = await _fixture(session)
        # A transaction begun (and GUC-stamped by the listener) before
        # validation, using a family id that is not one of org's memberships
        # — exactly the "request input, unvalidated" state the listener sees.
        bogus = uuid.uuid4()
        await session.execute(
            text(
                "SELECT set_config('app.user_id', :u, true), set_config('app.family_id', :f, true)"
            ),
            {"u": "", "f": str(bogus)},
        )
        active_family_var.set(bogus)
        cap = await get_capability(user=org, session=session)
        assert cap.family_id is None  # fell back to personal (spec §7)
        assert await _current_guc(session, "app.family_id") == ""
        assert await _current_guc(session, "app.user_id") == str(org.id)


@pytest.mark.asyncio
async def test_valid_active_family_restamps_guc(pool):
    async with async_session_maker() as session:
        org, kid, fam = await _fixture(session)
        active_family_var.set(fam.id)
        cap = await get_capability(user=org, session=session)
        assert cap.family_id == fam.id
        assert await _current_guc(session, "app.family_id") == str(fam.id)
        assert await _current_guc(session, "app.user_id") == str(org.id)
