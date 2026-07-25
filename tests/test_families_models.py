"""Family + FamilyMembership model and constraint tests (dev Postgres)."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from cherryai_api.families import (
    FAMILY_ROLE_ADULT,
    FAMILY_ROLE_ORGANIZER,
    Family,
    FamilyMembership,
)
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


@pytest.mark.asyncio
async def test_family_membership_roundtrip(pool):
    async with async_session_maker() as session:
        user = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        fam = Family(name="Ztest Family")
        session.add(fam)
        await session.flush()
        session.add(
            FamilyMembership(
                family_id=fam.id,
                user_id=user.id,
                role=FAMILY_ROLE_ORGANIZER,
                perm_wiki="edit",
                perm_meals="edit",
                perm_planner="edit",
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_one_organizer_per_family(pool):
    async with async_session_maker() as session:
        u1 = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        u2 = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        fam = Family(name="Ztest Family")
        session.add(fam)
        await session.flush()
        session.add(
            FamilyMembership(
                family_id=fam.id,
                user_id=u1.id,
                role=FAMILY_ROLE_ORGANIZER,
                perm_wiki="edit",
                perm_meals="edit",
                perm_planner="edit",
            )
        )
        await session.flush()
        session.add(
            FamilyMembership(
                family_id=fam.id,
                user_id=u2.id,
                role=FAMILY_ROLE_ORGANIZER,
                perm_wiki="edit",
                perm_meals="edit",
                perm_planner="edit",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()


@pytest.mark.asyncio
async def test_duplicate_membership_rejected(pool):
    async with async_session_maker() as session:
        u1 = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        fam = Family(name="Ztest Family")
        session.add(fam)
        await session.flush()
        session.add(
            FamilyMembership(
                family_id=fam.id,
                user_id=u1.id,
                role=FAMILY_ROLE_ORGANIZER,
                perm_wiki="edit",
                perm_meals="edit",
                perm_planner="edit",
            )
        )
        await session.flush()
        session.add(
            FamilyMembership(
                family_id=fam.id,
                user_id=u1.id,
                role=FAMILY_ROLE_ADULT,
                perm_wiki="edit",
                perm_meals="edit",
                perm_planner="edit",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
