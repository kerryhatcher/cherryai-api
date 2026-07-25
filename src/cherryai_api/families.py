"""Families: org-like groups with per-member roles and module permissions.

Models + constants here; service functions and routes are added by later
tasks in this same module (house style: one module per feature).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.auth import current_verified_user
from cherryai_api.family_context import ACTIVE_FAMILY_COOKIE
from cherryai_api.orm import Base, get_async_session
from cherryai_api.users import User

FAMILY_ROLE_ORGANIZER = "organizer"
FAMILY_ROLE_ADMIN = "admin"
FAMILY_ROLE_ADULT = "adult"
FAMILY_ROLE_CHILD = "child"
FAMILY_ROLES = (
    FAMILY_ROLE_ORGANIZER,
    FAMILY_ROLE_ADMIN,
    FAMILY_ROLE_ADULT,
    FAMILY_ROLE_CHILD,
)

PERM_NONE = "none"
PERM_VIEW = "view"
PERM_EDIT = "edit"
PERM_LEVELS = (PERM_NONE, PERM_VIEW, PERM_EDIT)

# Matrix defaults applied when a membership is created (spec §2).
DEFAULT_PERMS = {FAMILY_ROLE_ADULT: PERM_EDIT, FAMILY_ROLE_CHILD: PERM_NONE}


class Family(Base):
    __tablename__ = "families"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FamilyMembership(Base):
    __tablename__ = "family_memberships"
    __table_args__ = (
        UniqueConstraint("family_id", "user_id", name="uq_membership_family_user"),
        # Exactly one organizer per family (spec §2).
        Index(
            "uq_family_organizer",
            "family_id",
            unique=True,
            postgresql_where=text("role = 'organizer'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("families.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    perm_wiki: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    perm_meals: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    perm_planner: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    chat_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    web_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------- service layer (shared by routes and CLI) ----------


async def create_family(session: AsyncSession, *, name: str, organizer_id: uuid.UUID) -> Family:
    family = Family(name=name)
    session.add(family)
    await session.flush()
    session.add(
        FamilyMembership(
            family_id=family.id,
            user_id=organizer_id,
            role=FAMILY_ROLE_ORGANIZER,
            perm_wiki=PERM_EDIT,
            perm_meals=PERM_EDIT,
            perm_planner=PERM_EDIT,
        )
    )
    await session.commit()
    return family


async def list_memberships(session: AsyncSession, user_id: uuid.UUID):
    rows = await session.execute(
        select(Family, FamilyMembership)
        .join(FamilyMembership, FamilyMembership.family_id == Family.id)
        .where(FamilyMembership.user_id == user_id)
        .order_by(Family.name)
    )
    return list(rows.all())


async def get_membership(
    session: AsyncSession, family_id: uuid.UUID, user_id: uuid.UUID
) -> FamilyMembership | None:
    return (
        await session.execute(
            select(FamilyMembership).where(
                FamilyMembership.family_id == family_id,
                FamilyMembership.user_id == user_id,
            )
        )
    ).scalar_one_or_none()


async def rename_family(session: AsyncSession, family_id: uuid.UUID, name: str) -> None:
    await session.execute(update(Family).where(Family.id == family_id).values(name=name))
    await session.commit()


# Tables that carry family content (grows with later phases' adoption).
_CONTENT_TABLES = (
    "wiki_entries",
    "meal_plans",
    "recipes",
    "shopping_lists",
    "pantry_items",
    "stores",
    "planner_projects",
)


async def delete_family(
    session: AsyncSession,
    family_id: uuid.UUID,
    *,
    content: str,
    organizer_id: uuid.UUID,
) -> None:
    """Delete a family. ``content='delete'`` drops family rows (FK CASCADE);
    ``'keep_personal'`` converts them to the organizer's personal rows first,
    suffixing colliding wiki slugs (spec §3)."""
    if content == "keep_personal":
        # De-conflict wiki slugs against the organizer's personal pages.
        await session.execute(
            text(
                "UPDATE wiki_entries w SET slug = w.slug || '-' || left(w.id::text, 8) "
                "WHERE w.family_id = :fid AND EXISTS (SELECT 1 FROM wiki_entries p "
                "WHERE p.family_id IS NULL AND p.owner_id = :oid AND p.slug = w.slug)"
            ),
            {"fid": family_id, "oid": organizer_id},
        )
        for table in _CONTENT_TABLES:
            await session.execute(
                text(
                    f"UPDATE {table} SET owner_id = :oid, family_id = NULL WHERE family_id = :fid"
                ),
                {"fid": family_id, "oid": organizer_id},
            )
    await session.execute(delete(Family).where(Family.id == family_id))
    await session.commit()


# ---------- routes ----------

families_router = APIRouter(prefix="/api/families", tags=["families"])


class FamilyCreate(BaseModel):
    name: str


class FamilyRename(BaseModel):
    name: str


class FamilyDelete(BaseModel):
    confirm_name: str
    content: str  # 'delete' | 'keep_personal'


class ActiveFamily(BaseModel):
    family_id: uuid.UUID | None


class MembershipOut(BaseModel):
    id: uuid.UUID
    name: str
    role: str


async def _require_role(
    session: AsyncSession,
    family_id: uuid.UUID,
    user_id: uuid.UUID,
    roles: tuple[str, ...],
) -> FamilyMembership:
    membership = await get_membership(session, family_id, user_id)
    if membership is None:
        raise HTTPException(status_code=404)  # existence-protecting (spec §7)
    if membership.role not in roles:
        raise HTTPException(status_code=403, detail={"code": "family_permission_denied"})
    return membership


@families_router.get("", response_model=list[MembershipOut])
async def list_my_families(
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    return [
        MembershipOut(id=family.id, name=family.name, role=m.role)
        for family, m in await list_memberships(session, user.id)
    ]


@families_router.post("", status_code=201, response_model=MembershipOut)
async def create_family_route(
    payload: FamilyCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    family = await create_family(session, name=payload.name, organizer_id=user.id)
    return MembershipOut(id=family.id, name=family.name, role=FAMILY_ROLE_ORGANIZER)


@families_router.post("/active", status_code=204)
async def set_active_family(
    payload: ActiveFamily,
    response: Response,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    if payload.family_id is None:
        response.delete_cookie(ACTIVE_FAMILY_COOKIE)
        return
    if await get_membership(session, payload.family_id, user.id) is None:
        raise HTTPException(status_code=403, detail={"code": "family_permission_denied"})
    # samesite/secure mirror the auth cookie; the value is server-validated
    # on every request, so this cookie is a selector, not an authority.
    response.set_cookie(
        ACTIVE_FAMILY_COOKIE,
        str(payload.family_id),
        httponly=True,
        samesite="lax",
        secure=True,
        max_age=60 * 60 * 24 * 365,
    )


@families_router.patch("/{family_id}", status_code=200)
async def rename_family_route(
    family_id: uuid.UUID,
    payload: FamilyRename,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    await _require_role(session, family_id, user.id, (FAMILY_ROLE_ORGANIZER,))
    await rename_family(session, family_id, payload.name)
    return {"id": str(family_id), "name": payload.name}


@families_router.delete("/{family_id}", status_code=204)
async def delete_family_route(
    family_id: uuid.UUID,
    payload: FamilyDelete,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    await _require_role(session, family_id, user.id, (FAMILY_ROLE_ORGANIZER,))
    family = (await session.execute(select(Family).where(Family.id == family_id))).scalar_one()
    if payload.confirm_name != family.name:
        raise HTTPException(status_code=400, detail={"code": "confirm_name_mismatch"})
    if payload.content not in ("delete", "keep_personal"):
        raise HTTPException(status_code=400, detail={"code": "invalid_content_choice"})
    await delete_family(session, family_id, content=payload.content, organizer_id=user.id)
