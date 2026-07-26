"""Families: org-like groups with per-member roles and module permissions.

Models + constants here; service functions and routes are added by later
tasks in this same module (house style: one module per feature).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi_users import exceptions as fastapi_users_exceptions
from pydantic import BaseModel, ConfigDict, EmailStr
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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.auth import current_verified_user
from cherryai_api.family_context import ACTIVE_FAMILY_COOKIE
from cherryai_api.orm import Base, get_async_session
from cherryai_api.users import User, get_user_manager

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
    """An org-like group that owns shared wiki/meals/planner content."""

    __tablename__ = "families"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FamilyMembership(Base):
    """One user's role and per-module permissions within one family."""

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


class AlreadyMemberError(Exception):
    """Raised by add_member when the user already has a membership in the family."""


async def create_family(session: AsyncSession, *, name: str, organizer_id: uuid.UUID) -> Family:
    """Create a family and add ``organizer_id`` as its organizer."""
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
    """Return ``(Family, FamilyMembership)`` pairs for every family the user belongs to."""
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
    """Return the user's membership in the family, or None if they aren't a member."""
    return (
        await session.execute(
            select(FamilyMembership).where(
                FamilyMembership.family_id == family_id,
                FamilyMembership.user_id == user_id,
            )
        )
    ).scalar_one_or_none()


async def rename_family(session: AsyncSession, family_id: uuid.UUID, name: str) -> None:
    """Rename a family in place."""
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
        # Stamp RLS GUCs so the UPDATEs below pass the wiki_entries RLS policy.
        # The delete-family route checks the caller's role but doesn't call
        # get_capability, so the ORM's after_begin listener has no validated
        # family id to stamp — and without it, the RLS WITH CHECK policy
        # rejects the reassignment UPDATEs (fail-closed).
        await session.execute(
            text(
                "SELECT set_config('app.user_id', :u, true), set_config('app.family_id', :f, true)"
            ),
            {"u": str(organizer_id), "f": str(family_id)},
        )
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


async def add_member(
    session: AsyncSession, *, family_id: uuid.UUID, email: str, role: str
) -> FamilyMembership:
    """Add the user with ``email`` to ``family_id`` with ``role`` and default perms.

    Raises ``LookupError`` if no user has that email, or ``AlreadyMemberError``
    if they already have a membership in the family (uq_membership_family_user).
    """
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        raise LookupError(email)
    default = DEFAULT_PERMS.get(role, PERM_EDIT)
    membership = FamilyMembership(
        family_id=family_id,
        user_id=user.id,
        role=role,
        perm_wiki=default,
        perm_meals=default,
        perm_planner=default,
    )
    session.add(membership)
    try:
        await session.commit()
    except IntegrityError as err:
        await session.rollback()
        raise AlreadyMemberError(email) from err
    return membership


async def create_child_account(
    session: AsyncSession,
    user_manager,
    *,
    family_id: uuid.UUID,
    display_name: str,
    password: str,
    email: str | None,
) -> FamilyMembership:
    """Create a new user account for a child and add them as a family member."""
    from cherryai_api.users import UserCreate

    # `.local` is an IANA special-use TLD that email-validator (and thus
    # fastapi-users' EmailStr field) rejects outright, so the synthetic
    # address uses `.internal` instead — not deliverable, but not on that
    # reserved list either.
    synthetic = email or f"child-{uuid.uuid4().hex}@family.internal"
    user = await user_manager.create(
        UserCreate(email=synthetic, password=password, display_name=display_name),
        safe=True,
    )
    try:
        # Child accounts are approved by construction: a parent just created them.
        user.is_verified = True
        session.add(user)
        await session.flush()
        return await add_member(
            session, family_id=family_id, email=user.email, role=FAMILY_ROLE_CHILD
        )
    except Exception:
        # user_manager.create() already committed the user row above; if
        # anything after that fails, compensate by deleting it rather than
        # leaving an orphaned account with no family membership.
        await session.rollback()
        await session.delete(user)
        await session.commit()
        raise


async def update_member(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    user_id: uuid.UUID,
    changes: dict,
) -> FamilyMembership:
    """Apply ``changes`` (role/perm/gate fields) to an existing membership."""
    membership = await get_membership(session, family_id, user_id)
    if membership is None:
        raise HTTPException(status_code=404)
    for field, value in changes.items():
        setattr(membership, field, value)
    await session.commit()
    await session.refresh(membership)
    return membership


async def remove_member(session: AsyncSession, *, family_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Remove a user's membership from a family."""
    await session.execute(
        delete(FamilyMembership).where(
            FamilyMembership.family_id == family_id,
            FamilyMembership.user_id == user_id,
        )
    )
    await session.commit()


async def transfer_organizer(
    session: AsyncSession,
    *,
    family_id: uuid.UUID,
    from_user_id: uuid.UUID,
    to_user_id: uuid.UUID,
) -> None:
    """Swap organizer->admin and admin->organizer atomically.

    Demote first, then promote, so the partial unique organizer index never
    sees two organizer rows inside the transaction.
    """
    await session.execute(
        update(FamilyMembership)
        .where(
            FamilyMembership.family_id == family_id,
            FamilyMembership.user_id == from_user_id,
        )
        .values(role=FAMILY_ROLE_ADMIN)
    )
    await session.execute(
        update(FamilyMembership)
        .where(
            FamilyMembership.family_id == family_id,
            FamilyMembership.user_id == to_user_id,
        )
        .values(
            role=FAMILY_ROLE_ORGANIZER,
            perm_wiki=PERM_EDIT,
            perm_meals=PERM_EDIT,
            perm_planner=PERM_EDIT,
        )
    )
    await session.commit()


# ---------- routes ----------

families_router = APIRouter(prefix="/api/families", tags=["families"])


class FamilyCreate(BaseModel):
    """Body for creating a family."""

    name: str


class FamilyRename(BaseModel):
    """Body for renaming a family."""

    name: str


class FamilyDelete(BaseModel):
    """Body for deleting a family; requires typing the name back to confirm."""

    confirm_name: str
    content: str  # 'delete' | 'keep_personal'


class ActiveFamily(BaseModel):
    """Body for switching (or clearing, with ``None``) the active family."""

    family_id: uuid.UUID | None


class MembershipOut(BaseModel):
    """A family the caller belongs to, with their role and per-module permissions."""

    id: uuid.UUID
    name: str
    role: str
    perm_wiki: str = PERM_NONE
    perm_meals: str = PERM_NONE
    perm_planner: str = PERM_NONE
    chat_enabled: bool = True
    web_enabled: bool = True


class MemberAdd(BaseModel):
    """Body for adding an existing user to a family by email."""

    email: str
    role: str


class ChildCreate(BaseModel):
    """Body for creating a new child account within a family."""

    display_name: str
    password: str
    email: EmailStr | None = None


class MemberPatch(BaseModel):
    """Partial update to a membership's role, permissions, or gates."""

    model_config = ConfigDict(extra="forbid")

    role: Literal[FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD] | None = None
    perm_wiki: Literal[PERM_NONE, PERM_VIEW, PERM_EDIT] | None = None
    perm_meals: Literal[PERM_NONE, PERM_VIEW, PERM_EDIT] | None = None
    perm_planner: Literal[PERM_NONE, PERM_VIEW, PERM_EDIT] | None = None
    chat_enabled: bool | None = None
    web_enabled: bool | None = None


class TransferBody(BaseModel):
    """Body for transferring the organizer role to an existing admin."""

    user_id: uuid.UUID


class MemberOut(BaseModel):
    """A family member's role and per-module permissions/gates."""

    user_id: uuid.UUID
    display_name: str = ""
    email: str = ""
    role: str
    perm_wiki: str
    perm_meals: str
    perm_planner: str
    chat_enabled: bool
    web_enabled: bool


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
    """List every family the caller belongs to, with their role in each."""
    return [
        MembershipOut(
            id=family.id,
            name=family.name,
            role=m.role,
            perm_wiki=m.perm_wiki,
            perm_meals=m.perm_meals,
            perm_planner=m.perm_planner,
            chat_enabled=m.chat_enabled,
            web_enabled=m.web_enabled,
        )
        for family, m in await list_memberships(session, user.id)
    ]


@families_router.post("", status_code=201, response_model=MembershipOut)
async def create_family_route(
    payload: FamilyCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Create a family with the caller as its organizer."""
    family = await create_family(session, name=payload.name, organizer_id=user.id)
    membership = await get_membership(session, family.id, user.id)
    if membership is None:
        raise HTTPException(status_code=500)  # pragma: no cover
    return MembershipOut(
        id=family.id,
        name=family.name,
        role=FAMILY_ROLE_ORGANIZER,
        perm_wiki=membership.perm_wiki,
        perm_meals=membership.perm_meals,
        perm_planner=membership.perm_planner,
        chat_enabled=membership.chat_enabled,
        web_enabled=membership.web_enabled,
    )


@families_router.post("/active", status_code=204)
async def set_active_family(
    payload: ActiveFamily,
    response: Response,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Switch the caller's active family (or clear it, back to personal)."""
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
    """Rename a family (organizer only)."""
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
    """Delete a family (organizer only), after confirming its name."""
    await _require_role(session, family_id, user.id, (FAMILY_ROLE_ORGANIZER,))
    family = (await session.execute(select(Family).where(Family.id == family_id))).scalar_one()
    if payload.confirm_name != family.name:
        raise HTTPException(status_code=400, detail={"code": "confirm_name_mismatch"})
    if payload.content not in ("delete", "keep_personal"):
        raise HTTPException(status_code=400, detail={"code": "invalid_content_choice"})
    await delete_family(session, family_id, content=payload.content, organizer_id=user.id)


@families_router.get("/{family_id}/members", response_model=list[MemberOut])
async def list_members_route(
    family_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """List every member of a family (any member may view)."""
    await _require_role(
        session,
        family_id,
        user.id,
        (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD),
    )
    rows = (
        await session.execute(
            select(FamilyMembership, User.display_name, User.email)
            .join(User, User.id == FamilyMembership.user_id)
            .where(FamilyMembership.family_id == family_id)
        )
    ).all()
    return [
        MemberOut(
            user_id=m.FamilyMembership.user_id,
            display_name=m.display_name,
            email=m.email,
            role=m.FamilyMembership.role,
            perm_wiki=m.FamilyMembership.perm_wiki,
            perm_meals=m.FamilyMembership.perm_meals,
            perm_planner=m.FamilyMembership.perm_planner,
            chat_enabled=m.FamilyMembership.chat_enabled,
            web_enabled=m.FamilyMembership.web_enabled,
        )
        for m in rows
    ]


@families_router.post("/{family_id}/members", status_code=201, response_model=MemberOut)
async def add_member_route(
    family_id: uuid.UUID,
    payload: MemberAdd,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Add an existing user to a family by email (organizer/admin only)."""
    caller = await _require_role(
        session, family_id, user.id, (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN)
    )
    if payload.role not in (FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD):
        raise HTTPException(status_code=400, detail={"code": "invalid_role"})
    if payload.role == FAMILY_ROLE_ADMIN and caller.role != FAMILY_ROLE_ORGANIZER:
        # minting an admin is organizer-only, same as any other admin
        # involvement in a role change (spec §3) — an admin adding another
        # admin would let admins bootstrap peers with equal standing.
        raise HTTPException(status_code=403, detail={"code": "family_permission_denied"})
    try:
        membership = await add_member(
            session, family_id=family_id, email=payload.email, role=payload.role
        )
    except LookupError as err:
        raise HTTPException(status_code=404, detail={"code": "no_such_user"}) from err
    except AlreadyMemberError as err:
        raise HTTPException(status_code=409, detail={"code": "already_member"}) from err
    return MemberOut.model_validate(membership, from_attributes=True)


@families_router.post("/{family_id}/children", status_code=201, response_model=MemberOut)
async def create_child_route(
    family_id: uuid.UUID,
    payload: ChildCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
    user_manager=Depends(get_user_manager),  # noqa: B008
):
    """Create a new child account within a family (organizer/admin only)."""
    await _require_role(session, family_id, user.id, (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN))
    try:
        membership = await create_child_account(
            session,
            user_manager,
            family_id=family_id,
            display_name=payload.display_name,
            password=payload.password,
            email=payload.email,
        )
    except fastapi_users_exceptions.UserAlreadyExists as err:
        raise HTTPException(status_code=409, detail={"code": "email_taken"}) from err
    return MemberOut.model_validate(membership, from_attributes=True)


@families_router.patch("/{family_id}/members/{member_user_id}", response_model=MemberOut)
async def update_member_route(
    family_id: uuid.UUID,
    member_user_id: uuid.UUID,
    payload: MemberPatch,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Update a member's role, permissions, or gates (organizer/admin only)."""
    caller = await _require_role(
        session, family_id, user.id, (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN)
    )
    target = await get_membership(session, family_id, member_user_id)
    if target is None:
        raise HTTPException(status_code=404)
    if target.role == FAMILY_ROLE_ORGANIZER:
        raise HTTPException(status_code=400, detail={"code": "cannot_modify_organizer"})
    changes = payload.model_dump(exclude_none=True)
    role_change = changes.pop("role", None)
    if role_change is not None:
        # admin<->adult (and any admin involvement) is organizer-only (spec §3)
        if FAMILY_ROLE_ADMIN in (role_change, target.role) and caller.role != FAMILY_ROLE_ORGANIZER:
            raise HTTPException(status_code=403, detail={"code": "family_permission_denied"})
        if role_change == FAMILY_ROLE_ORGANIZER:
            raise HTTPException(status_code=400, detail={"code": "use_transfer"})
        changes["role"] = role_change
    membership = await update_member(
        session, family_id=family_id, user_id=member_user_id, changes=changes
    )
    return MemberOut.model_validate(membership, from_attributes=True)


@families_router.delete("/{family_id}/members/{member_user_id}", status_code=204)
async def remove_member_route(
    family_id: uuid.UUID,
    member_user_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Remove a member from a family (organizer/admin only)."""
    caller = await _require_role(
        session, family_id, user.id, (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN)
    )
    target = await get_membership(session, family_id, member_user_id)
    if target is None:
        raise HTTPException(status_code=404)
    if target.role == FAMILY_ROLE_ORGANIZER:
        raise HTTPException(status_code=400, detail={"code": "organizer_must_transfer_first"})
    if target.role == FAMILY_ROLE_ADMIN and caller.role != FAMILY_ROLE_ORGANIZER:
        # peers can't remove peers; only the organizer can remove an admin
        raise HTTPException(status_code=403, detail={"code": "family_permission_denied"})
    await remove_member(session, family_id=family_id, user_id=member_user_id)


@families_router.post("/{family_id}/transfer")
async def transfer_route(
    family_id: uuid.UUID,
    payload: TransferBody,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
):
    """Transfer the organizer role to an existing admin (organizer only)."""
    await _require_role(session, family_id, user.id, (FAMILY_ROLE_ORGANIZER,))
    target = await get_membership(session, family_id, payload.user_id)
    if target is None:
        raise HTTPException(status_code=404)
    if target.role != FAMILY_ROLE_ADMIN:
        raise HTTPException(status_code=400, detail={"code": "transfer_target_not_admin"})
    await transfer_organizer(
        session, family_id=family_id, from_user_id=user.id, to_user_id=payload.user_id
    )
    return {"organizer": str(payload.user_id)}
