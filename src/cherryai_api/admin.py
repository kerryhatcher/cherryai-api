"""Admin-only user management: approval queue, roles, deactivation."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cherryai_api.auth import require_admin
from cherryai_api.orm import get_async_session
from cherryai_api.users import ROLE_ADMIN, ROLE_CHAT, ROLES, AccessToken, User

router = APIRouter(prefix="/admin/users", tags=["admin"])


class AdminUserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str
    display_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ApproveRequest(BaseModel):
    role: str = ROLE_CHAT


class AdminUserPatch(BaseModel):
    role: str | None = None
    display_name: str | None = None


def _validate_role(role: str) -> None:
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"Unknown role: {role}")


async def _get_user_or_404(session: AsyncSession, user_id: uuid.UUID) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def _revoke_tokens(session: AsyncSession, user_id: uuid.UUID) -> None:
    await session.execute(delete(AccessToken).where(AccessToken.user_id == user_id))


@router.get("", response_model=list[AdminUserOut])
async def list_users(
    status: str | None = None,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> list[User]:
    query = select(User).order_by(User.created_at)
    if status == "pending":
        query = query.where(User.is_verified.is_(False), User.is_active.is_(True))
    result = await session.execute(query)
    return list(result.scalars())


@router.post("/{user_id}/approve", response_model=AdminUserOut)
async def approve_user(
    user_id: uuid.UUID,
    body: ApproveRequest | None = None,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> User:
    role = (body.role if body else ROLE_CHAT) or ROLE_CHAT
    _validate_role(role)
    user = await _get_user_or_404(session, user_id)
    user.is_verified = True
    user.role = role
    user.is_superuser = role == ROLE_ADMIN
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/{user_id}/reject", status_code=204)
async def reject_user(
    user_id: uuid.UUID,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> None:
    user = await _get_user_or_404(session, user_id)
    if user.is_verified:
        raise HTTPException(status_code=409, detail="Only pending users can be rejected")
    await _revoke_tokens(session, user.id)
    await session.delete(user)
    await session.commit()


@router.patch("/{user_id}", response_model=AdminUserOut)
async def patch_user(
    user_id: uuid.UUID,
    body: AdminUserPatch,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> User:
    user = await _get_user_or_404(session, user_id)
    if body.role is not None:
        _validate_role(body.role)
        if user.id == admin.id and body.role != ROLE_ADMIN:
            raise HTTPException(status_code=409, detail="Admins cannot demote themselves")
        user.role = body.role
        user.is_superuser = body.role == ROLE_ADMIN
    if body.display_name is not None:
        user.display_name = body.display_name
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/{user_id}/deactivate", response_model=AdminUserOut)
async def deactivate_user(
    user_id: uuid.UUID,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> User:
    if user_id == admin.id:
        raise HTTPException(status_code=409, detail="Admins cannot deactivate themselves")
    user = await _get_user_or_404(session, user_id)
    user.is_active = False
    await _revoke_tokens(session, user.id)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/{user_id}/reactivate", response_model=AdminUserOut)
async def reactivate_user(
    user_id: uuid.UUID,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> User:
    user = await _get_user_or_404(session, user_id)
    user.is_active = True
    await session.commit()
    await session.refresh(user)
    return user
