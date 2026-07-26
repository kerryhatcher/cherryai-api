"""Admin-only user management: approval queue, roles, deactivation."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cherryai_api.auth import require_admin
from cherryai_api.model_configs import (
    ALL_CALL_SITES,
    delete_config,
    list_configs,
    upsert_config,
)
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
    user = await _get_user_or_404(session, user_id)
    _validate_role(role)
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


# ── Model configs ────────────────────────────────────────────────────────────


model_configs_router = APIRouter(prefix="/admin/model-configs", tags=["admin"])


class ModelConfigOut(BaseModel):
    call_site: str
    provider: str
    base_url: str
    has_api_key: bool
    model_name: str
    updated_at: datetime


class ModelConfigUpsert(BaseModel):
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""


@model_configs_router.get("", response_model=list[ModelConfigOut])
async def list_model_configs(
    request: Request,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> list[dict]:
    """List all model configs (masking API keys)."""
    pool = request.app.state.db.pool
    rows = await list_configs(pool)
    result: list[dict] = []
    for site in ALL_CALL_SITES:
        row = next((r for r in rows if r.call_site == site), None)
        result.append(
            {
                "call_site": site,
                "provider": row.provider if row else "",
                "base_url": row.base_url if row else "",
                "has_api_key": bool(row.api_key) if row else False,
                "model_name": row.model_name if row else "",
                "updated_at": row.updated_at if row else None,
            }
        )
    return result


@model_configs_router.put("/{call_site}", response_model=ModelConfigOut)
async def update_model_config(
    request: Request,
    call_site: str,
    body: ModelConfigUpsert,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> dict:
    """Upsert a model config for *call_site*."""
    if call_site not in ALL_CALL_SITES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown call_site '{call_site}'. Valid: {', '.join(ALL_CALL_SITES)}",
        )
    pool = request.app.state.db.pool
    row = await upsert_config(
        pool,
        call_site=call_site,
        provider=body.provider,
        base_url=body.base_url,
        api_key=body.api_key,
        model_name=body.model_name,
    )
    return {
        "call_site": row.call_site,
        "provider": row.provider,
        "base_url": row.base_url,
        "has_api_key": bool(row.api_key),
        "model_name": row.model_name,
        "updated_at": row.updated_at,
    }


@model_configs_router.delete("/{call_site}", status_code=204)
async def delete_model_config(
    request: Request,
    call_site: str,
    admin: User = Depends(require_admin),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> None:
    """Delete a model config, reverting to env-var defaults."""
    if call_site not in ALL_CALL_SITES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown call_site '{call_site}'. Valid: {', '.join(ALL_CALL_SITES)}",
        )
    pool = request.app.state.db.pool
    if not await delete_config(pool, call_site):
        raise HTTPException(status_code=404, detail="Model config not found")
