"""Family authorization: the whole permission model in one place.

Scope strings are OAuth-shaped (spec §1) so a future MCP/on-behalf-of
mapping is a relabeling, not a redesign.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cherryai_api.auth import current_verified_user
from cherryai_api.families import (
    FAMILY_ROLE_ADMIN,
    FAMILY_ROLE_CHILD,
    FAMILY_ROLE_ORGANIZER,
    PERM_EDIT,
    PERM_VIEW,
    FamilyMembership,
)
from cherryai_api.family_context import active_family_var, current_user_var
from cherryai_api.orm import get_async_session
from cherryai_api.users import User

MODULES = ("wiki", "meals", "planner")
SCOPE_ADULT_WIKI = "wiki:read:adult"
_FASTMAIL_SCOPES = frozenset({"calendar", "contacts", "email"})


@dataclass(frozen=True)
class Capability:
    """What the requesting user may do in the current (personal|family) context."""

    user_id: uuid.UUID
    family_id: uuid.UUID | None  # None = personal context
    role: str | None  # family role; None in personal context
    scopes: frozenset[str]

    def has(self, scope: str) -> bool:
        return scope in self.scopes


def build_scopes(
    *,
    role: str | None,
    perms: dict[str, str],
    chat_enabled: bool,
    web_enabled: bool,
    is_child_anywhere: bool,
) -> frozenset[str]:
    """Compute the scope set. ``role=None`` means personal context.

    ``chat_enabled``/``web_enabled`` must already be strictest-wins resolved
    across the user's child memberships (spec §3); ``is_child_anywhere`` is
    True when the user holds a child role in any family (hard invariant #2:
    no Fastmail scopes, ever).
    """
    scopes: set[str] = set()
    if role is None or role in (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN):
        for m in MODULES:
            scopes.add(f"{m}:read")
            scopes.add(f"{m}:write")
    else:
        for m in MODULES:
            level = perms[m]
            if level in (PERM_VIEW, PERM_EDIT):
                scopes.add(f"{m}:read")
            if level == PERM_EDIT:
                scopes.add(f"{m}:write")
    if role == FAMILY_ROLE_ORGANIZER:
        scopes.update({"family:manage", "family:own"})
    elif role == FAMILY_ROLE_ADMIN:
        scopes.add("family:manage")
    if role != FAMILY_ROLE_CHILD:
        scopes.add(SCOPE_ADULT_WIKI)  # hard invariant #1: never for children
    if not is_child_anywhere:
        scopes.update(_FASTMAIL_SCOPES)
    if chat_enabled:
        scopes.add("chat")
    if web_enabled:
        scopes.add("web")
    return frozenset(scopes)


_LEVEL_SCOPE = {"view": "read", "edit": "write"}


async def get_capability(
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> Capability:
    """Resolve the caller's Capability for the active (or personal) context."""
    current_user_var.set(user.id)
    rows = (
        (await session.execute(select(FamilyMembership).where(FamilyMembership.user_id == user.id)))
        .scalars()
        .all()
    )
    child_rows = [m for m in rows if m.role == FAMILY_ROLE_CHILD]
    is_child_anywhere = bool(child_rows)
    chat_enabled = all(m.chat_enabled for m in child_rows)  # vacuously True
    web_enabled = all(m.web_enabled for m in child_rows)
    requested = active_family_var.get()
    active = next((m for m in rows if m.family_id == requested), None)
    if active is None:
        active_family_var.set(None)  # stale/invalid → personal (spec §7)
        return Capability(
            user_id=user.id,
            family_id=None,
            role=None,
            scopes=build_scopes(
                role=None,
                perms={},
                chat_enabled=chat_enabled,
                web_enabled=web_enabled,
                is_child_anywhere=is_child_anywhere,
            ),
        )
    perms = {
        "wiki": active.perm_wiki,
        "meals": active.perm_meals,
        "planner": active.perm_planner,
    }
    return Capability(
        user_id=user.id,
        family_id=active.family_id,
        role=active.role,
        scopes=build_scopes(
            role=active.role,
            perms=perms,
            chat_enabled=chat_enabled,
            web_enabled=web_enabled,
            is_child_anywhere=is_child_anywhere,
        ),
    )


def require_permission(module: str, level: str = "view"):
    """Endpoint guard: 403 with a stable code unless the scope is held."""
    scope = f"{module}:{_LEVEL_SCOPE[level]}"

    async def dependency(
        capability: Capability = Depends(get_capability),  # noqa: B008
    ) -> Capability:
        if not capability.has(scope):
            raise HTTPException(
                status_code=403,
                detail={"code": "family_permission_denied", "module": module},
            )
        return capability

    return dependency
