"""Family authorization: the whole permission model in one place.

Scope strings are OAuth-shaped (spec §1) so a future MCP/on-behalf-of
mapping is a relabeling, not a redesign.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from sqlalchemy import select, text
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
from cherryai_api.family_context import active_family_var, current_user_var, validated_family_var
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
        """Return whether this capability includes ``scope``."""
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
        validated_family_var.set(None)
        await _restamp_gucs(session, user_id=user.id, family_id=None)
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
    validated_family_var.set(active.family_id)
    await _restamp_gucs(session, user_id=user.id, family_id=active.family_id)
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


async def _restamp_gucs(
    session: AsyncSession, *, user_id: uuid.UUID, family_id: uuid.UUID | None
) -> None:
    """Re-stamp the session's already-open transaction with validated GUCs.

    The ORM ``after_begin`` listener (orm.py) stamps GUCs from the request
    ContextVars when the transaction *begins* — which for most requests is
    during fastapi-users' token lookup, well before this function has
    validated ``active_family_var`` against the caller's memberships. A
    ContextVar update after that point can't reach an already-open
    transaction, so without this explicit restamp a bogus/unauthorized
    family id from the request would sit in ``app.family_id`` for the whole
    request. This runs after validation, on the same session, so it always
    reflects the resolved Capability rather than raw request input.
    """
    await session.execute(
        text("SELECT set_config('app.user_id', :u, true), set_config('app.family_id', :f, true)"),
        {"u": str(user_id), "f": str(family_id) if family_id else ""},
    )


def require_permission(module: str, level: str = "view"):
    """Endpoint guard: 403 with a stable code unless the scope is held.

    Validates ``level`` immediately so a typo'd level fails at route
    definition/import time, not on the first request that hits it.
    """
    if level not in _LEVEL_SCOPE:
        raise ValueError(f"level must be one of {tuple(_LEVEL_SCOPE)}")
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


def scope_sql(capability: Capability, *, alias: str = "", start: int = 1) -> tuple[str, list]:
    """Raw-SQL WHERE fragment limiting rows to the capability's context.

    For the legacy asyncpg queries. Placeholders are numbered from ``start``;
    append the returned params to the query's argument list in order.
    """
    p = f"{alias}." if alias else ""
    if capability.family_id is None:
        return f"({p}family_id IS NULL AND {p}owner_id = ${start})", [capability.user_id]
    return f"({p}family_id = ${start})", [capability.family_id]


def wiki_visibility_sql(
    capability: Capability, *, alias: str = "", start: int = 1
) -> tuple[str, list]:
    """scope_sql + the audience predicate (hard invariant #1) for wiki reads."""
    clause, params = scope_sql(capability, alias=alias, start=start)
    if capability.family_id is not None and not capability.has(SCOPE_ADULT_WIKI):
        p = f"{alias}." if alias else ""
        clause = f"({clause} AND {p}audience = 'family')"
    return clause, params


def scope_clause(model, capability: Capability):
    """SQLAlchemy predicate for models bearing owner_id/family_id columns."""
    from sqlalchemy import and_

    if capability.family_id is None:
        return and_(model.family_id.is_(None), model.owner_id == capability.user_id)
    return model.family_id == capability.family_id


# Tables with RLS enabled. EMPTY in phase 1 by design: legacy module queries
# don't set GUCs yet, and FORCE RLS would blank them. Each module phase
# appends its tables here in the same commit that adopts scoped queries.
RLS_TABLES: tuple[str, ...] = (
    "wiki_entries",
    "planner_projects",
    "meal_plans",
    "recipes",
    "shopping_lists",
    "pantry_items",
    "stores",
)


def rls_policy_sql(table: str) -> list[str]:
    """DDL enabling fail-closed family isolation on a root content table.

    current_setting(..., true) returns NULL (not an error) when the GUC is
    unset → no branch matches → zero rows. FORCE subjects even the table
    owner to the policy, so the app's role cannot silently bypass it.
    """
    predicate = (
        "((family_id IS NULL AND owner_id::text = current_setting('app.user_id', true)) "
        "OR (family_id IS NOT NULL AND family_id::text = current_setting('app.family_id', true)))"
    )
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        f"CREATE POLICY {table}_family_isolation ON {table} "
        f"USING {predicate} WITH CHECK {predicate}",
    ]


@asynccontextmanager
async def scoped_connection(pool, capability: Capability):
    """asyncpg connection inside a transaction with the RLS GUCs set.

    SET LOCAL via set_config(..., true): transaction-scoped, auto-reverts on
    commit/rollback — never leaks across pooled connections. All legacy
    raw-SQL module queries must run through this from phase 2 on.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.user_id', $1, true), set_config('app.family_id', $2, true)",
                str(capability.user_id),
                str(capability.family_id) if capability.family_id else "",
            )
            yield conn


async def assert_rls_enforced(pool) -> None:
    """Startup self-check: abort boot if any registered table lost FORCE RLS.

    Computes the set of registered tables that ARE present with RLS enabled
    and forced, then diffs RLS_TABLES against it -- rather than filtering for
    unenforced rows -- so a typo'd table name or an unrun migration (no
    pg_class row at all) shows up as missing too, instead of silently
    matching zero rows and letting boot proceed.
    """
    if not RLS_TABLES:
        return
    rows = await pool.fetch(
        "SELECT c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname = ANY($1::text[]) "
        "AND c.relrowsecurity AND c.relforcerowsecurity",
        list(RLS_TABLES),
    )
    enforced = {r["relname"] for r in rows}
    missing_or_unenforced = set(RLS_TABLES) - enforced
    if missing_or_unenforced:
        names = ", ".join(sorted(missing_or_unenforced))
        raise RuntimeError(f"FORCE ROW LEVEL SECURITY missing on: {names} — refusing to start")
