"""Family authorization: the whole permission model in one place.

Scope strings are OAuth-shaped (spec §1) so a future MCP/on-behalf-of
mapping is a relabeling, not a redesign.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from cherryai_api.families import (
    FAMILY_ROLE_ADMIN,
    FAMILY_ROLE_CHILD,
    FAMILY_ROLE_ORGANIZER,
    PERM_EDIT,
    PERM_VIEW,
)

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
