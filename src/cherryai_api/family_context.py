"""Active-family request context.

Raw ASGI middleware (NOT BaseHTTPMiddleware — it breaks contextvar
propagation into streaming responses; research doc: family-authz-multitenancy)
parses the active-family id from header or cookie into a ContextVar. It does
NOT validate membership — that needs the DB and happens in
``authz.get_capability``, which falls back to personal context when the id
isn't one of the caller's families.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

ACTIVE_FAMILY_COOKIE = "cherryai_family"
ACTIVE_FAMILY_HEADER = "X-CherryAI-Family"

active_family_var: ContextVar[uuid.UUID | None] = ContextVar("active_family", default=None)
current_user_var: ContextVar[uuid.UUID | None] = ContextVar("current_user", default=None)


def _cookie_value(cookie_header: str, name: str) -> str | None:
    for part in cookie_header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value or None
    return None


class FamilyContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        raw = headers.get(ACTIVE_FAMILY_HEADER) or _cookie_value(
            headers.get("cookie", ""), ACTIVE_FAMILY_COOKIE
        )
        family_id: uuid.UUID | None = None
        if raw:
            try:
                family_id = uuid.UUID(raw)
            except ValueError:
                family_id = None  # silent fallback to personal (spec §7)
        token_f = active_family_var.set(family_id)
        token_u = current_user_var.set(None)
        try:
            await self.app(scope, receive, send)
        finally:
            active_family_var.reset(token_f)
            current_user_var.reset(token_u)
