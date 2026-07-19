"""Every data route must reject anonymous requests.

Guards against a future route being added without auth: walk the real
app's route table and assert everything outside the public allowlist
depends (directly or transitively) on an authenticated user.

Two detection signals are needed, not just a literal `User` annotation
match, to actually work against this codebase:

1. ``inspect.signature(call, eval_str=True)`` resolves string annotations
   back to real classes. Several route modules (``wiki.py``, ``feedback.py``,
   ``workflows.py``) use ``from __future__ import annotations`` (PEP 563),
   which stringifies every annotation at import time; a plain
   ``inspect.signature(call)`` would see the literal string ``"User"``
   rather than the ``User`` class and silently fail to match.
2. fastapi-users' generated routers (``/users/me``, ``/users/{id}``, ...)
   build their handlers generically, so the ``user`` parameter is annotated
   with the library's internal TypeVar (``~UP``), never resolved to the
   concrete ``User`` class — not even with ``eval_str=True``. Those routes
   are still genuinely auth-protected (they depend on fastapi-users'
   ``Authenticator.current_user`` closure under the hood), so a second
   signal recognizes that dependency by module + qualname regardless of its
   parameter annotations.

Route enumeration also has to go through ``fastapi.routing.iter_route_contexts``
rather than a plain scan of ``app.routes``: this project pins
``fastapi>=0.139.2``, where ``include_router`` no longer eagerly flattens
sub-router routes into ``app.routes`` as ``APIRoute`` instances — they are
kept behind an internal ``_IncludedRouter`` wrapper and only resolved to
their effective (fully-prefixed) form lazily. Without this, entire routers
(wiki, feedback, workflows, auth, users, admin) would be silently skipped by
the sweep instead of being checked.
"""

import inspect

from fastapi import routing
from fastapi.routing import APIRoute

_PUBLIC_PATHS = {
    "/api/health",
    "/auth/login",
    "/auth/logout",
    "/auth/register",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/docs/oauth2-redirect",
}


def _is_user_annotated(call) -> bool:
    from cherryai_api.users import User

    try:
        sig = inspect.signature(call, eval_str=True)
    except Exception:
        sig = inspect.signature(call)
    return any(p.annotation is User for p in sig.parameters.values())


def _is_authenticator_dependency(call) -> bool:
    """True for fastapi-users' ``Authenticator.current_user`` closure.

    Catches routes (e.g. fastapi-users' own ``/users/me``) whose handler
    parameter is annotated with the library's unresolved TypeVar rather than
    the concrete ``User`` class, where `_is_user_annotated` cannot match.
    """
    return getattr(
        call, "__module__", ""
    ) == "fastapi_users.authentication.authenticator" and getattr(
        call, "__qualname__", ""
    ).startswith("Authenticator.current_user")


def _depends_on_user(route: APIRoute) -> bool:
    seen = set()

    def walk(dependant) -> bool:
        if id(dependant) in seen:
            return False
        seen.add(id(dependant))
        for param in dependant.dependencies:
            if walk(param):
                return True
        call = getattr(dependant, "call", None)
        if call is not None and (_is_user_annotated(call) or _is_authenticator_dependency(call)):
            return True
        return False

    return walk(route.dependant)


def test_all_routes_require_auth_except_allowlist():
    from cherryai_api.api import app

    unprotected = []
    for ctx in routing.iter_route_contexts(app.routes):
        route = ctx.original_route
        if not isinstance(route, APIRoute):
            continue
        if ctx.path in _PUBLIC_PATHS:
            continue
        if not _depends_on_user(route):
            unprotected.append(f"{sorted(ctx.methods)} {ctx.path}")
    assert unprotected == [], f"Unprotected routes: {unprotected}"
