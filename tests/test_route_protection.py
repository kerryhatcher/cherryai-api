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
   signal recognizes that dependency by module + qualname.

   Module + qualname alone is *not* sufficient, though:
   ``Authenticator.current_user(optional=True)`` produces a closure with the
   exact same module/qualname while NOT enforcing auth (anonymous requests
   get ``user=None`` instead of a 401). No route in this codebase uses
   ``optional=True`` today, but the detector must not silently false-pass
   if one is ever added. Two checks are combined, matching whichever fires
   first:

   a. Identity against the two known-good dependency callables this
      codebase actually exports (``cherryai_api.auth.current_active_user``
      and ``current_verified_user``) — the strongest possible signal, immune
      to any fastapi-users internals changing.
   b. For any other module/qualname match (e.g. fastapi-users' own
      generated routers, which build their own ``current_user(...)``
      dependency internally rather than reusing ours), reach into the
      closure fastapi-users actually builds and read back the baked-in
      ``optional`` value, accepting only when it is verifiably ``False``.
      fastapi-users constructs this dependency via
      ``makefun.with_signature``, which returns a *wrapper* function (this
      is what appears in ``dependant.call``) whose real implementation —
      the one whose closure cells hold ``optional``/``active``/etc. — is
      stashed on the wrapper's ``__func_impl__`` attribute. That attribute
      is a `makefun` implementation detail, not a public fastapi-users API,
      so if it ever disappears or the closure layout changes, the check
      fails *closed* (returns ``False``, route reported as unprotected)
      rather than false-passing.

Route enumeration also has to go through ``fastapi.routing.iter_route_contexts``
rather than a plain scan of ``app.routes``: this project pins
``fastapi>=0.139.2``, where ``include_router`` no longer eagerly flattens
sub-router routes into ``app.routes`` as ``APIRoute`` instances — they are
kept behind an internal ``_IncludedRouter`` wrapper and only resolved to
their effective (fully-prefixed) form lazily. Without this, entire routers
(wiki, feedback, workflows, auth, users, admin) would be silently skipped by
the sweep instead of being checked. As a canary against that regression (or
any other enumeration shrink), the test also asserts a floor on how many
``APIRoute`` contexts it actually visited — under today's route table that
number is 35; the floor is set to 30 to leave headroom for minor route
changes while still catching a collapse back to the ~5 routes the naive
(pre-``iter_route_contexts``) version saw.
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
    """True for fastapi-users' ``Authenticator.current_user`` closure,
    provided it actually enforces authentication (``optional=False``).

    Catches routes (e.g. fastapi-users' own ``/users/me``) whose handler
    parameter is annotated with the library's unresolved TypeVar rather than
    the concrete ``User`` class, where `_is_user_annotated` cannot match.

    See the module docstring for why module/qualname alone is not enough
    (``optional=True`` closures share the same module/qualname) and why this
    function checks identity against this codebase's own dependencies first,
    falling back to closure inspection — reading the ``optional`` value
    fastapi-users bakes into the dependency via its ``makefun``-generated
    wrapper's ``__func_impl__`` attribute — only for other matches, and only
    accepting those verifiably built with ``optional=False``.
    """
    from cherryai_api.auth import current_active_user, current_verified_user

    if call is current_active_user or call is current_verified_user:
        return True

    if getattr(call, "__module__", "") != "fastapi_users.authentication.authenticator":
        return False
    if not getattr(call, "__qualname__", "").startswith("Authenticator.current_user"):
        return False

    impl = getattr(call, "__func_impl__", None)
    impl_code = getattr(impl, "__code__", None)
    if impl_code is None:
        return False
    for name, cell in zip(impl_code.co_freevars, impl.__closure__ or (), strict=True):
        if name == "optional":
            return not cell.cell_contents
    return False


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
    checked = 0
    for ctx in routing.iter_route_contexts(app.routes):
        route = ctx.original_route
        if not isinstance(route, APIRoute):
            continue
        checked += 1
        if ctx.path in _PUBLIC_PATHS:
            continue
        if not _depends_on_user(route):
            unprotected.append(f"{sorted(ctx.methods)} {ctx.path}")
    # Canary: under today's route table this is 35. If route flattening
    # regresses (see module docstring), enumeration silently shrinks to a
    # handful of routes and the assertion below would pass vacuously.
    assert checked >= 30, (
        f"only {checked} routes enumerated — route flattening may have changed; "
        "the sweep would pass vacuously"
    )
    assert unprotected == [], f"Unprotected routes: {unprotected}"


def test_detector_rejects_optional_current_user():
    """``current_user(optional=True)`` must not be mistaken for an
    auth-enforcing dependency merely because it shares module/qualname with
    the closures fastapi-users builds for ``current_active_user`` and
    ``current_verified_user``.
    """
    from cherryai_api.auth import fastapi_users_app

    optional_dep = fastapi_users_app.current_user(active=True, optional=True)
    assert not _is_authenticator_dependency(optional_dep)
