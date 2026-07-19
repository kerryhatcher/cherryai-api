"""Authentication wiring: cookie transport + database-backed tokens.

Server-side tokens make deactivation immediate: deleting a user's
accesstoken rows logs them out everywhere. The SPA never handles a token —
the browser carries an httpOnly cookie.
"""

import uuid

from fastapi import Depends, HTTPException
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from fastapi_users.authentication.strategy.db import (
    AccessTokenDatabase,
    DatabaseStrategy,
)

from cherryai_api.settings import get_settings
from cherryai_api.users import (
    ROLE_ADMIN,
    ROLE_CHAT,
    AccessToken,
    User,
    get_access_token_db,
    get_user_manager,
)

_settings = get_settings()

cookie_transport = CookieTransport(
    cookie_name="cherryai_auth",
    cookie_max_age=_settings.auth_token_lifetime_seconds,
    cookie_secure=_settings.auth_cookie_secure,
    cookie_samesite="lax",
)


def get_database_strategy(
    access_token_db: AccessTokenDatabase[AccessToken] = Depends(get_access_token_db),  # noqa: B008
) -> DatabaseStrategy:
    return DatabaseStrategy(access_token_db, lifetime_seconds=_settings.auth_token_lifetime_seconds)


auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_database_strategy,
)

fastapi_users_app = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# Pending users may call /users/me (to see their own is_verified=False);
# everything else demands verified=True.
current_active_user = fastapi_users_app.current_user(active=True)
current_verified_user = fastapi_users_app.current_user(active=True, verified=True)


def require_chat(user: User = Depends(current_verified_user)) -> User:  # noqa: B008
    """Chat capability: admins and chat users; restricted users are refused."""
    if user.role not in (ROLE_ADMIN, ROLE_CHAT):
        raise HTTPException(status_code=403, detail="Chat is not enabled for this account")
    return user


def require_admin(user: User = Depends(current_verified_user)) -> User:  # noqa: B008
    if user.role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
