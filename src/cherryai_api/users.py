"""User identity: SQLAlchemy models, fastapi-users schemas, and the manager.

Flag semantics (see the multi-user design spec):
- ``is_verified=False`` — registered, pending admin approval.
- ``is_verified=True``  — approved.
- ``is_active=False``   — deactivated by an admin.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import Depends
from fastapi_users import BaseUserManager, UUIDIDMixin, schemas
from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseUserTableUUID,
    SQLAlchemyUserDatabase,
)
from fastapi_users_db_sqlalchemy.access_token import (
    SQLAlchemyAccessTokenDatabase,
    SQLAlchemyBaseAccessTokenTableUUID,
)
from loguru import logger
from sqlalchemy import DateTime, String, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.orm import Base, get_async_session
from cherryai_api.settings import get_settings

ROLE_ADMIN = "admin"
ROLE_CHAT = "chat"
ROLE_RESTRICTED = "restricted"
ROLES = (ROLE_ADMIN, ROLE_CHAT, ROLE_RESTRICTED)


class User(SQLAlchemyBaseUserTableUUID, Base):
    """A CherryAI account. Core auth columns come from the fastapi-users base."""

    role: Mapped[str] = mapped_column(String, nullable=False, default=ROLE_CHAT)
    display_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    # Cognee dataset name. Empty at insert; filled with "user-<id>" in
    # on_after_register (the id does not exist before the row does). The
    # bootstrap admin instead inherits the legacy configured dataset.
    memory_dataset: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    """Server-side auth token row; deleting it revokes the session."""


class UserRead(schemas.BaseUser[uuid.UUID]):
    role: str
    display_name: str


class UserCreate(schemas.BaseUserCreate):
    display_name: str = ""


class UserUpdate(schemas.BaseUserUpdate):
    display_name: str | None = None


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """fastapi-users manager; hooks assign the per-user memory dataset."""

    def __init__(self, user_db: SQLAlchemyUserDatabase) -> None:
        super().__init__(user_db)
        secret = get_settings().auth_secret
        self.reset_password_token_secret = secret
        self.verification_token_secret = secret

    async def on_after_register(self, user: User, request=None) -> None:
        if not user.memory_dataset:
            user = await self.user_db.update(user, {"memory_dataset": f"user-{user.id}"})
        logger.info(f"User registered (pending approval): {user.email}")


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> AsyncIterator[SQLAlchemyUserDatabase]:
    yield SQLAlchemyUserDatabase(session, User)


async def get_access_token_db(
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> AsyncIterator[SQLAlchemyAccessTokenDatabase]:
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),  # noqa: B008
) -> AsyncIterator[UserManager]:
    yield UserManager(user_db)
