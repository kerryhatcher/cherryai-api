"""Per-user Fastmail credentials: SQLAlchemy model, encryption, and API router.

Credentials are encrypted at rest with Fernet (symmetric, keyed by auth_secret).
The router provides CRUD endpoints plus a validation endpoint that tests the
credentials against Fastmail's JMAP API before saving.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.auth import current_verified_user
from cherryai_api.orm import Base, get_async_session
from cherryai_api.settings import get_settings
from cherryai_api.users import User

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


def _get_fernet() -> Fernet:
    """Derive a Fernet key from the auth secret (must be 32 url-safe b64 bytes)."""
    import base64
    import hashlib

    secret = get_settings().auth_secret
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _encrypt(plain: str) -> str:
    return _get_fernet().encrypt(plain.encode()).decode()


def _decrypt(token: str) -> str:
    return _get_fernet().decrypt(token.encode()).decode()


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------


class UserFastmailCredential(Base):
    __tablename__ = "user_fastmail_credentials"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    label: Mapped[str] = mapped_column(String, nullable=False, default="Fastmail")
    username: Mapped[str] = mapped_column(String, nullable=False)
    app_password_encrypted: Mapped[str] = mapped_column(String, nullable=False)
    api_token_encrypted: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class FastmailCredentialIn(BaseModel):
    """Request body for creating/updating Fastmail credentials."""

    label: str = "Fastmail"
    username: str = Field(min_length=1, description="Fastmail email address")
    app_password: str = Field(min_length=1, description="Fastmail app password")
    api_token: str | None = Field(
        default=None, description="JMAP API token (optional; app password covers JMAP too)"
    )


class FastmailCredentialOut(BaseModel):
    """Response body — secrets are never returned."""

    id: uuid.UUID
    label: str
    username: str
    has_api_token: bool = False
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FastmailCredentialUpdateIn(BaseModel):
    label: str | None = None
    username: str | None = None
    app_password: str | None = None
    api_token: str | None = None
    is_active: bool | None = None


class ValidateResult(BaseModel):
    ok: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_credential(
    session: AsyncSession, credential_id: uuid.UUID, user_id: uuid.UUID
) -> UserFastmailCredential:
    row = (
        await session.execute(
            select(UserFastmailCredential).where(
                UserFastmailCredential.id == credential_id,
                UserFastmailCredential.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return row


async def get_active_credential_for_user(
    session: AsyncSession, user_id: uuid.UUID
) -> UserFastmailCredential | None:
    """Return the user's active Fastmail credential, or None."""
    row = (
        await session.execute(
            select(UserFastmailCredential).where(
                UserFastmailCredential.user_id == user_id,
                UserFastmailCredential.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()
    return row


def decrypt_credential(cred: UserFastmailCredential) -> tuple[str, str, str | None]:
    """Return (username, app_password, api_token_or_none) from an encrypted row."""
    return (
        cred.username,
        _decrypt(cred.app_password_encrypted),
        _decrypt(cred.api_token_encrypted) if cred.api_token_encrypted else None,
    )


# ---------------------------------------------------------------------------
# Validation (test credentials against Fastmail)
# ---------------------------------------------------------------------------


async def _validate_fastmail_creds(username: str, app_password: str) -> tuple[bool, str]:
    """Try to connect to Fastmail JMAP with the given credentials."""
    try:
        from fastmail_sdk import JmapClient
        from fastmail_sdk.errors import FastmailError

        client = JmapClient(token=app_password, username=username)
        # A lightweight call to verify the credentials work.
        mailboxes = await client.get_mailboxes()
        if mailboxes:
            return True, f"Connected — {len(mailboxes)} mailbox(es) found"
        return True, "Connected successfully"
    except FastmailError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Connection failed: {e}"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@router.get("/fastmail", response_model=list[FastmailCredentialOut])
async def list_fastmail_credentials(
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> list[FastmailCredentialOut]:
    rows = (
        (
            await session.execute(
                select(UserFastmailCredential)
                .where(UserFastmailCredential.user_id == user.id)
                .order_by(UserFastmailCredential.created_at)
            )
        )
        .scalars()
        .all()
    )
    return [
        FastmailCredentialOut(
            id=r.id,
            label=r.label,
            username=r.username,
            has_api_token=r.api_token_encrypted is not None,
            is_active=r.is_active,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@router.post("/fastmail", response_model=FastmailCredentialOut, status_code=201)
async def create_fastmail_credential(
    body: FastmailCredentialIn,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> FastmailCredentialOut:
    # Validate before saving.
    ok, detail = await _validate_fastmail_creds(body.username, body.app_password)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Credential validation failed: {detail}")

    cred = UserFastmailCredential(
        user_id=user.id,
        label=body.label,
        username=body.username,
        app_password_encrypted=_encrypt(body.app_password),
        api_token_encrypted=_encrypt(body.api_token) if body.api_token else None,
    )
    session.add(cred)
    await session.commit()
    await session.refresh(cred)
    return FastmailCredentialOut(
        id=cred.id,
        label=cred.label,
        username=cred.username,
        has_api_token=cred.api_token_encrypted is not None,
        is_active=cred.is_active,
        created_at=cred.created_at,
        updated_at=cred.updated_at,
    )


@router.post("/fastmail/validate", response_model=ValidateResult)
async def validate_fastmail_credential(
    body: FastmailCredentialIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ValidateResult:
    """Test credentials without saving them."""
    ok, detail = await _validate_fastmail_creds(body.username, body.app_password)
    return ValidateResult(ok=ok, detail=detail)


@router.put("/fastmail/{credential_id}", response_model=FastmailCredentialOut)
async def update_fastmail_credential(
    credential_id: uuid.UUID,
    body: FastmailCredentialUpdateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> FastmailCredentialOut:
    cred = await _get_credential(session, credential_id, user.id)

    if body.label is not None:
        cred.label = body.label
    if body.username is not None:
        cred.username = body.username
    if body.app_password is not None:
        # Validate new password before saving.
        username = body.username or cred.username
        ok, detail = await _validate_fastmail_creds(username, body.app_password)
        if not ok:
            raise HTTPException(status_code=400, detail=f"Credential validation failed: {detail}")
        cred.app_password_encrypted = _encrypt(body.app_password)
    if body.api_token is not None:
        cred.api_token_encrypted = _encrypt(body.api_token) if body.api_token else None
    if body.is_active is not None:
        cred.is_active = body.is_active

    await session.commit()
    await session.refresh(cred)
    return FastmailCredentialOut(
        id=cred.id,
        label=cred.label,
        username=cred.username,
        has_api_token=cred.api_token_encrypted is not None,
        is_active=cred.is_active,
        created_at=cred.created_at,
        updated_at=cred.updated_at,
    )


@router.delete("/fastmail/{credential_id}", status_code=204)
async def delete_fastmail_credential(
    credential_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
    session: AsyncSession = Depends(get_async_session),  # noqa: B008
) -> None:
    cred = await _get_credential(session, credential_id, user.id)
    await session.delete(cred)
    await session.commit()
