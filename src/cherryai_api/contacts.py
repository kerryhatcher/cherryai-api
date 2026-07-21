"""Contacts integration: Fastmail CardDAV address books, contacts, and groups.

Mirrors ``email.py`` / ``calendar.py``: pydantic response models, data access
functions that call the SDK, and a FastAPI router mounted under
``/api/contacts``. The chat agent reuses the data access functions for its
contact-lookup tools.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastmail_sdk import CardDavClient
from fastmail_sdk.errors import FastmailError
from fastmail_sdk.models.contacts import (
    Contact as SdkContact,
)
from fastmail_sdk.models.contacts import (
    ContactEmail,
    ContactPhone,
)
from fastmail_sdk.models.contacts import (
    ContactGroup as SdkContactGroup,
)
from pydantic import BaseModel, Field

from cherryai_api.auth import current_verified_user
from cherryai_api.users import User

router = APIRouter(prefix="/api/contacts", tags=["contacts"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ContactEmailOut(BaseModel):
    email: str
    label: str | None = None


class ContactPhoneOut(BaseModel):
    number: str
    label: str | None = None


class ContactOut(BaseModel):
    id: str
    name: str
    emails: list[ContactEmailOut] = []
    phones: list[ContactPhoneOut] = []
    organization: str | None = None
    title: str | None = None
    notes: str | None = None
    address: str | None = None


class ContactGroupOut(BaseModel):
    id: str
    name: str
    member_uids: list[str] = []
    member_count: int = 0


class ContactCreateIn(BaseModel):
    name: str = Field(min_length=1)
    emails: list[ContactEmailOut] = []
    phones: list[ContactPhoneOut] = []
    organization: str | None = None
    title: str | None = None
    notes: str | None = None
    address: str | None = None


class ContactUpdateIn(BaseModel):
    name: str | None = None
    emails: list[ContactEmailOut] | None = None
    phones: list[ContactPhoneOut] | None = None
    organization: str | None = None
    title: str | None = None
    notes: str | None = None
    address: str | None = None


class GroupCreateIn(BaseModel):
    name: str = Field(min_length=1)


class GroupRenameIn(BaseModel):
    name: str = Field(min_length=1)


class MemberIn(BaseModel):
    contact_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_client(
    username: str | None = None,
    app_password: str | None = None,
) -> CardDavClient:
    """Build a CardDAV client.

    If username and app_password are provided (per-user credentials), use those.
    Otherwise fall back to system-level credentials.
    """
    from fastmail_sdk.config import load_credentials

    if username and app_password:
        return CardDavClient(username=username, app_password=app_password)

    try:
        username, app_password = load_credentials()
    except Exception as error:
        raise FastmailError(
            "Fastmail credentials not found. "
            "Set FASTMAIL_USERNAME and FASTMAIL_APP_PASSWORD, "
            "or add a [calendar] section to ~/.config/fastmail-cli/config.toml."
        ) from error
    return CardDavClient(username=username, app_password=app_password)


async def _resolve_user_creds(user_id: uuid.UUID) -> tuple[str | None, str | None]:
    """Return (username, app_password) for a user's active Fastmail credential."""
    from cherryai_api.integrations import decrypt_credential, get_active_credential_for_user
    from cherryai_api.orm import async_session_maker

    async with async_session_maker() as session:
        cred = await get_active_credential_for_user(session, user_id)
        if cred is None:
            return None, None
        username, app_password, _api_token = decrypt_credential(cred)
        return username, app_password


def _to_contact_out(c: SdkContact) -> ContactOut:
    return ContactOut(
        id=c.id,
        name=c.name,
        emails=[ContactEmailOut(email=e.email, label=e.label) for e in c.emails],
        phones=[ContactPhoneOut(number=p.number, label=p.label) for p in c.phones],
        organization=c.organization,
        title=c.title,
        notes=c.notes,
        address=c.address,
    )


def _to_group_out(g: SdkContactGroup) -> ContactGroupOut:
    return ContactGroupOut(
        id=g.id,
        name=g.name,
        member_uids=g.member_uids,
        member_count=len(g.member_uids),
    )


# ---------------------------------------------------------------------------
# Data access (used by both the router and the agent tools)
# ---------------------------------------------------------------------------


async def list_contacts(
    user_id: uuid.UUID | None = None,
) -> list[ContactOut]:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        addressbooks = await client.list_addressbooks()
        all_contacts: list[SdkContact] = []
        for ab in addressbooks:
            try:
                contacts = await client.list_contacts(ab.href)
                all_contacts.extend(contacts)
            except Exception:
                pass
    # Sort by name
    all_contacts.sort(key=lambda c: c.name.lower())
    return [_to_contact_out(c) for c in all_contacts]


async def search_contacts(
    query: str,
    user_id: uuid.UUID | None = None,
) -> list[ContactOut]:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        results = await client.search_contacts(query)
    results.sort(key=lambda c: c.name.lower())
    return [_to_contact_out(c) for c in results]


async def get_contact(
    contact_id: str,
    user_id: uuid.UUID | None = None,
) -> ContactOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        contact = await client.get_contact_by_id(contact_id)
    return _to_contact_out(contact)


async def create_contact(
    data: ContactCreateIn,
    user_id: uuid.UUID | None = None,
) -> ContactOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        addressbooks = await client.list_addressbooks()
        if not addressbooks:
            raise HTTPException(status_code=400, detail="No address books found")
        ab = addressbooks[0]

        contact = SdkContact(
            id=str(uuid.uuid4()),
            name=data.name,
            emails=[ContactEmail(email=e.email, label=e.label) for e in data.emails],
            phones=[ContactPhone(number=p.number, label=p.label) for p in data.phones],
            organization=data.organization,
            title=data.title,
            notes=data.notes,
            address=data.address,
        )
        result = await client.create_contact(ab.href, contact)
        contact.href = result.href
        contact.etag = result.etag
    return _to_contact_out(contact)


async def update_contact(
    contact_id: str,
    data: ContactUpdateIn,
    user_id: uuid.UUID | None = None,
) -> ContactOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        existing = await client.get_contact_by_id(contact_id)
        if not existing.href or not existing.etag:
            raise HTTPException(status_code=409, detail="Contact has no href/etag — cannot update")

        updated = SdkContact(
            id=existing.id,
            name=data.name if data.name is not None else existing.name,
            emails=(
                [ContactEmail(email=e.email, label=e.label) for e in data.emails]
                if data.emails is not None
                else existing.emails
            ),
            phones=(
                [ContactPhone(number=p.number, label=p.label) for p in data.phones]
                if data.phones is not None
                else existing.phones
            ),
            organization=(
                data.organization if data.organization is not None else existing.organization
            ),
            title=data.title if data.title is not None else existing.title,
            notes=data.notes if data.notes is not None else existing.notes,
            address=data.address if data.address is not None else existing.address,
            href=existing.href,
            etag=existing.etag,
        )
        new_etag = await client.update_contact(existing.href, existing.etag, updated)
        updated.etag = new_etag
    return _to_contact_out(updated)


async def delete_contact(
    contact_id: str,
    user_id: uuid.UUID | None = None,
) -> None:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        contact = await client.get_contact_by_id(contact_id)
        if not contact.href or not contact.etag:
            raise HTTPException(status_code=409, detail="Contact has no href/etag — cannot delete")
        await client.delete_contact(contact.href, contact.etag, contact_id)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


async def list_groups(
    user_id: uuid.UUID | None = None,
) -> list[ContactGroupOut]:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        groups = await client.list_groups()
    groups.sort(key=lambda g: g.name.lower())
    return [_to_group_out(g) for g in groups]


async def create_group(
    data: GroupCreateIn,
    user_id: uuid.UUID | None = None,
) -> ContactGroupOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        addressbooks = await client.list_addressbooks()
        if not addressbooks:
            raise HTTPException(status_code=400, detail="No address books found")
        ab = addressbooks[0]

        group = SdkContactGroup(id=str(uuid.uuid4()), name=data.name)
        result = await client.create_group(ab.href, group)
        group.href = result.href
        group.etag = result.etag
    return _to_group_out(group)


async def rename_group(
    group_id: str,
    data: GroupRenameIn,
    user_id: uuid.UUID | None = None,
) -> ContactGroupOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        group = await client.get_group_by_id(group_id)
        if not group.href or not group.etag:
            raise HTTPException(status_code=409, detail="Group has no href/etag")
        new_etag = await client.rename_group(group.href, group.etag, group, data.name)
        group.name = data.name
        group.etag = new_etag
    return _to_group_out(group)


async def delete_group(
    group_id: str,
    user_id: uuid.UUID | None = None,
) -> None:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        group = await client.get_group_by_id(group_id)
        if not group.href or not group.etag:
            raise HTTPException(status_code=409, detail="Group has no href/etag")
        await client.delete_group(group.href, group.etag, group_id)


async def add_group_member(
    group_id: str,
    contact_id: str,
    user_id: uuid.UUID | None = None,
) -> ContactGroupOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        group = await client.get_group_by_id(group_id)
        if not group.href or not group.etag:
            raise HTTPException(status_code=409, detail="Group has no href/etag")
        new_etag = await client.add_group_member(group.href, group.etag, group, contact_id)
        group.member_uids.append(contact_id)
        group.etag = new_etag
    return _to_group_out(group)


async def remove_group_member(
    group_id: str,
    contact_id: str,
    user_id: uuid.UUID | None = None,
) -> ContactGroupOut:
    username, app_password = await _resolve_user_creds(user_id) if user_id else (None, None)
    client = _build_client(username=username, app_password=app_password)
    async with client:
        group = await client.get_group_by_id(group_id)
        if not group.href or not group.etag:
            raise HTTPException(status_code=409, detail="Group has no href/etag")
        new_etag = await client.remove_group_member(group.href, group.etag, group, contact_id)
        group.member_uids = [uid for uid in group.member_uids if uid != contact_id]
        group.etag = new_etag
    return _to_group_out(group)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ContactOut])
async def list_contacts_route(
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[ContactOut]:
    try:
        return await list_contacts(user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/search", response_model=list[ContactOut])
async def search_contacts_route(
    q: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[ContactOut]:
    try:
        return await search_contacts(q, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/groups", response_model=list[ContactGroupOut])
async def list_groups_route(
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[ContactGroupOut]:
    try:
        return await list_groups(user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/groups", response_model=ContactGroupOut, status_code=201)
async def create_group_route(
    body: GroupCreateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactGroupOut:
    try:
        return await create_group(body, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.put("/groups/{group_id}", response_model=ContactGroupOut)
async def rename_group_route(
    group_id: str,
    body: GroupRenameIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactGroupOut:
    try:
        return await rename_group(group_id, body, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group_route(
    group_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    try:
        await delete_group(group_id, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/groups/{group_id}/members", response_model=ContactGroupOut)
async def add_group_member_route(
    group_id: str,
    body: MemberIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactGroupOut:
    try:
        return await add_group_member(group_id, body.contact_id, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.delete("/groups/{group_id}/members/{contact_id}", response_model=ContactGroupOut)
async def remove_group_member_route(
    group_id: str,
    contact_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactGroupOut:
    try:
        return await remove_group_member(group_id, contact_id, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/{contact_id}", response_model=ContactOut)
async def get_contact_route(
    contact_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactOut:
    try:
        return await get_contact(contact_id, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("", response_model=ContactOut, status_code=201)
async def create_contact_route(
    body: ContactCreateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactOut:
    try:
        return await create_contact(body, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.put("/{contact_id}", response_model=ContactOut)
async def update_contact_route(
    contact_id: str,
    body: ContactUpdateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> ContactOut:
    try:
        return await update_contact(contact_id, body, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.delete("/{contact_id}", status_code=204)
async def delete_contact_route(
    contact_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    try:
        await delete_contact(contact_id, user_id=user.id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
