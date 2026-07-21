"""Email integration: Fastmail JMAP mailboxes, emails, and sending.

Mirrors ``calendar.py``: pydantic response models, data access functions that
call the SDK, and a FastAPI router mounted under ``/api/emails``. The chat
agent reuses the data access functions for its email tools.

Key difference from Calendar: agent-initiated sends land in an approval queue
(``email_approvals`` table) instead of being sent immediately. A human must
review and approve before the email is actually dispatched.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastmail_sdk import JmapClient
from fastmail_sdk.errors import FastmailError
from fastmail_sdk.models.email import EmailAddress, SearchFilter
from pydantic import BaseModel

from cherryai_api.auth import current_verified_user
from cherryai_api.settings import get_settings
from cherryai_api.users import User

# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------


class MailboxOut(BaseModel):
    id: str
    name: str
    role: str | None = None
    total_emails: int = 0
    unread_emails: int = 0


class EmailAddressOut(BaseModel):
    name: str | None = None
    email: str


class EmailOut(BaseModel):
    id: str
    thread_id: str | None = None
    subject: str | None = None
    from_: list[EmailAddressOut] = []
    to: list[EmailAddressOut] = []
    cc: list[EmailAddressOut] = []
    received_at: str | None = None
    preview: str | None = None
    has_attachment: bool = False
    is_unread: bool = False
    mailbox_ids: dict[str, bool] = {}


class EmailDetailOut(BaseModel):
    id: str
    thread_id: str | None = None
    subject: str | None = None
    from_: list[EmailAddressOut] = []
    to: list[EmailAddressOut] = []
    cc: list[EmailAddressOut] = []
    bcc: list[EmailAddressOut] = []
    received_at: str | None = None
    sent_at: str | None = None
    preview: str | None = None
    has_attachment: bool = False
    is_unread: bool = False
    text_body: str | None = None
    html_body: str | None = None
    message_id: list[str] | None = None
    in_reply_to: list[str] | None = None
    references: list[str] | None = None


class ThreadEmailOut(BaseModel):
    id: str
    subject: str | None = None
    from_: list[EmailAddressOut] = []
    to: list[EmailAddressOut] = []
    received_at: str | None = None
    preview: str | None = None
    text_body: str | None = None
    is_unread: bool = False


class EmailSendIn(BaseModel):
    to: list[str]
    subject: str
    body: str
    cc: list[str] = []
    bcc: list[str] = []
    from_email: str | None = None
    in_reply_to: str | None = None


class EmailReplyIn(BaseModel):
    body: str
    reply_all: bool = False
    cc: list[str] = []
    bcc: list[str] = []
    from_email: str | None = None


# ------------------------------------------------------------------
# Approval queue models
# ------------------------------------------------------------------


class EmailApprovalOut(BaseModel):
    id: uuid.UUID
    status: str  # pending, approved, rejected, sent
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    to_: list[str] = []
    cc: list[str] = []
    bcc: list[str] = []
    from_email: str | None = None
    subject: str
    body: str
    in_reply_to: str | None = None
    agent_session_id: str | None = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_client() -> JmapClient:
    """Build a JMAP client from env, settings, or the shared config file.

    Precedence:
    1. ``FASTMAIL_API_TOKEN`` env var
    2. ``fastmail_api_token`` in app settings (.env)
    3. ``[core].api_token`` in ``~/.config/fastmail-cli/config.toml``
    """
    import os
    from pathlib import Path

    token = os.environ.get("FASTMAIL_API_TOKEN")
    if not token:
        token = get_settings().fastmail_api_token
    if not token:
        config_file = Path.home() / ".config" / "fastmail-cli" / "config.toml"
        if config_file.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            data = tomllib.loads(config_file.read_text())
            token = data.get("core", {}).get("api_token", "")
    if not token:
        raise FastmailError(
            "Fastmail API token not configured. "
            "Set FASTMAIL_API_TOKEN in your .env file, "
            "or add [core].api_token to ~/.config/fastmail-cli/config.toml."
        )
    return JmapClient(token=token)


def _to_mailbox_out(mb) -> MailboxOut:
    return MailboxOut(
        id=mb.id,
        name=mb.name,
        role=mb.role,
        total_emails=mb.total_emails,
        unread_emails=mb.unread_emails,
    )


def _to_email_out(email) -> EmailOut:
    return EmailOut(
        id=email.id,
        thread_id=email.thread_id,
        subject=email.subject,
        from_=[EmailAddressOut(name=a.name, email=a.email) for a in (email.from_ or [])],
        to=[EmailAddressOut(name=a.name, email=a.email) for a in (email.to or [])],
        cc=[EmailAddressOut(name=a.name, email=a.email) for a in (email.cc or [])],
        received_at=email.received_at,
        preview=email.preview,
        has_attachment=email.has_attachment,
        is_unread=email.is_unread(),
        mailbox_ids=email.mailbox_ids,
    )


def _to_email_detail_out(email) -> EmailDetailOut:
    return EmailDetailOut(
        id=email.id,
        thread_id=email.thread_id,
        subject=email.subject,
        from_=[EmailAddressOut(name=a.name, email=a.email) for a in (email.from_ or [])],
        to=[EmailAddressOut(name=a.name, email=a.email) for a in (email.to or [])],
        cc=[EmailAddressOut(name=a.name, email=a.email) for a in (email.cc or [])],
        bcc=[EmailAddressOut(name=a.name, email=a.email) for a in (email.bcc or [])],
        received_at=email.received_at,
        sent_at=email.sent_at,
        preview=email.preview,
        has_attachment=email.has_attachment,
        is_unread=email.is_unread(),
        text_body=email.text_content(),
        html_body=email.html_content(),
        message_id=email.message_id,
        in_reply_to=email.in_reply_to,
        references=email.references,
    )


# ------------------------------------------------------------------
# Data access (used by both the router and the agent tools)
# ------------------------------------------------------------------


async def list_mailboxes(request: Request | None = None) -> list[MailboxOut]:
    client = _build_client()
    async with client:
        await client.authenticate()
        mailboxes = await client.list_mailboxes()
    return [_to_mailbox_out(m) for m in mailboxes]


async def list_emails(
    request: Request | None = None,
    mailbox_id: str | None = None,
    limit: int = 50,
) -> list[EmailOut]:
    client = _build_client()
    async with client:
        await client.authenticate()
        if mailbox_id:
            emails = await client.list_emails(mailbox_id, limit=limit)
        else:
            # Default to inbox
            inbox = await client.find_mailbox("inbox")
            emails = await client.list_emails(inbox.id, limit=limit)
    return [_to_email_out(e) for e in emails]


async def get_email(
    request: Request | None = None,
    email_id: str = "",
) -> EmailDetailOut:
    client = _build_client()
    async with client:
        await client.authenticate()
        email = await client.get_email(email_id)
    return _to_email_detail_out(email)


async def get_thread(
    request: Request | None = None,
    email_id: str = "",
) -> list[ThreadEmailOut]:
    client = _build_client()
    async with client:
        await client.authenticate()
        emails = await client.get_thread(email_id)
    # Sort by received_at ascending for thread view
    emails.sort(key=lambda e: e.received_at or "")
    return [
        ThreadEmailOut(
            id=e.id,
            subject=e.subject,
            from_=[EmailAddressOut(name=a.name, email=a.email) for a in (e.from_ or [])],
            to=[EmailAddressOut(name=a.name, email=a.email) for a in (e.to or [])],
            received_at=e.received_at,
            preview=e.preview,
            text_body=e.text_content(),
            is_unread=e.is_unread(),
        )
        for e in emails
    ]


async def search_emails(
    request: Request | None = None,
    query: str = "",
    mailbox_id: str | None = None,
    limit: int = 50,
) -> list[EmailOut]:
    client = _build_client()
    async with client:
        await client.authenticate()
        filt = SearchFilter(text=query)
        emails = await client.search_emails(filt, mailbox_id=mailbox_id, limit=limit)
    return [_to_email_out(e) for e in emails]


async def send_email_direct(
    request: Request | None = None,
    data: EmailSendIn | None = None,
) -> str:
    """Send an email immediately (human-initiated). Returns the email ID."""
    client = _build_client()
    async with client:
        await client.authenticate()
        to_addrs = [EmailAddress(email=addr) for addr in data.to]
        cc_addrs = [EmailAddress(email=addr) for addr in data.cc]
        bcc_addrs = [EmailAddress(email=addr) for addr in data.bcc]
        email_id = await client.send_email(
            to=to_addrs,
            subject=data.subject,
            body=data.body,
            cc=cc_addrs if cc_addrs else None,
            bcc=bcc_addrs if bcc_addrs else None,
            from_email=data.from_email,
            in_reply_to=data.in_reply_to,
        )
    return email_id


async def reply_email_direct(
    request: Request | None = None,
    email_id: str = "",
    data: EmailReplyIn | None = None,
) -> str:
    """Reply to an email immediately (human-initiated). Returns the new email ID."""
    client = _build_client()
    async with client:
        await client.authenticate()
        original = await client.get_email(email_id)
        cc_addrs = [EmailAddress(email=addr) for addr in (data.cc or [])]
        bcc_addrs = [EmailAddress(email=addr) for addr in (data.bcc or [])]
        new_id = await client.reply_email(
            original=original,
            body=data.body,
            reply_all=data.reply_all,
            cc=cc_addrs if cc_addrs else None,
            bcc=bcc_addrs if bcc_addrs else None,
            from_email=data.from_email,
        )
    return new_id


# ------------------------------------------------------------------
# Approval queue data access
# ------------------------------------------------------------------

CREATE_EMAIL_APPROVALS_TABLE = """
CREATE TABLE IF NOT EXISTS email_approvals (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by UUID,
    to_addrs JSONB NOT NULL DEFAULT '[]',
    cc_addrs JSONB NOT NULL DEFAULT '[]',
    bcc_addrs JSONB NOT NULL DEFAULT '[]',
    from_email TEXT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    in_reply_to TEXT,
    agent_session_id TEXT
);
"""


async def _ensure_approvals_table(pool) -> None:
    """Create the email_approvals table if it doesn't exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_EMAIL_APPROVALS_TABLE)


async def create_approval(
    pool,
    to_addrs: list[str],
    subject: str,
    body: str,
    *,
    cc_addrs: list[str] | None = None,
    bcc_addrs: list[str] | None = None,
    from_email: str | None = None,
    in_reply_to: str | None = None,
    agent_session_id: str | None = None,
) -> EmailApprovalOut:
    """Create a pending approval for an agent-initiated email."""
    import json

    row = await pool.fetchrow(
        "INSERT INTO email_approvals "
        "(id, status, to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
        "in_reply_to, agent_session_id) "
        "VALUES ($1, 'pending', $2, $3, $4, $5, $6, $7, $8, $9) "
        "RETURNING id, status, created_at, reviewed_at, reviewed_by, "
        "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
        "in_reply_to, agent_session_id",
        uuid.uuid4(),
        json.dumps(to_addrs),
        json.dumps(cc_addrs or []),
        json.dumps(bcc_addrs or []),
        from_email,
        subject,
        body,
        in_reply_to,
        agent_session_id,
    )
    return _row_to_approval_out(row)


async def list_approvals(pool, status: str | None = None) -> list[EmailApprovalOut]:
    """List approval queue entries, optionally filtered by status."""
    if status:
        rows = await pool.fetch(
            "SELECT id, status, created_at, reviewed_at, reviewed_by, "
            "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
            "in_reply_to, agent_session_id "
            "FROM email_approvals WHERE status = $1 ORDER BY created_at DESC",
            status,
        )
    else:
        rows = await pool.fetch(
            "SELECT id, status, created_at, reviewed_at, reviewed_by, "
            "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
            "in_reply_to, agent_session_id "
            "FROM email_approvals ORDER BY created_at DESC"
        )
    return [_row_to_approval_out(r) for r in rows]


async def get_approval(pool, approval_id: uuid.UUID) -> EmailApprovalOut | None:
    row = await pool.fetchrow(
        "SELECT id, status, created_at, reviewed_at, reviewed_by, "
        "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
        "in_reply_to, agent_session_id "
        "FROM email_approvals WHERE id = $1",
        approval_id,
    )
    if row is None:
        return None
    return _row_to_approval_out(row)


async def approve_email(pool, approval_id: uuid.UUID, reviewer_id: uuid.UUID) -> EmailApprovalOut:
    """Approve and send an agent-initiated email."""

    approval = await get_approval(pool, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval is already {approval.status}")

    # Actually send the email
    client = _build_client()
    try:
        async with client:
            await client.authenticate()
            to_addrs = [EmailAddress(email=addr) for addr in approval.to_]
            cc_addrs = [EmailAddress(email=addr) for addr in approval.cc]
            bcc_addrs = [EmailAddress(email=addr) for addr in approval.bcc]
            await client.send_email(
                to=to_addrs,
                subject=approval.subject,
                body=approval.body,
                cc=cc_addrs if cc_addrs else None,
                bcc=bcc_addrs if bcc_addrs else None,
                from_email=approval.from_email,
                in_reply_to=approval.in_reply_to,
            )
    except Exception as error:
        # Mark as failed but keep in queue
        await pool.execute(
            "UPDATE email_approvals SET status = 'failed', reviewed_at = now(), "
            "reviewed_by = $2 WHERE id = $1",
            approval_id,
            reviewer_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Email send failed after approval: {error}",
        ) from error

    row = await pool.fetchrow(
        "UPDATE email_approvals SET status = 'sent', reviewed_at = now(), "
        "reviewed_by = $2 WHERE id = $1 "
        "RETURNING id, status, created_at, reviewed_at, reviewed_by, "
        "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
        "in_reply_to, agent_session_id",
        approval_id,
        reviewer_id,
    )
    return _row_to_approval_out(row)


async def reject_approval(pool, approval_id: uuid.UUID, reviewer_id: uuid.UUID) -> EmailApprovalOut:
    """Reject an agent-initiated email (don't send)."""
    approval = await get_approval(pool, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval is already {approval.status}")

    row = await pool.fetchrow(
        "UPDATE email_approvals SET status = 'rejected', reviewed_at = now(), "
        "reviewed_by = $2 WHERE id = $1 "
        "RETURNING id, status, created_at, reviewed_at, reviewed_by, "
        "to_addrs, cc_addrs, bcc_addrs, from_email, subject, body, "
        "in_reply_to, agent_session_id",
        approval_id,
        reviewer_id,
    )
    return _row_to_approval_out(row)


def _row_to_approval_out(row) -> EmailApprovalOut:
    import json

    return EmailApprovalOut(
        id=row["id"],
        status=row["status"],
        created_at=row["created_at"].isoformat(),
        reviewed_at=row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
        reviewed_by=str(row["reviewed_by"]) if row["reviewed_by"] else None,
        to_=json.loads(row["to_addrs"]) if isinstance(row["to_addrs"], str) else row["to_addrs"],
        cc=json.loads(row["cc_addrs"]) if isinstance(row["cc_addrs"], str) else row["cc_addrs"],
        bcc=json.loads(row["bcc_addrs"]) if isinstance(row["bcc_addrs"], str) else row["bcc_addrs"],
        from_email=row["from_email"],
        subject=row["subject"],
        body=row["body"],
        in_reply_to=row["in_reply_to"],
        agent_session_id=row["agent_session_id"],
    )


# ------------------------------------------------------------------
# Agent-facing: queue an email for approval instead of sending
# ------------------------------------------------------------------


async def agent_send_email(
    pool,
    to_addrs: list[str],
    subject: str,
    body: str,
    *,
    cc_addrs: list[str] | None = None,
    bcc_addrs: list[str] | None = None,
    from_email: str | None = None,
    in_reply_to: str | None = None,
    agent_session_id: str | None = None,
) -> EmailApprovalOut:
    """Queue an agent-initiated email for human approval. Does NOT send."""
    return await create_approval(
        pool,
        to_addrs=to_addrs,
        subject=subject,
        body=body,
        cc_addrs=cc_addrs,
        bcc_addrs=bcc_addrs,
        from_email=from_email,
        in_reply_to=in_reply_to,
        agent_session_id=agent_session_id,
    )


def format_email_list(emails: list[EmailOut]) -> str:
    """Render emails as compact text for the agent's search_emails tool."""
    if not emails:
        return "No emails matched."
    lines: list[str] = []
    for e in emails:
        sender = ""
        if e.from_:
            first = e.from_[0]
            sender = f"{first.name} <{first.email}>" if first.name else first.email
        lines.append(
            f"📧 {e.subject or '(no subject)'} — from {sender}"
            + (f" — {e.preview[:80]}..." if e.preview else "")
            + (f" — {e.received_at}" if e.received_at else "")
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(prefix="/api/emails", tags=["email"])


@router.get("/mailboxes")
async def list_mailboxes_route(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        mailboxes = await list_mailboxes(request)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [m.model_dump(mode="json") for m in mailboxes]


@router.get("/mailboxes/{mailbox_id}")
async def list_mailbox_emails_route(
    request: Request,
    mailbox_id: str,
    limit: int = 50,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        emails = await list_emails(request, mailbox_id, limit)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in emails]


@router.get("/inbox")
async def list_inbox_route(
    request: Request,
    limit: int = 50,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        emails = await list_emails(request, None, limit)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in emails]


@router.get("/search")
async def search_emails_route(
    request: Request,
    q: str,
    mailbox_id: str | None = None,
    limit: int = 50,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        emails = await search_emails(request, q, mailbox_id, limit)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in emails]


@router.get("/{email_id}")
async def get_email_route(
    request: Request,
    email_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        email = await get_email(request, email_id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return email.model_dump(mode="json")


@router.get("/{email_id}/thread")
async def get_thread_route(
    request: Request,
    email_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        thread = await get_thread(request, email_id)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [t.model_dump(mode="json") for t in thread]


@router.post("/send", status_code=201)
async def send_email_route(
    request: Request,
    body: EmailSendIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    """Human-initiated send — goes out immediately."""
    try:
        email_id = await send_email_direct(request, body)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"id": email_id}


@router.post("/{email_id}/reply", status_code=201)
async def reply_email_route(
    request: Request,
    email_id: str,
    body: EmailReplyIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    """Human-initiated reply — goes out immediately."""
    try:
        new_id = await reply_email_direct(request, email_id, body)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"id": new_id}


# ------------------------------------------------------------------
# Approval queue endpoints
# ------------------------------------------------------------------


@router.get("/approvals")
async def list_approvals_route(
    request: Request,
    status: str | None = None,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        approvals = await list_approvals(request.app.state.db.pool, status)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return [a.model_dump(mode="json") for a in approvals]


@router.get("/approvals/{approval_id}")
async def get_approval_route(
    request: Request,
    approval_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    approval = await get_approval(request.app.state.db.pool, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval.model_dump(mode="json")


@router.post("/approvals/{approval_id}/approve")
async def approve_email_route(
    request: Request,
    approval_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        approval = await approve_email(request.app.state.db.pool, approval_id, user.id)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return approval.model_dump(mode="json")


@router.post("/approvals/{approval_id}/reject")
async def reject_approval_route(
    request: Request,
    approval_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        approval = await reject_approval(request.app.state.db.pool, approval_id, user.id)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return approval.model_dump(mode="json")
