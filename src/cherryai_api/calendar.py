"""Calendar integration: Fastmail CalDAV calendars and events via fastmail-sdk.

Mirrors ``feedback.py`` / ``wiki.py``: pydantic response models, data access
functions that call the SDK, and a FastAPI router mounted under
``/api/calendars``. The chat agent reuses the data access functions for its
calendar tools.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastmail_sdk import CalDavClient
from fastmail_sdk.errors import (
    CalendarNotFound,
    EventConflict,
    EventNotFound,
    FastmailError,
)
from fastmail_sdk.models.event import EventDateTime, EventQuery
from pydantic import BaseModel

from cherryai_api.auth import current_verified_user
from cherryai_api.users import User

# ------------------------------------------------------------------
# Response models (flat, no SDK coupling)
# ------------------------------------------------------------------


class CalendarOut(BaseModel):
    id: str
    name: str
    color: str | None = None
    description: str | None = None
    is_default: bool = False


class EventDateTimeOut(BaseModel):
    value: str
    timezone: str | None = None
    all_day: bool = False


class EventAttendeeOut(BaseModel):
    email: str
    name: str | None = None
    role: str | None = None
    partstat: str | None = None
    rsvp: bool | None = None


class EventRecurrenceOut(BaseModel):
    frequency: str
    interval: int | None = None
    count: int | None = None
    until: str | None = None
    by_day: list[str] = []


class EventReminderOut(BaseModel):
    minutes_before: int
    action: str = "DISPLAY"


class CalendarEventOut(BaseModel):
    id: str
    calendar_id: str
    calendar_name: str | None = None
    title: str
    start: EventDateTimeOut
    end: EventDateTimeOut
    location: str | None = None
    description: str | None = None
    attendees: list[EventAttendeeOut] = []
    recurrence: EventRecurrenceOut | None = None
    reminders: list[EventReminderOut] = []


class EventCreateIn(BaseModel):
    calendar_id: str | None = None
    title: str
    start: str
    end: str
    timezone: str | None = None
    location: str | None = None
    description: str | None = None
    attendees: list[str] = []
    recurrence_freq: str | None = None
    recurrence_interval: int | None = None
    recurrence_count: int | None = None
    recurrence_until: str | None = None
    recurrence_by_day: list[str] = []
    reminder_minutes: list[int] = []


class EventUpdateIn(BaseModel):
    title: str | None = None
    start: str | None = None
    end: str | None = None
    timezone: str | None = None
    location: str | None = None
    description: str | None = None
    attendees: list[str] | None = None
    clear_attendees: bool = False
    recurrence_freq: str | None = None
    recurrence_interval: int | None = None
    recurrence_count: int | None = None
    recurrence_until: str | None = None
    recurrence_by_day: list[str] | None = None
    clear_recurrence: bool = False
    reminder_minutes: list[int] | None = None
    clear_reminders: bool = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_client() -> CalDavClient:
    """Build a CalDAV client from env vars or config file."""
    from fastmail_sdk.config import load_credentials

    try:
        username, app_password = load_credentials()
    except Exception as error:
        raise FastmailError(
            "Fastmail calendar credentials not found. "
            "Set FASTMAIL_USERNAME and FASTMAIL_APP_PASSWORD, "
            "or add a [calendar] section to ~/.config/fastmail-cli/config.toml."
        ) from error
    return CalDavClient(username=username, app_password=app_password)


def _get_client(request: Request) -> CalDavClient:
    """Build a client from request-scoped settings (router use)."""
    return _build_client()


def _to_calendar_out(cal) -> CalendarOut:
    return CalendarOut(
        id=cal.id,
        name=cal.name,
        color=cal.color,
        description=cal.description,
        is_default=cal.is_default,
    )


def _to_event_out(event) -> CalendarEventOut:
    return CalendarEventOut(
        id=event.id,
        calendar_id=event.calendar_id,
        calendar_name=event.calendar_name,
        title=event.title,
        start=EventDateTimeOut(
            value=event.start.value,
            timezone=event.start.timezone,
            all_day=event.start.all_day,
        ),
        end=EventDateTimeOut(
            value=event.end.value,
            timezone=event.end.timezone,
            all_day=event.end.all_day,
        ),
        location=event.location,
        description=event.description,
        attendees=[
            EventAttendeeOut(
                email=a.email,
                name=a.name,
                role=a.role,
                partstat=a.partstat,
                rsvp=a.rsvp,
            )
            for a in event.attendees
        ],
        recurrence=EventRecurrenceOut(
            frequency=event.recurrence.frequency,
            interval=event.recurrence.interval,
            count=event.recurrence.count,
            until=event.recurrence.until,
            by_day=event.recurrence.by_day,
        )
        if event.recurrence
        else None,
        reminders=[
            EventReminderOut(minutes_before=r.minutes_before, action=r.action)
            for r in event.reminders
        ],
    )


# ------------------------------------------------------------------
# Data access (used by both the router and the agent tools)
# ------------------------------------------------------------------


async def list_calendars(request: Request | None = None) -> list[CalendarOut]:
    client = _build_client()
    async with client:
        calendars = await client.list_calendars()
    return [_to_calendar_out(c) for c in calendars]


async def list_events(
    request: Request | None = None,
    calendar_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    week: bool = False,
) -> list[CalendarEventOut]:
    """List events for a specific calendar."""
    client = _build_client()
    async with client:
        range_start, range_end = _resolve_range(start, end, week)
        events = await client.list_events(
            EventQuery(
                calendar_id=calendar_id,
                start=range_start,
                end=range_end,
            )
        )
    return [_to_event_out(e) for e in events]


async def list_all_events(
    request: Request | None = None,
    start: str | None = None,
    end: str | None = None,
    week: bool = False,
) -> list[CalendarEventOut]:
    """List events across ALL calendars."""
    client = _build_client()
    async with client:
        range_start, range_end = _resolve_range(start, end, week)
        events = await client.list_events(EventQuery(start=range_start, end=range_end))
    return [_to_event_out(e) for e in events]


def _resolve_range(start: str | None, end: str | None, week: bool) -> tuple[datetime, datetime]:
    """Resolve a date range from parameters."""
    if week:
        from fastmail_sdk.ical import current_week_range

        return current_week_range()
    if start and end:
        from fastmail_sdk.ical import parse_range_end, parse_range_start

        return parse_range_start(start), parse_range_end(end)
    from fastmail_sdk.ical import default_today_range

    return default_today_range()


async def get_event(
    request: Request | None = None,
    event_id: str = "",
    calendar_id: str | None = None,
) -> CalendarEventOut:
    client = _build_client()
    async with client:
        event = await client.get_event_by_id(event_id, calendar_id)
    return _to_event_out(event)


async def create_event(
    request: Request | None = None, data: EventCreateIn | None = None
) -> CalendarEventOut:
    from fastmail_sdk.ical import build_event_uid
    from fastmail_sdk.models.event import (
        CalendarEvent,
        EventAttendee,
        EventRecurrence,
        EventReminder,
    )

    client = _build_client()

    # Build the SDK model
    start_dt = _parse_datetime(data.start, data.timezone)
    end_dt = _parse_datetime(data.end, data.timezone)

    recurrence = None
    if data.recurrence_freq:
        recurrence = EventRecurrence(
            frequency=data.recurrence_freq,
            interval=data.recurrence_interval,
            count=data.recurrence_count,
            until=data.recurrence_until,
            by_day=data.recurrence_by_day,
        )

    reminders = [EventReminder(minutes_before=m, action="DISPLAY") for m in data.reminder_minutes]

    attendees = [EventAttendee(email=email) for email in data.attendees]

    event = CalendarEvent(
        id=build_event_uid(),
        calendar_id="",
        title=data.title,
        start=start_dt,
        end=end_dt,
        location=data.location,
        description=data.description,
        attendees=attendees,
        recurrence=recurrence,
        reminders=reminders,
    )

    async with client:
        created = await client.create_event(data.calendar_id, event)
    return _to_event_out(created)


async def update_event(
    request: Request | None = None,
    event_id: str = "",
    calendar_id: str | None = None,
    data: EventUpdateIn | None = None,
) -> CalendarEventOut:
    client = _build_client()
    async with client:
        existing = await client.get_event_by_id(event_id, calendar_id)
        if not existing.etag:
            raise HTTPException(status_code=409, detail="Event has no ETag — cannot update")

        # Apply patch
        from fastmail_sdk.models.event import (
            EventAttendee,
            EventRecurrence,
            EventReminder,
        )

        updated = existing.model_copy()
        if data.title is not None:
            updated.title = data.title
        if data.start is not None:
            updated.start = _parse_datetime(data.start, data.timezone or existing.start.timezone)
        if data.end is not None:
            updated.end = _parse_datetime(data.end, data.timezone or existing.end.timezone)
        if data.timezone is not None and not updated.start.all_day:
            updated.start.timezone = data.timezone
            updated.end.timezone = data.timezone
        if data.location is not None:
            updated.location = data.location if data.location else None
        if data.description is not None:
            updated.description = data.description if data.description else None
        if data.clear_attendees:
            updated.attendees = []
        elif data.attendees is not None:
            updated.attendees = [EventAttendee(email=e) for e in data.attendees]
        if data.clear_recurrence:
            updated.recurrence = None
        elif data.recurrence_freq is not None:
            updated.recurrence = EventRecurrence(
                frequency=data.recurrence_freq,
                interval=data.recurrence_interval,
                count=data.recurrence_count,
                until=data.recurrence_until,
                by_day=data.recurrence_by_day or [],
            )
        if data.clear_reminders:
            updated.reminders = []
        elif data.reminder_minutes is not None:
            updated.reminders = [
                EventReminder(minutes_before=m, action="DISPLAY") for m in data.reminder_minutes
            ]

        result = await client.update_event(updated, existing.etag)
    return _to_event_out(result)


async def delete_event(
    request: Request | None = None,
    event_id: str = "",
    calendar_id: str | None = None,
) -> None:
    client = _build_client()
    async with client:
        event = await client.get_event_by_id(event_id, calendar_id)
        await client.delete_event(event)


async def search_events(request: Request | None = None, query: str = "") -> list[CalendarEventOut]:
    """Substring search over event titles/descriptions across all calendars."""
    client = _build_client()
    async with client:
        events = await client.list_events(EventQuery())
    query_lower = query.lower()
    filtered = [
        e
        for e in events
        if query_lower in e.title.lower()
        or (e.description and query_lower in e.description.lower())
        or (e.location and query_lower in e.location.lower())
    ]
    return [_to_event_out(e) for e in filtered]


def format_event_list(events: list[CalendarEventOut]) -> str:
    """Render events as compact text for the agent's search_calendar tool."""
    if not events:
        return "No calendar events matched."
    lines: list[str] = []
    for e in events:
        cal = f" [{e.calendar_name or e.calendar_id}]" if e.calendar_name else ""
        lines.append(
            f"📅 {e.title}{cal} — {e.start.value} → {e.end.value}"
            + (f" — {e.location}" if e.location else "")
            + (f" — /calendar/{e.calendar_id}" if e.calendar_id else "")
        )
    return "\n".join(lines)


def _parse_datetime(value: str, timezone: str | None) -> EventDateTime:
    """Parse a user-supplied date/time string into an EventDateTime."""

    # All-day: YYYY-MM-DD
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return EventDateTime(value=value, all_day=True)
    except ValueError:
        pass

    # UTC: RFC 3339 with Z
    if value.endswith("Z"):
        return EventDateTime(value=value, all_day=False)

    # Naive local: YYYY-MM-DDTHH:MM[:SS]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            datetime.strptime(value, fmt)
            return EventDateTime(value=value, timezone=timezone, all_day=False)
        except ValueError:
            pass

    # RFC 3339 with offset
    try:
        dt = datetime.fromisoformat(value)
        return EventDateTime(
            value=dt.isoformat(),
            all_day=False,
        )
    except ValueError:
        pass

    raise HTTPException(status_code=400, detail=f"Invalid date/time: {value}")


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------


router = APIRouter(prefix="/api/calendars", tags=["calendar"])


@router.get("")
async def list_calendars_route(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        calendars = await list_calendars(request)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [c.model_dump(mode="json") for c in calendars]


@router.get("/search")
async def search_calendar_events(
    request: Request,
    q: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        events = await search_events(request, q)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in events]


@router.get("/events")
async def list_all_events_route(
    request: Request,
    start: str | None = None,
    end: str | None = None,
    week: bool = False,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    """List events across ALL calendars."""
    try:
        events = await list_all_events(request, start, end, week)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in events]


@router.get("/{calendar_id}/events")
async def list_events_route(
    request: Request,
    calendar_id: str,
    start: str | None = None,
    end: str | None = None,
    week: bool = False,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    try:
        events = await list_events(request, calendar_id, start, end, week)
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return [e.model_dump(mode="json") for e in events]


@router.get("/{calendar_id}/events/{event_id}")
async def get_event_route(
    request: Request,
    calendar_id: str,
    event_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        event = await get_event(request, event_id, calendar_id)
    except EventNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return event.model_dump(mode="json")


@router.post("/{calendar_id}/events", status_code=201)
async def create_event_route(
    request: Request,
    calendar_id: str,
    body: EventCreateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    if body.calendar_id is None:
        body.calendar_id = calendar_id
    try:
        event = await create_event(request, body)
    except CalendarNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return event.model_dump(mode="json")


@router.put("/{calendar_id}/events/{event_id}")
async def update_event_route(
    request: Request,
    calendar_id: str,
    event_id: str,
    body: EventUpdateIn,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        event = await update_event(request, event_id, calendar_id, body)
    except EventNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except EventConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return event.model_dump(mode="json")


@router.delete("/{calendar_id}/events/{event_id}", status_code=204)
async def delete_event_route(
    request: Request,
    calendar_id: str,
    event_id: str,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    try:
        await delete_event(request, event_id, calendar_id)
    except EventNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except EventConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FastmailError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
