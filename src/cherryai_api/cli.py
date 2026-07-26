"""Typer CLI for running the API and smoke-testing the agent."""

from __future__ import annotations

import asyncio
import uuid

import typer

# Only a lightweight constant, not the rest of `cherryai_api` (unlike every
# other command below, which imports lazily inside its function body to keep
# `cherryai --help` fast) — Typer needs the real default at decoration time
# so `--help` can display it, and importing it doesn't pull in anything
# heavier than `cherryai_api.orm` (which builds a `create_async_engine()`
# object but performs no I/O until first use).
from cherryai_api.frontend_errors import FRONTEND_ERROR_RETENTION_DAYS

app = typer.Typer(help="CherryAI API management commands.", no_args_is_help=True)
sessions_app = typer.Typer(help="Inspect chat sessions.")
app.add_typer(sessions_app, name="sessions")
wiki_app = typer.Typer(help="Inspect the wiki.")
app.add_typer(wiki_app, name="wiki")
feedback_app = typer.Typer(help="Inspect feedback (bugs, features, user stories).")
app.add_typer(feedback_app, name="feedback")
users_app = typer.Typer(help="Manage user accounts.")
app.add_typer(users_app, name="users")
calendar_app = typer.Typer(help="Inspect Fastmail calendars and events.")
app.add_typer(calendar_app, name="calendar")
errors_app = typer.Typer(help="Inspect frontend error reports.")
app.add_typer(errors_app, name="errors")
families_app = typer.Typer(help="Manage families and memberships.")
app.add_typer(families_app, name="families")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on changes."),
) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run(
        "cherryai_api.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def chat(prompt: str) -> None:
    """Run a one-shot prompt through the agent (smoke test, no persistence)."""
    from cherryai_api.agent import AgentDeps, build_agent, run_turn
    from cherryai_api.memory import build_memory

    async def _run() -> str:
        agent = await build_agent()
        deps = AgentDeps(memory=build_memory(), user_id=uuid.uuid4())
        result = await run_turn(agent, prompt, deps=deps)
        return result.output

    output = asyncio.run(_run())
    typer.echo(output)


@sessions_app.command("list")
def sessions_list() -> None:
    """List recent chat sessions across all users, newest first (admin tool)."""
    from cherryai_api.db import build_database

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            sessions = await db.list_all_sessions()
        finally:
            await db.close()
        if not sessions:
            typer.echo("No sessions yet.")
            return
        for session in sessions:
            typer.echo(
                f"{session.id}  {session.created_at:%Y-%m-%d %H:%M}  "
                f"user={session.user_id}  {session.title}"
            )

    asyncio.run(_run())


@wiki_app.command("list")
def wiki_list() -> None:
    """List wiki pages across all owners, newest-updated first."""
    from cherryai_api.db import build_database
    from cherryai_api.wiki import list_all_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            entries = await list_all_entries(db.pool)
        finally:
            await db.close()
        if not entries:
            typer.echo("No wiki pages yet.")
            return
        for entry in entries:
            tags = f"  [{', '.join(entry.tags)}]" if entry.tags else ""
            location = f"  ({entry.folder})" if entry.folder else ""
            typer.echo(
                f"{entry.slug}  {entry.updated_at:%Y-%m-%d %H:%M}  {entry.title}{location}{tags}"
            )

    asyncio.run(_run())


@wiki_app.command("search")
def wiki_search(query: str) -> None:
    """Full-text search the wiki across all owners and print matching pages."""
    from cherryai_api.db import build_database
    from cherryai_api.wiki import format_search_results, search_all_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            hits = await search_all_entries(db.pool, query)
        finally:
            await db.close()
        typer.echo(format_search_results(hits))

    asyncio.run(_run())


@feedback_app.command("list")
def feedback_list(
    status: str = typer.Option(None, "--status", help="Filter by status."),
    type: str = typer.Option(None, "--type", help="Filter by type."),
    priority: str = typer.Option(None, "--priority", help="Filter by priority."),
) -> None:
    """List feedback entries, newest-updated first."""
    from cherryai_api.db import build_database
    from cherryai_api.feedback import list_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            entries = await list_entries(db.pool, type=type, status=status, priority=priority)
        finally:
            await db.close()
        if not entries:
            typer.echo("No feedback entries yet.")
            return
        for entry in entries:
            tags = f"  [{', '.join(entry.tags)}]" if entry.tags else ""
            typer.echo(
                f"#{entry.id}  {entry.updated_at:%Y-%m-%d %H:%M}  "
                f"[{entry.type}/{entry.status}/{entry.priority}]  {entry.title}{tags}"
            )

    try:
        asyncio.run(_run())
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@feedback_app.command("search")
def feedback_search(query: str) -> None:
    """Full-text search feedback and print matching entries."""
    from cherryai_api.db import build_database
    from cherryai_api.feedback import format_search_results, search_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            hits = await search_entries(db.pool, query)
        finally:
            await db.close()
        typer.echo(format_search_results(hits))

    asyncio.run(_run())


def _run_with_session(fn):
    """Run an async callable with one SQLAlchemy session, then dispose."""
    from cherryai_api.orm import async_session_maker, engine

    async def _wrapped():
        try:
            async with async_session_maker() as session:
                return await fn(session)
        finally:
            await engine.dispose()

    return asyncio.run(_wrapped())


@users_app.command("bootstrap")
def users_bootstrap(
    email: str = typer.Option(None, "--email"),
    password: str = typer.Option(None, "--password"),
) -> None:
    """Create the first admin account (idempotent)."""
    from cherryai_api.settings import get_settings
    from cherryai_api.users import ensure_admin

    settings = get_settings()
    email = email or settings.cherryai_admin_email or typer.prompt("Admin email")
    password = (
        password
        or settings.cherryai_admin_password
        or typer.prompt("Admin password", hide_input=True, confirmation_prompt=True)
    )

    async def _do(session):
        return await ensure_admin(session, email, password)

    user, created = _run_with_session(_do)
    typer.echo(f"{'Created' if created else 'Already exists'}: {user.email} ({user.id})")


@users_app.command("list")
def users_list() -> None:
    """List all accounts with role and status."""
    from sqlalchemy import select

    from cherryai_api.users import User

    async def _do(session):
        return list((await session.execute(select(User).order_by(User.created_at))).scalars())

    for u in _run_with_session(_do):
        status = "pending" if not u.is_verified else ("active" if u.is_active else "deactivated")
        typer.echo(f"{u.id}  {u.email:35}  {u.role:10}  {status}")


def _mutate_by_email(email: str, mutate) -> None:
    from sqlalchemy import select

    from cherryai_api.users import User

    async def _do(session):
        user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            typer.echo(f"No user with email {email}", err=True)
            raise typer.Exit(code=1)
        mutate(user)
        await session.commit()
        return user

    user = _run_with_session(_do)
    typer.echo(
        f"OK: {user.email} role={user.role} active={user.is_active} verified={user.is_verified}"
    )


@users_app.command("approve")
def users_approve(email: str, role: str = typer.Option("chat", "--role")) -> None:
    """Approve a pending account, assigning a role."""
    from cherryai_api.users import ROLE_ADMIN, ROLES

    if role not in ROLES:
        typer.echo(f"Unknown role: {role}", err=True)
        raise typer.Exit(code=1)

    def _mutate(user):
        user.is_verified = True
        user.role = role
        user.is_superuser = role == ROLE_ADMIN

    _mutate_by_email(email, _mutate)


@users_app.command("set-role")
def users_set_role(email: str, role: str) -> None:
    """Change an account's role."""
    from cherryai_api.users import ROLE_ADMIN, ROLES

    if role not in ROLES:
        typer.echo(f"Unknown role: {role}", err=True)
        raise typer.Exit(code=1)

    def _mutate(user):
        user.role = role
        user.is_superuser = role == ROLE_ADMIN

    _mutate_by_email(email, _mutate)


@users_app.command("deactivate")
def users_deactivate(email: str) -> None:
    """Deactivate an account and revoke its sessions."""
    from sqlalchemy import delete, select

    from cherryai_api.users import AccessToken, User

    async def _do(session):
        user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            typer.echo(f"No user with email {email}", err=True)
            raise typer.Exit(code=1)
        user.is_active = False
        await session.execute(delete(AccessToken).where(AccessToken.user_id == user.id))
        await session.commit()
        return user

    user = _run_with_session(_do)
    typer.echo(
        f"OK: {user.email} role={user.role} active={user.is_active} verified={user.is_verified}"
    )


@users_app.command("reactivate")
def users_reactivate(email: str) -> None:
    """Reactivate a deactivated account."""

    def _mutate(user):
        user.is_active = True

    _mutate_by_email(email, _mutate)


@calendar_app.command("list")
def calendar_list() -> None:
    """List Fastmail calendars."""
    from cherryai_api.calendar import list_calendars

    async def _run() -> None:
        try:
            calendars = await list_calendars()
        except Exception as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1) from error
        if not calendars:
            typer.echo("No calendars found.")
            return
        for cal in calendars:
            default = " (default)" if cal.is_default else ""
            color = f" {cal.color}" if cal.color else ""
            typer.echo(f"{cal.id}  {cal.name}{default}{color}")

    asyncio.run(_run())


@calendar_app.command("events")
def calendar_events(
    calendar: str = typer.Option(None, "--calendar", "-c", help="Calendar ID."),
    week: bool = typer.Option(False, "--week", help="Show this week's events."),
    start: str = typer.Option(None, "--start", help="Range start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Range end (YYYY-MM-DD)."),
) -> None:
    """List calendar events (defaults to today)."""
    from cherryai_api.calendar import list_events

    async def _run() -> None:
        try:
            events = await list_events(
                calendar_id=calendar,
                start=start,
                end=end,
                week=week,
            )
        except Exception as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1) from error
        if not events:
            typer.echo("No events found.")
            return
        for event in events:
            cal = f" [{event.calendar_name or event.calendar_id}]"
            typer.echo(
                f"{event.id[:8]}  {event.start.value} → {event.end.value}  {event.title}{cal}"
            )

    asyncio.run(_run())


@calendar_app.command("search")
def calendar_search(query: str) -> None:
    """Search calendar events by title, description, or location."""
    from cherryai_api.calendar import search_events

    async def _run() -> None:
        try:
            events = await search_events(query=query)
        except Exception as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1) from error
        if not events:
            typer.echo(f"No events match '{query}'.")
            return
        for event in events:
            cal = f" [{event.calendar_name or event.calendar_id}]"
            typer.echo(f"{event.id[:8]}  {event.start.value}  {event.title}{cal}")

    asyncio.run(_run())


# `errors list` prints fixed-width columns so a screenful stays scannable:
# messages and stack traces are unbounded (see api.py's 4KB/16KB caps) and
# would otherwise shred the alignment. The widths below sum to 120 columns.
_ERROR_ID_PREFIX_LEN = 8
_ERROR_USER_COL_WIDTH = 28
_ERROR_CONTEXT_COL_WIDTH = 18
_ERROR_MESSAGE_PREVIEW_LEN = 40
# Width of the `label:` gutter in `errors show`, used to indent the
# continuation lines of multi-line values back under their first line.
_ERROR_LABEL_WIDTH = 12


def _fit(value: str, width: int) -> str:
    """Truncate `value` to `width` (marking the cut) and pad it out to it."""
    if len(value) > width:
        value = value[: width - 1] + "…"
    return f"{value:<{width}}"


@errors_app.command("list")
def errors_list(
    limit: int = typer.Option(50, "--limit", help="Maximum rows to show."),
) -> None:
    """List recent frontend error reports, newest first.

    Each row is a one-line, truncated summary. The leading short id is
    enough to identify a row — pass it to `errors show` for the full
    message, stack trace and request metadata.
    """
    from sqlalchemy import select

    from cherryai_api.frontend_errors import FrontendError
    from cherryai_api.users import User

    async def _do(session):
        stmt = (
            select(FrontendError, User.email)
            .outerjoin(User, FrontendError.user_id == User.id)
            .order_by(FrontendError.created_at.desc())
            .limit(limit)
        )
        return list((await session.execute(stmt)).all())

    rows = _run_with_session(_do)
    if not rows:
        typer.echo("No frontend errors yet.")
        return
    for error, email in rows:
        who = _fit(email or "(anonymous)", _ERROR_USER_COL_WIDTH)
        context = _fit((error.context or "-").replace("\n", " "), _ERROR_CONTEXT_COL_WIDTH)
        message = error.message.replace("\n", " ")
        if len(message) > _ERROR_MESSAGE_PREVIEW_LEN:
            message = message[: _ERROR_MESSAGE_PREVIEW_LEN - 1] + "…"
        typer.echo(
            f"{str(error.id)[:_ERROR_ID_PREFIX_LEN]}  {error.created_at:%Y-%m-%d %H:%M}  "
            f"{who}  [{context}]  {message}"
        )


def _echo_field(label: str, value: object) -> None:
    """Print one `label: value` line, indenting any continuation lines."""
    text = "-" if value is None or value == "" else str(value)
    first, *rest = text.split("\n")
    typer.echo(f"{label + ':':<{_ERROR_LABEL_WIDTH}}{first}")
    for line in rest:
        typer.echo(f"{'':<{_ERROR_LABEL_WIDTH}}{line}")


@errors_app.command("show")
def errors_show(error_id: str) -> None:
    """Show one error in full, by its id or any unique prefix of it."""
    from sqlalchemy import String, select

    from cherryai_api.frontend_errors import FrontendError
    from cherryai_api.users import User

    async def _do(session):
        stmt = (
            select(FrontendError, User.email)
            .outerjoin(User, FrontendError.user_id == User.id)
            .where(FrontendError.id.cast(String).startswith(error_id.lower(), autoescape=True))
            .order_by(FrontendError.created_at.desc())
            .limit(2)
        )
        return list((await session.execute(stmt)).all())

    rows = _run_with_session(_do)
    if not rows:
        typer.echo(f"No frontend error with id {error_id}", err=True)
        raise typer.Exit(code=1)
    if len(rows) > 1:
        typer.echo(f"Error: id {error_id!r} is ambiguous — use more characters", err=True)
        raise typer.Exit(code=1)

    error, email = rows[0]
    lineno = error.lineno if error.lineno is not None else "-"
    colno = error.colno if error.colno is not None else "-"
    _echo_field("id", error.id)
    _echo_field("created_at", f"{error.created_at:%Y-%m-%d %H:%M:%S %Z}")
    _echo_field("user", f"{email or '(anonymous)'} ({error.user_id or '-'})")
    _echo_field("context", error.context)
    _echo_field("url", error.url)
    _echo_field("source", f"{error.source or '-'}:{lineno}:{colno}")
    _echo_field("user_agent", error.user_agent)
    _echo_field("client_ip", error.client_ip)
    _echo_field("client_ts", error.client_timestamp)
    _echo_field("message", error.message)
    if error.stack:
        typer.echo("stack:")
        typer.echo(error.stack)


@errors_app.command("prune")
def errors_prune(
    days: int = typer.Option(
        FRONTEND_ERROR_RETENTION_DAYS,
        "--days",
        help="Delete reports older than this many days.",
    ),
) -> None:
    """Delete frontend error reports older than the retention window.

    The API also runs this automatically roughly once a day (see the
    lifespan in api.py); this command is for on-demand manual cleanup, e.g.
    to reclaim space immediately after lowering the retention window.
    """
    from cherryai_api.frontend_errors import prune_frontend_errors

    async def _do(session):
        return await prune_frontend_errors(session, retention_days=days)

    deleted = _run_with_session(_do)
    typer.echo(f"Deleted {deleted} frontend error report(s) older than {days} day(s).")


@families_app.command("list")
def families_list() -> None:
    """All families with member counts."""

    async def run() -> None:
        from sqlalchemy import func, select

        from cherryai_api.families import Family, FamilyMembership
        from cherryai_api.orm import async_session_maker

        async with async_session_maker() as session:
            rows = await session.execute(
                select(Family, func.count(FamilyMembership.id))
                .join(FamilyMembership, isouter=True)
                .group_by(Family.id)
                .order_by(Family.name)
            )
            for family, count in rows.all():
                typer.echo(f"{family.name}  members={count}  id={family.id}")

    asyncio.run(run())


@families_app.command("create")
def families_create(name: str, organizer_email: str = typer.Option(...)) -> None:
    """Create a family with an existing user as organizer."""

    async def run() -> None:
        from sqlalchemy import select

        from cherryai_api.families import create_family
        from cherryai_api.orm import async_session_maker
        from cherryai_api.users import User

        async with async_session_maker() as session:
            user = (
                await session.execute(select(User).where(User.email == organizer_email))
            ).scalar_one_or_none()
            if user is None:
                typer.echo(f"no such user: {organizer_email}", err=True)
                raise typer.Exit(1)
            family = await create_family(session, name=name, organizer_id=user.id)
            typer.echo(f"created {family.name} id={family.id}")

    asyncio.run(run())


@families_app.command("show")
def families_show(family_id: uuid.UUID) -> None:
    """Members, roles, and permission matrix for one family."""

    async def run() -> None:
        from sqlalchemy import select

        from cherryai_api.families import FamilyMembership
        from cherryai_api.orm import async_session_maker
        from cherryai_api.users import User

        async with async_session_maker() as session:
            rows = await session.execute(
                select(FamilyMembership, User.email)
                .join(User, User.id == FamilyMembership.user_id)
                .where(FamilyMembership.family_id == family_id)
            )
            for m, email in rows.all():
                typer.echo(
                    f"{email}  {m.role}  wiki={m.perm_wiki} meals={m.perm_meals} "
                    f"planner={m.perm_planner} chat={m.chat_enabled} web={m.web_enabled}"
                )

    asyncio.run(run())


@families_app.command("set-perm")
def families_set_perm(family_id: uuid.UUID, email: str, module: str, level: str) -> None:
    """Set one module permission for an adult/child member."""

    async def run() -> None:
        from sqlalchemy import select

        from cherryai_api.authz import MODULES
        from cherryai_api.families import (
            FAMILY_ROLE_ADULT,
            FAMILY_ROLE_CHILD,
            PERM_LEVELS,
            get_membership,
            update_member,
        )
        from cherryai_api.orm import async_session_maker
        from cherryai_api.users import User

        if module not in MODULES or level not in PERM_LEVELS:
            typer.echo(f"module must be one of {MODULES}, level one of {PERM_LEVELS}", err=True)
            raise typer.Exit(2)
        async with async_session_maker() as session:
            user = (
                await session.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
            membership = user and await get_membership(session, family_id, user.id)
            if not membership:
                typer.echo("no such membership", err=True)
                raise typer.Exit(1)
            if membership.role not in (FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD):
                typer.echo(f"{membership.role} permissions are implicit", err=True)
                raise typer.Exit(1)
            await update_member(
                session,
                family_id=family_id,
                user_id=user.id,
                changes={f"perm_{module}": level},
            )
            typer.echo("ok")

    asyncio.run(run())


@families_app.command("add-member")
def families_add_member(family_id: uuid.UUID, email: str, role: str = typer.Option(...)) -> None:
    """Add an existing user to a family with a specified role."""
    from cherryai_api.families import FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD

    allowed_roles = (FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD)
    if role not in allowed_roles:
        typer.echo(f"role must be one of {allowed_roles}", err=True)
        raise typer.Exit(2)

    async def run() -> None:
        from cherryai_api.families import AlreadyMemberError, add_member
        from cherryai_api.orm import async_session_maker

        async with async_session_maker() as session:
            try:
                membership = await add_member(session, family_id=family_id, email=email, role=role)
                typer.echo(f"added {email} as {membership.role} id={membership.id}")
            except LookupError as err:
                typer.echo(f"no such user: {email}", err=True)
                raise typer.Exit(1) from err
            except AlreadyMemberError as err:
                typer.echo(f"{email} is already a member of this family", err=True)
                raise typer.Exit(1) from err

    asyncio.run(run())


@families_app.command("transfer")
def families_transfer(family_id: uuid.UUID, email: str) -> None:
    """Transfer organizer role to an existing admin member."""

    async def run() -> None:
        from sqlalchemy import select

        from cherryai_api.families import (
            FAMILY_ROLE_ADMIN,
            FAMILY_ROLE_ORGANIZER,
            FamilyMembership,
            get_membership,
            transfer_organizer,
        )
        from cherryai_api.orm import async_session_maker
        from cherryai_api.users import User

        async with async_session_maker() as session:
            user = (
                await session.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
            if user is None:
                typer.echo(f"no such user: {email}", err=True)
                raise typer.Exit(1)
            target = await get_membership(session, family_id, user.id)
            if target is None:
                typer.echo("target is not a member", err=True)
                raise typer.Exit(1)
            if target.role != FAMILY_ROLE_ADMIN:
                typer.echo("transfer target must be an admin", err=True)
                raise typer.Exit(1)
            current_organizer = (
                await session.execute(
                    select(FamilyMembership).where(
                        FamilyMembership.family_id == family_id,
                        FamilyMembership.role == FAMILY_ROLE_ORGANIZER,
                    )
                )
            ).scalar_one_or_none()
            if current_organizer is None:
                typer.echo("family has no organizer", err=True)
                raise typer.Exit(1)
            await transfer_organizer(
                session,
                family_id=family_id,
                from_user_id=current_organizer.user_id,
                to_user_id=user.id,
            )
            typer.echo("ok")

    asyncio.run(run())


def main() -> None:
    from cherryai_api.logging_setup import setup_file_logging
    from cherryai_api.settings import get_settings

    setup_file_logging(get_settings().log_dir)
    app()


if __name__ == "__main__":
    main()
