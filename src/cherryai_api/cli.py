"""Typer CLI for running the API and smoke-testing the agent."""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(help="CherryAI API management commands.", no_args_is_help=True)
sessions_app = typer.Typer(help="Inspect chat sessions.")
app.add_typer(sessions_app, name="sessions")
wiki_app = typer.Typer(help="Inspect the wiki.")
app.add_typer(wiki_app, name="wiki")
feedback_app = typer.Typer(help="Inspect feedback (bugs, features, user stories).")
app.add_typer(feedback_app, name="feedback")
users_app = typer.Typer(help="Manage user accounts.")
app.add_typer(users_app, name="users")


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
    from cherryai_api.agent import build_agent, run_turn

    async def _run() -> str:
        agent = build_agent()
        result = await run_turn(agent, prompt)
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
    """List wiki pages, newest-updated first."""
    from cherryai_api.db import build_database
    from cherryai_api.wiki import list_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            entries = await list_entries(db.pool)
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
    """Full-text search the wiki and print matching pages."""
    from cherryai_api.db import build_database
    from cherryai_api.wiki import format_search_results, search_entries

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            hits = await search_entries(db.pool, query)
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


def main() -> None:
    from cherryai_api.logging_setup import setup_file_logging
    from cherryai_api.settings import get_settings

    setup_file_logging(get_settings().log_dir)
    app()


if __name__ == "__main__":
    main()
