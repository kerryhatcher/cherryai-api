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
    """List recent chat sessions, newest first."""
    from cherryai_api.db import build_database

    async def _run() -> None:
        db = build_database()
        await db.connect()
        try:
            sessions = await db.list_sessions()
        finally:
            await db.close()
        if not sessions:
            typer.echo("No sessions yet.")
            return
        for session in sessions:
            typer.echo(f"{session.id}  {session.created_at:%Y-%m-%d %H:%M}  {session.title}")

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
            typer.echo(f"{entry.slug}  {entry.updated_at:%Y-%m-%d %H:%M}  {entry.title}{tags}")

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
