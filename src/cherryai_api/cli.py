"""Typer CLI for running the API and smoke-testing the agent."""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(help="CherryAI API management commands.", no_args_is_help=True)
sessions_app = typer.Typer(help="Inspect chat sessions.")
app.add_typer(sessions_app, name="sessions")


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
