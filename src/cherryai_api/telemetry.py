"""Pydantic Logfire observability for the API.

Exports traces, metrics, and logs to the ``kerryhatcher/cherry-ai`` Logfire
project. ``send_to_logfire="if-token-present"`` means nothing is exported
unless a write token exists (``LOGFIRE_TOKEN`` env var or the gitignored
``.logfire/`` credentials created by ``logfire projects use``), so imports,
tests, and CI never fail or phone home without credentials.
"""

from fastapi import FastAPI
from loguru import logger

_configured = False


def setup_telemetry(app: FastAPI) -> None:
    """Configure Logfire and instrument the app's core dependencies.

    Must run before the asyncpg pool is created so database calls are traced.
    Idempotent so uvicorn reload / repeated lifespans install everything once.
    """
    global _configured
    if _configured:
        return
    _configured = True

    import logfire

    logfire.configure(
        service_name="cherryai-api",
        send_to_logfire="if-token-present",
        console=False,
    )
    logfire.instrument_fastapi(app)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    logfire.instrument_asyncpg()
    logfire.instrument_system_metrics()
    # Forward existing loguru log lines (alongside the stderr and JSONL sinks).
    logger.add(**logfire.loguru_handler())
