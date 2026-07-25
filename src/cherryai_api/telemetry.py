"""Pydantic Logfire observability for the API.

Exports traces, metrics, and logs to the ``kerryhatcher/cherry-ai`` Logfire
project. ``send_to_logfire="if-token-present"`` means nothing is exported
unless a write token exists (``LOGFIRE_TOKEN`` env var or the gitignored
``.logfire/`` credentials created by ``logfire projects use``), so imports,
tests, and CI never fail or phone home without credentials.

Setup is split into two phases because they have incompatible timing
requirements — see each function's docstring. Getting the split wrong fails
*silently*: no exception, no warning, just missing spans.
"""

from fastapi import FastAPI
from loguru import logger

_configured = False
_instrumented = False


def configure_telemetry(app: FastAPI) -> None:
    """Configure Logfire and instrument the ASGI app. MUST run at import time.

    Call this at module scope, immediately after ``FastAPI()`` is constructed —
    never from inside the lifespan.

    ``logfire.instrument_fastapi()`` delegates to
    ``FastAPIInstrumentor.instrument_app()``, which monkeypatches the app's
    ``build_middleware_stack`` *method*. Starlette calls that method exactly
    once, lazily, the first time the app receives any ASGI scope — and that
    includes the ``"lifespan"`` scope, which is dispatched *before* the lifespan
    body runs. So by the time lifespan code executes, ``app.middleware_stack``
    is already built and cached, and Starlette never rebuilds it. Instrumenting
    from the lifespan patches a method that will never be called again: the
    OpenTelemetry middleware never enters the request path and zero HTTP spans
    are ever produced.

    Nothing detects this. The instrumentation call itself succeeds, no
    exception is raised, and every other instrumentation keeps working — which
    is exactly how it went unnoticed in production. (It used to fail loudly:
    the instrumentor called ``app.add_middleware()``, which raises
    ``RuntimeError`` after startup. opentelemetry-instrumentation-fastapi
    0.58b0 replaced that with the unguarded monkeypatch to stop instrumentation
    from crashing services.)

    Idempotent so repeated imports install everything once.
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


def instrument_dependencies() -> None:
    """Instrument the libraries the app calls at runtime.

    Safe to call from the lifespan, and it must be: these patch libraries or
    start background collectors rather than touching Starlette's middleware
    stack, so they are unaffected by the ordering trap described in
    :func:`configure_telemetry`.

    Must still run before the asyncpg pool is created, or database calls made
    on already-open connections aren't traced. ``configure_telemetry`` must
    have run first, since ``logfire.configure()`` lives there — module import
    always precedes the lifespan, so that ordering holds automatically.

    Idempotent so uvicorn reload / repeated lifespans install everything once.
    """
    global _instrumented
    if _instrumented:
        return
    _instrumented = True

    import logfire

    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    logfire.instrument_asyncpg()
    logfire.instrument_system_metrics()
    # Forward existing loguru log lines (alongside the stderr and JSONL sinks).
    logger.add(**logfire.loguru_handler())
