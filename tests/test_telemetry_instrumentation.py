"""Guard the timing contract for FastAPI/ASGI instrumentation.

Regression test for a bug where `logfire.instrument_fastapi(app)` was called
from inside the FastAPI lifespan. That produced **zero** HTTP spans in
production for weeks with no error, no warning, and no log line — every other
instrumentation (asyncpg, metrics, httpx, pydantic-ai) kept working normally,
so nothing looked wrong.

Why it fails silently: `instrument_fastapi` monkeypatches the app's
`build_middleware_stack` *method*. Starlette calls that method once, lazily, on
the first ASGI scope the app receives — which is the ``"lifespan"`` scope,
dispatched *before* the lifespan body runs. So instrumenting from the lifespan
patches a method Starlette will never call again, and the OpenTelemetry
middleware never enters the request path.

If someone moves `configure_telemetry(app)` back into the lifespan (or drops
the call), these assertions fail. Without them the only symptom is an empty
Logfire dashboard that nobody notices.
"""

from __future__ import annotations

import cherryai_api.telemetry as telemetry
from cherryai_api.api import app


def test_telemetry_configured_at_import_time() -> None:
    """`configure_telemetry` must run at module scope, not in the lifespan.

    No test drives `cherryai_api.api.lifespan` end to end (see conftest), so
    the guard flag can only be set by the module-scope call in `api.py`. If
    that call moves into the lifespan, this is False.
    """
    assert telemetry._configured is True


def test_app_is_instrumented_before_serving_any_request() -> None:
    """The instrumentation must have actually taken effect on the app object.

    `FastAPIInstrumentor.instrument_app()` sets this flag. Importing `app`
    above is the only thing that has happened to it here — no ASGI scope has
    been sent — so a True flag means instrumentation landed while
    `app.middleware_stack` was still unbuilt, which is the whole contract.
    """
    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True


def test_instrumentation_precedes_middleware_stack_build() -> None:
    """Instrumentation must land before Starlette caches the middleware stack.

    Once `app.middleware_stack` is non-None it is never rebuilt, so any
    instrumentation applied afterwards is dead code. Asserting the patch marker
    exists proves `build_middleware_stack` was replaced; the patched version is
    only ever *invoked* if it was installed before the first ASGI call.
    """
    assert hasattr(app, "_original_build_middleware_stack")
