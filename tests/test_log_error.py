"""POST /api/log/error: persists frontend error reports to Postgres.

Uses the real app (same approach as test_route_protection.py) rather than
the lifespan-triggering TestClient — this endpoint only depends on
``get_async_session``, which is independent of the app's lifespan-managed
state (agent, workflows, Cognee memory).
"""

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError

from cherryai_api.api import app
from cherryai_api.orm import get_async_session


@pytest.fixture
async def log_error_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_valid_payload_persists_expected_fields(log_error_client, pool):
    payload = {
        "message": "Ztest boom",
        "source": "app.js",
        "lineno": 42,
        "colno": 7,
        "stack": "Error: Ztest boom\n  at foo (app.js:42:7)",
        "url": "https://example.com/page",
        "user_agent": "ZtestAgent/1.0",
        "timestamp": "2026-07-24T00:00:00.000Z",
        "context": "unhandled:window.onerror",
    }
    res = await log_error_client.post("/api/log/error", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    row = await pool.fetchrow(
        "SELECT * FROM frontend_errors WHERE message = $1", payload["message"]
    )
    assert row is not None
    assert row["source"] == payload["source"]
    assert row["lineno"] == payload["lineno"]
    assert row["colno"] == payload["colno"]
    assert row["stack"] == payload["stack"]
    assert row["url"] == payload["url"]
    assert row["user_agent"] == payload["user_agent"]
    assert row["client_timestamp"] == payload["timestamp"]
    assert row["context"] == payload["context"]
    assert row["created_at"] is not None


@pytest.mark.asyncio
async def test_missing_url_and_user_agent_fall_back_to_headers(log_error_client, pool):
    payload = {"message": "Ztest fallback"}
    res = await log_error_client.post(
        "/api/log/error",
        json=payload,
        headers={"referer": "https://example.com/referred", "user-agent": "ZtestHeaderAgent/2.0"},
    )
    assert res.status_code == 200

    row = await pool.fetchrow(
        "SELECT * FROM frontend_errors WHERE message = $1", payload["message"]
    )
    assert row is not None
    assert row["url"] == "https://example.com/referred"
    assert row["user_agent"] == "ZtestHeaderAgent/2.0"
    # httpx's ASGITransport reports the client as a loopback address.
    assert row["client_ip"]


@pytest.mark.parametrize(
    "failure",
    [
        # SQLAlchemy renders the failing INSERT and its bound parameters into
        # str(exc)...
        SQLAlchemyError("Ztest simulated outage"),
        # ...and an unreachable database never gets far enough to raise a
        # SQLAlchemyError at all: asyncio's socket layer raises a bare
        # OSError carrying the database host and port.
        ConnectionRefusedError(111, "Connect call failed ('127.0.0.1', 5432)"),
    ],
    ids=["sqlalchemy-error", "connection-refused"],
)
@pytest.mark.asyncio
async def test_persistence_failure_still_returns_success_shaped_response(log_error_client, failure):
    class _RaisingSession:
        def add(self, _row) -> None:
            pass

        async def commit(self) -> None:
            raise failure

    async def _broken_session():
        yield _RaisingSession()

    app.dependency_overrides[get_async_session] = _broken_session
    try:
        res = await log_error_client.post(
            "/api/log/error", json={"message": "Ztest will not persist"}
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "error"
        # Neither the exception text nor the connection target may reach an
        # unauthenticated caller.
        assert "Ztest simulated outage" not in body["detail"]
        assert "5432" not in body["detail"]
    finally:
        app.dependency_overrides.pop(get_async_session, None)


@pytest.mark.asyncio
async def test_oversized_fields_are_truncated_not_rejected(log_error_client, pool):
    # Every capped field, each sent well over its limit. A 422 here would
    # discard the whole report, since errorLogger.ts ignores the response.
    caps = {
        "message": 4096,
        "source": 2048,
        "stack": 16384,
        "url": 2048,
        "user_agent": 512,
        "timestamp": 64,
        "context": 256,
    }
    payload = {field: "x" * (cap + 100) for field, cap in caps.items()}
    payload["message"] = "Ztest " + payload["message"]
    res = await log_error_client.post("/api/log/error", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    row = await pool.fetchrow(
        "SELECT * FROM frontend_errors WHERE message LIKE 'Ztest x%' "
        "ORDER BY created_at DESC LIMIT 1"
    )
    assert row is not None
    for field, cap in caps.items():
        column = "client_timestamp" if field == "timestamp" else field
        assert len(row[column]) == cap, f"{field} not truncated to {cap}"


@pytest.mark.asyncio
async def test_out_of_range_line_numbers_are_clamped_not_dropped(log_error_client, pool):
    # lineno/colno land in PG `integer`; an out-of-range value would raise a
    # DataError at commit and silently lose the report.
    res = await log_error_client.post(
        "/api/log/error",
        json={"message": "Ztest huge lineno", "lineno": 99_999_999_999_999, "colno": -(2**40)},
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    row = await pool.fetchrow(
        "SELECT * FROM frontend_errors WHERE message = $1", "Ztest huge lineno"
    )
    assert row is not None
    assert row["lineno"] == 2**31 - 1
    assert row["colno"] == -(2**31)
