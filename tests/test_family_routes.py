"""Family CRUD + switcher routes via TestClient with auth overridden."""

import uuid

import pytest
from starlette.testclient import TestClient

from cherryai_api.api import app
from cherryai_api.auth import current_verified_user
from cherryai_api.family_context import ACTIVE_FAMILY_COOKIE
from cherryai_api.orm import async_session_maker, engine
from tests.test_families_models import _mk_user


@pytest.fixture()
async def actor(pool):
    async with async_session_maker() as session:
        user = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        await session.commit()
        await session.refresh(user)
    # `engine` is a process-wide singleton bound to whichever event loop last
    # touched it; TestClient below drives the app from its own thread/loop
    # (Starlette's sync test transport), so dispose here to force fresh
    # connections rather than reusing ones bound to this fixture's loop
    # (same mechanics as test_cli_errors.py).
    await engine.dispose()
    app.dependency_overrides[current_verified_user] = lambda: user
    try:
        yield user
    finally:
        app.dependency_overrides.pop(current_verified_user, None)


def test_create_list_rename_delete_family(actor):
    # `with` keeps one portal (and one event loop) alive for every request
    # this client makes; without it, TestClient opens a fresh portal/loop per
    # request, and the shared async engine's pooled connection — bound to the
    # first request's loop — errors with "attached to a different loop" on
    # the second.
    with TestClient(app) as client:
        created = client.post("/api/families", json={"name": "Ztest Fam"})
        assert created.status_code == 201
        fam_id = created.json()["id"]

        listed = client.get("/api/families").json()
        assert [f for f in listed if f["id"] == fam_id][0]["role"] == "organizer"

        assert (
            client.patch(f"/api/families/{fam_id}", json={"name": "Ztest Renamed"}).status_code
            == 200
        )

        deleted = client.request(
            "DELETE",
            f"/api/families/{fam_id}",
            json={"confirm_name": "Ztest Renamed", "content": "delete"},
        )
        assert deleted.status_code == 204
        assert client.get("/api/families").json() == []


def test_delete_requires_typed_confirmation(actor):
    with TestClient(app) as client:
        fam_id = client.post("/api/families", json={"name": "Ztest Fam"}).json()["id"]
        r = client.request(
            "DELETE",
            f"/api/families/{fam_id}",
            json={"confirm_name": "wrong name", "content": "delete"},
        )
        assert r.status_code == 400


def test_switcher_sets_cookie(actor):
    with TestClient(app) as client:
        fam_id = client.post("/api/families", json={"name": "Ztest Fam"}).json()["id"]
        r = client.post("/api/families/active", json={"family_id": fam_id})
        assert r.status_code == 204
        assert client.cookies.get(ACTIVE_FAMILY_COOKIE) == fam_id
        # switching back to personal clears it
        r2 = client.post("/api/families/active", json={"family_id": None})
        assert r2.status_code == 204
        assert not client.cookies.get(ACTIVE_FAMILY_COOKIE)


def test_switcher_rejects_non_membership(actor):
    with TestClient(app) as client:
        r = client.post("/api/families/active", json={"family_id": str(uuid.uuid4())})
        assert r.status_code == 403


def test_rename_requires_organizer(actor, pool):
    with TestClient(app) as client:
        client.post("/api/families", json={"name": "Ztest Fam"})
        # a stranger (not organizer of fam) — create second user and re-override
        r = client.patch(f"/api/families/{uuid.uuid4()}", json={"name": "X"})
        assert r.status_code == 404
