"""Family CRUD + switcher routes via httpx.AsyncClient + ASGITransport.

Uses the established pattern from ``test_log_error.py`` (ASGITransport, no
real lifespan) rather than ``with TestClient(app) as client:`` — the latter
also runs the app's actual lifespan (Cognee import, agent build, workflow
runtime), which this endpoint doesn't need. All requests in a given test
run inside the same coroutine/event loop (the one pytest-asyncio assigns to
the test item), so there's no cross-event-loop asyncpg connection issue to
work around here.
"""

import uuid

import httpx
import pytest

from cherryai_api.api import app
from cherryai_api.auth import current_verified_user
from cherryai_api.families import FAMILY_ROLE_ADULT, FamilyMembership
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
    # touched it. Disposing here forces fresh connections for this test's
    # loop rather than reusing ones bound to a previous test's now-closed
    # loop (same mechanics as test_cli_errors.py).
    await engine.dispose()
    app.dependency_overrides[current_verified_user] = lambda: user
    try:
        yield user
    finally:
        app.dependency_overrides.pop(current_verified_user, None)


@pytest.fixture()
async def api_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_create_list_rename_delete_family(actor, api_client):
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    assert created.status_code == 201
    fam_id = created.json()["id"]

    listed = (await api_client.get("/api/families")).json()
    match = [f for f in listed if f["id"] == fam_id][0]
    assert match["role"] == "organizer"
    assert match["perm_wiki"] == "edit"
    assert match["perm_meals"] == "edit"
    assert match["perm_planner"] == "edit"
    assert match["chat_enabled"] is True
    assert match["web_enabled"] is True

    renamed = await api_client.patch(f"/api/families/{fam_id}", json={"name": "Ztest Renamed"})
    assert renamed.status_code == 200

    deleted = await api_client.request(
        "DELETE",
        f"/api/families/{fam_id}",
        json={"confirm_name": "Ztest Renamed", "content": "delete"},
    )
    assert deleted.status_code == 204
    assert (await api_client.get("/api/families")).json() == []


async def test_delete_requires_typed_confirmation(actor, api_client):
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    fam_id = created.json()["id"]
    r = await api_client.request(
        "DELETE",
        f"/api/families/{fam_id}",
        json={"confirm_name": "wrong name", "content": "delete"},
    )
    assert r.status_code == 400


async def test_switcher_sets_cookie(actor, api_client):
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    fam_id = created.json()["id"]
    r = await api_client.post("/api/families/active", json={"family_id": fam_id})
    assert r.status_code == 204
    assert api_client.cookies.get(ACTIVE_FAMILY_COOKIE) == fam_id
    # switching back to personal clears it
    r2 = await api_client.post("/api/families/active", json={"family_id": None})
    assert r2.status_code == 204
    assert not api_client.cookies.get(ACTIVE_FAMILY_COOKIE)


async def test_switcher_rejects_non_membership(actor, api_client):
    r = await api_client.post("/api/families/active", json={"family_id": str(uuid.uuid4())})
    assert r.status_code == 403


async def test_rename_unknown_family_404(actor, api_client, pool):
    await api_client.post("/api/families", json={"name": "Ztest Fam"})
    # existence-protecting: a family the caller has no membership in 404s,
    # rather than 403ing and thereby confirming the id exists (spec §7)
    r = await api_client.patch(f"/api/families/{uuid.uuid4()}", json={"name": "X"})
    assert r.status_code == 404


async def test_rename_requires_organizer_role(actor, api_client):
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    fam_id = created.json()["id"]

    async with async_session_maker() as session:
        member = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        session.add(
            FamilyMembership(
                family_id=uuid.UUID(fam_id),
                user_id=member.id,
                role=FAMILY_ROLE_ADULT,
            )
        )
        await session.commit()
        await session.refresh(member)

    app.dependency_overrides[current_verified_user] = lambda: member
    r = await api_client.patch(f"/api/families/{fam_id}", json={"name": "X"})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "family_permission_denied"


async def test_delete_keep_personal_reassigns_and_deconflicts_wiki(actor, api_client, pool):
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    fam_id = created.json()["id"]

    async with async_session_maker() as session:
        other = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        await session.commit()
        await session.refresh(other)

    family_entry_id = uuid.uuid4()
    personal_entry_id = uuid.uuid4()
    # A family-owned page (owned by some other member) whose slug collides
    # with one of the organizer's existing personal pages.
    await pool.execute(
        "INSERT INTO wiki_entries (id, slug, title, owner_id, family_id) "
        "VALUES ($1, $2, $3, $4, $5)",
        family_entry_id,
        "ztest-keep",
        "Ztest Keep",
        other.id,
        uuid.UUID(fam_id),
    )
    await pool.execute(
        "INSERT INTO wiki_entries (id, slug, title, owner_id, family_id) "
        "VALUES ($1, $2, $3, $4, $5)",
        personal_entry_id,
        "ztest-keep",
        "Ztest Keep Personal",
        actor.id,
        None,
    )

    deleted = await api_client.request(
        "DELETE",
        f"/api/families/{fam_id}",
        json={"confirm_name": "Ztest Fam", "content": "keep_personal"},
    )
    assert deleted.status_code == 204

    row = await pool.fetchrow(
        "SELECT slug, owner_id, family_id FROM wiki_entries WHERE id = $1",
        family_entry_id,
    )
    assert row["family_id"] is None
    assert row["owner_id"] == actor.id
    assert row["slug"] != "ztest-keep"
    assert row["slug"].startswith("ztest-keep-")
