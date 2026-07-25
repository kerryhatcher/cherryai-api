"""Family membership management routes via httpx.AsyncClient + ASGITransport.

Same pattern as test_family_routes.py: ASGITransport (no real lifespan),
engine.dispose() before overriding dependencies (fresh connections for this
test's event loop), dependency_overrides for current_verified_user.
"""

import uuid

import httpx
import pytest

from cherryai_api.api import app
from cherryai_api.auth import current_verified_user
from cherryai_api.orm import async_session_maker, engine
from tests.test_families_models import _mk_user


@pytest.fixture()
async def api_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture()
async def org_and_family(pool, api_client):
    async with async_session_maker() as session:
        org = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        other = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        await session.commit()
        await session.refresh(org)
        await session.refresh(other)
    await engine.dispose()
    app.dependency_overrides[current_verified_user] = lambda: org
    created = await api_client.post("/api/families", json={"name": "Ztest Fam"})
    fam_id = created.json()["id"]
    try:
        yield api_client, fam_id, org, other
    finally:
        app.dependency_overrides.pop(current_verified_user, None)


async def test_add_member_defaults(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "adult"
    assert body["perm_wiki"] == "edit"  # adult default = edit


async def test_create_child_with_synthetic_email(org_and_family):
    client, fam_id, *_ = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/children",
        json={"display_name": "Ztest Kid", "password": "kid-password-123"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "child"
    assert body["perm_wiki"] == "none"  # child default = none


async def test_patch_member_perms_and_gates(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "child"},
    )
    r = await client.patch(
        f"/api/families/{fam_id}/members/{other.id}",
        json={"perm_wiki": "view", "chat_enabled": False},
    )
    assert r.status_code == 200
    assert r.json()["perm_wiki"] == "view"
    assert r.json()["chat_enabled"] is False


async def test_admin_demotion_is_organizer_only(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    # organizer CAN change admin<->adult
    r = await client.patch(f"/api/families/{fam_id}/members/{other.id}", json={"role": "adult"})
    assert r.status_code == 200


async def test_organizer_cannot_be_removed(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.request("DELETE", f"/api/families/{fam_id}/members/{org.id}")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "organizer_must_transfer_first"


async def test_transfer_swaps_roles(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    r = await client.post(f"/api/families/{fam_id}/transfer", json={"user_id": str(other.id)})
    assert r.status_code == 200
    members_resp = await client.get(f"/api/families/{fam_id}/members")
    members = {m["user_id"]: m["role"] for m in members_resp.json()}
    assert members[str(other.id)] == "organizer"
    assert members[str(org.id)] == "admin"
