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


@pytest.fixture()
async def admin_caller(org_and_family):
    """A third member, added as `admin` by the organizer, for tests that
    need an admin (not organizer) caller distinct from the target member."""
    client, fam_id, org, other = org_and_family
    async with async_session_maker() as session:
        admin_user = await _mk_user(session, f"ztest-{uuid.uuid4().hex[:8]}@example.com")
        await session.commit()
        await session.refresh(admin_user)
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": admin_user.email, "role": "admin"},
    )
    assert r.status_code == 201
    return client, fam_id, org, other, admin_user


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


async def test_organizer_can_change_admin_to_adult(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    # organizer CAN change admin<->adult
    r = await client.patch(f"/api/families/{fam_id}/members/{other.id}", json={"role": "adult"})
    assert r.status_code == 200


async def test_admin_cannot_change_admin_to_adult(admin_caller):
    client, fam_id, org, other, admin_user = admin_caller
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    app.dependency_overrides[current_verified_user] = lambda: admin_user
    r = await client.patch(f"/api/families/{fam_id}/members/{other.id}", json={"role": "adult"})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "family_permission_denied"


async def test_organizer_can_add_admin(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "admin"


async def test_admin_cannot_add_admin(admin_caller):
    client, fam_id, org, other, admin_user = admin_caller
    app.dependency_overrides[current_verified_user] = lambda: admin_user
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "family_permission_denied"


async def test_organizer_can_delete_admin(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    r = await client.request("DELETE", f"/api/families/{fam_id}/members/{other.id}")
    assert r.status_code == 204


async def test_admin_cannot_delete_admin(admin_caller):
    client, fam_id, org, other, admin_user = admin_caller
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "admin"},
    )
    app.dependency_overrides[current_verified_user] = lambda: admin_user
    r = await client.request("DELETE", f"/api/families/{fam_id}/members/{other.id}")
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "family_permission_denied"


async def test_add_member_duplicate_returns_409(org_and_family):
    """Re-adding an already-present member violates uq_membership_family_user
    at the DB level; the route must map that to a 409, not a 500."""
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    assert r.status_code == 201
    r2 = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == "already_member"


async def test_add_member_unknown_email_404(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": "ztest-nobody-zzz@example.com", "role": "adult"},
    )
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_such_user"


async def test_patch_organizer_row_rejected(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.patch(f"/api/families/{fam_id}/members/{org.id}", json={"perm_wiki": "view"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "cannot_modify_organizer"


async def test_patch_role_to_organizer_rejected(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    # "organizer" isn't in MemberPatch.role's Literal (admin|adult|child), so
    # this is now rejected by schema validation before it reaches the
    # (still-present, now unreachable-via-HTTP) use_transfer business check.
    r = await client.patch(f"/api/families/{fam_id}/members/{other.id}", json={"role": "organizer"})
    assert r.status_code == 422


async def test_patch_rejects_unknown_field(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    r = await client.patch(
        f"/api/families/{fam_id}/members/{other.id}", json={"email": "x@example.com"}
    )
    assert r.status_code == 422


async def test_transfer_to_non_admin_rejected(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    r = await client.post(f"/api/families/{fam_id}/transfer", json={"user_id": str(other.id)})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "transfer_target_not_admin"


async def test_non_member_gets_404_on_members_route(org_and_family):
    client, fam_id, org, other = org_and_family
    # `other` was never added to the family — existence-protecting (spec §7)
    app.dependency_overrides[current_verified_user] = lambda: other
    r = await client.get(f"/api/families/{fam_id}/members")
    assert r.status_code == 404


async def test_organizer_cannot_be_removed(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.request("DELETE", f"/api/families/{fam_id}/members/{org.id}")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "organizer_must_transfer_first"


async def test_patch_rejects_invalid_role(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    r = await client.patch(f"/api/families/{fam_id}/members/{other.id}", json={"role": "banana"})
    assert r.status_code == 422


async def test_patch_rejects_invalid_perm(org_and_family):
    client, fam_id, org, other = org_and_family
    await client.post(
        f"/api/families/{fam_id}/members",
        json={"email": other.email, "role": "adult"},
    )
    r = await client.patch(
        f"/api/families/{fam_id}/members/{other.id}", json={"perm_wiki": "banana"}
    )
    assert r.status_code == 422


async def test_create_child_duplicate_email_returns_409(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/children",
        json={"display_name": "Ztest Kid", "password": "kid-password-123", "email": other.email},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "email_taken"


async def test_create_child_malformed_email_returns_422(org_and_family):
    client, fam_id, org, other = org_and_family
    r = await client.post(
        f"/api/families/{fam_id}/children",
        json={
            "display_name": "Ztest Kid",
            "password": "kid-password-123",
            "email": "not-an-email",
        },
    )
    assert r.status_code == 422


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
