"""Admin user management: queue, approve, reject, roles, deactivation."""

import pytest


async def _login_admin(client, make_user):
    admin = await make_user("ztest-admin@example.com", role="admin")
    res = await client.post(
        "/auth/login", data={"username": admin["email"], "password": admin["password"]}
    )
    assert res.status_code == 204
    return admin


@pytest.mark.asyncio
async def test_admin_routes_forbidden_for_chat_user(client, make_user):
    user = await make_user("ztest-notadmin@example.com", role="chat")
    await client.post("/auth/login", data={"username": user["email"], "password": user["password"]})
    res = await client.get("/admin/users")
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_pending_queue_and_approve(client, make_user):
    await _login_admin(client, make_user)
    pending = await make_user("ztest-queue@example.com", is_verified=False)
    res = await client.get("/admin/users", params={"status": "pending"})
    emails = [u["email"] for u in res.json()]
    assert pending["email"] in emails

    res = await client.post(f"/admin/users/{pending['id']}/approve", json={"role": "restricted"})
    assert res.status_code == 200
    assert res.json()["is_verified"] is True
    assert res.json()["role"] == "restricted"


@pytest.mark.asyncio
async def test_reject_deletes_pending_only(client, make_user, pool):
    await _login_admin(client, make_user)
    pending = await make_user("ztest-reject@example.com", is_verified=False)
    approved = await make_user("ztest-noreject@example.com", is_verified=True)

    res = await client.post(f"/admin/users/{pending['id']}/reject")
    assert res.status_code == 204
    assert await pool.fetchval('SELECT count(*) FROM "user" WHERE id = $1', pending["id"]) == 0

    res = await client.post(f"/admin/users/{approved['id']}/reject")
    assert res.status_code == 409


@pytest.mark.asyncio
async def test_deactivate_revokes_tokens_and_blocks(client, make_user, pool):
    await _login_admin(client, make_user)
    victim = await make_user("ztest-victim@example.com")
    # Log the victim in via a second client cookie jar shape: simplest is a
    # direct token count check after admin deactivation.
    import httpx

    transport = client._transport  # same ASGI app
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c2:
        res = await c2.post(
            "/auth/login",
            data={"username": victim["email"], "password": victim["password"]},
        )
        assert res.status_code == 204
        res = await client.post(f"/admin/users/{victim['id']}/deactivate")
        assert res.status_code == 200
        assert res.json()["is_active"] is False
        count = await pool.fetchval(
            "SELECT count(*) FROM accesstoken WHERE user_id = $1", victim["id"]
        )
        assert count == 0
        # The victim's existing cookie no longer authenticates.
        me = await c2.get("/users/me")
        assert me.status_code == 401


@pytest.mark.asyncio
async def test_admin_cannot_deactivate_or_demote_self(client, make_user):
    admin = await _login_admin(client, make_user)
    res = await client.post(f"/admin/users/{admin['id']}/deactivate")
    assert res.status_code == 409
    res = await client.patch(f"/admin/users/{admin['id']}", json={"role": "chat"})
    assert res.status_code == 409


@pytest.mark.asyncio
async def test_patch_role_syncs_superuser(client, make_user, pool):
    await _login_admin(client, make_user)
    user = await make_user("ztest-promote@example.com", role="chat")
    res = await client.patch(f"/admin/users/{user['id']}", json={"role": "admin"})
    assert res.status_code == 200
    row = await pool.fetchrow('SELECT role, is_superuser FROM "user" WHERE id = $1', user["id"])
    assert row["role"] == "admin" and row["is_superuser"] is True


@pytest.mark.asyncio
async def test_approve_unknown_user_is_404_even_with_bad_role(client, make_user):
    await _login_admin(client, make_user)
    import uuid

    res = await client.post(f"/admin/users/{uuid.uuid4()}/approve", json={"role": "bogus"})
    assert res.status_code == 404
