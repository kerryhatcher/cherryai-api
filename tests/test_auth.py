"""Registration, login, pending gating, and /users/me."""

import pytest


async def _login(client, email: str, password: str):
    return await client.post("/auth/login", data={"username": email, "password": password})


@pytest.mark.asyncio
async def test_register_creates_pending_user(client, pool):
    res = await client.post(
        "/auth/register",
        json={
            "email": "ztest-reg@example.com",
            "password": "pw-ztest-123",
            "display_name": "Ztest Reg",
        },
    )
    assert res.status_code == 201
    body = res.json()
    assert body["is_verified"] is False
    assert body["role"] == "chat"
    row = await pool.fetchrow(
        "SELECT memory_dataset FROM \"user\" WHERE email = 'ztest-reg@example.com'"
    )
    assert row["memory_dataset"] == f"user-{body['id']}"


@pytest.mark.asyncio
async def test_login_sets_cookie_and_me_reports_pending(client, make_user):
    user = await make_user("ztest-pending@example.com", is_verified=False)
    res = await _login(client, user["email"], user["password"])
    assert res.status_code == 204
    assert "cherryai_auth" in res.cookies
    me = await client.get("/users/me")
    assert me.status_code == 200
    assert me.json()["is_verified"] is False


@pytest.mark.asyncio
async def test_login_rejects_bad_password(client, make_user):
    user = await make_user("ztest-badpw@example.com")
    res = await _login(client, user["email"], "wrong")
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_login_rejects_deactivated(client, make_user):
    user = await make_user("ztest-inactive@example.com", is_active=False)
    res = await _login(client, user["email"], user["password"])
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_logout_deletes_token_row(client, make_user, pool):
    user = await make_user("ztest-logout@example.com")
    await _login(client, user["email"], user["password"])
    count = await pool.fetchval("SELECT count(*) FROM accesstoken WHERE user_id = $1", user["id"])
    assert count == 1
    res = await client.post("/auth/logout")
    assert res.status_code == 204
    count = await pool.fetchval("SELECT count(*) FROM accesstoken WHERE user_id = $1", user["id"])
    assert count == 0
