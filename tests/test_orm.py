"""SQLAlchemy engine/session wiring."""

import uuid

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_get_async_session_executes_sql():
    from cherryai_api.orm import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1


def test_sqlalchemy_url_uses_asyncpg_driver():
    from cherryai_api.orm import sqlalchemy_url

    assert sqlalchemy_url().startswith("postgresql+asyncpg://")


_GUC_QUERY = text(
    "SELECT current_setting('app.user_id', true), current_setting('app.family_id', true)"
)


@pytest.mark.asyncio
async def test_after_begin_listener_stamps_gucs_from_contextvars():
    """The after_begin listener (orm.py) is the RLS backstop for every future
    SQLAlchemy-managed table: it must stamp app.user_id/app.family_id from
    family_context's ContextVars at the start of every ORM transaction.
    """
    from cherryai_api.family_context import active_family_var, current_user_var
    from cherryai_api.orm import async_session_maker

    user_id = uuid.uuid4()
    family_id = uuid.uuid4()
    user_token = current_user_var.set(user_id)
    family_token = active_family_var.set(family_id)
    try:
        async with async_session_maker() as session:
            row = (await session.execute(_GUC_QUERY)).one()
        assert row[0] == str(user_id)
        assert row[1] == str(family_id)
    finally:
        current_user_var.reset(user_token)
        active_family_var.reset(family_token)


@pytest.mark.asyncio
async def test_after_begin_listener_stamps_empty_string_when_contextvars_unset():
    """Outside a request (ContextVars at their None default), the listener
    must still stamp both GUCs -- as empty strings, not left unset -- so a
    stray query never inherits a *previous* transaction's identity.
    """
    from cherryai_api.orm import async_session_maker

    async with async_session_maker() as session:
        row = (await session.execute(_GUC_QUERY)).one()
    assert row[0] == ""
    assert row[1] == ""
