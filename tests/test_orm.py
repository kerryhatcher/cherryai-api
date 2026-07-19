"""SQLAlchemy engine/session wiring."""

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
