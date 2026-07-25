"""Alembic environment: async engine, URL from app settings.

An explicit ``sqlalchemy.url`` in the config (set programmatically by the
migration tests) wins over settings, so migrations can run against a
scratch database.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from cherryai_api.frontend_errors import FrontendError  # noqa: F401 (register tables)
from cherryai_api.orm import Base, sqlalchemy_url
from cherryai_api.users import AccessToken, User  # noqa: F401 (register tables)

config = context.config
if config.config_file_name is not None:
    # ``disable_existing_loggers=False`` is load-bearing, not cosmetic. This
    # env runs in-process inside the API (see db_migrations.py), not only
    # under the `alembic` CLI. fileConfig's default of True would disable
    # every logger absent from alembic.ini — including `uvicorn`,
    # `uvicorn.error` and `uvicorn.access` — for the life of the process,
    # silencing access logs and, worse, the "Application startup failed"
    # traceback, so a failed boot would exit with no explanation at all.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", sqlalchemy_url())

target_metadata = Base.metadata


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
