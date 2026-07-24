"""Enable the pgvector extension on DATABASE_URL. Run once before migrations.

Managed Postgres clusters (DO, RDS, etc.) don't pre-install pgvector; alembic
doesn't own extension management, so this runs as its own pre-deploy step.
"""

import asyncio
import os

import asyncpg


async def main() -> None:
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
