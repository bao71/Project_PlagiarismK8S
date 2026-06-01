import asyncpg
from plagiarism.app.core.config import settings

_pool: asyncpg.Pool | None = None


async def init_db():
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
    )


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_conn() -> asyncpg.Connection:
    if _pool is None:
        raise RuntimeError("Database pool chưa được khởi tạo")
    return await _pool.acquire()


async def release_conn(conn: asyncpg.Connection):
    if _pool:
        await _pool.release(conn)
