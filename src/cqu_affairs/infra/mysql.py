# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""MySQL 连接：异步引擎与会话工厂。

生命周期由 `main.py` 的 lifespan 管理：启动时 `init_mysql()`，关闭时 `close_mysql()`。
路由中通过 `api.deps.SessionDep` 注入 `AsyncSession`。
"""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cqu_affairs.core.config import MySQLSettings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_mysql(settings: MySQLSettings) -> AsyncEngine:
    global _engine, _session_factory
    _engine = create_async_engine(
        settings.url,
        pool_size=settings.pool_size,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.echo,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def close_mysql() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("MySQL 未初始化，请先调用 init_mysql()")
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每个请求一个会话。事务边界由业务层显式 commit。"""
    if _session_factory is None:
        raise RuntimeError("MySQL 未初始化，请先调用 init_mysql()")
    async with _session_factory() as session:
        yield session


async def ping_mysql() -> None:
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))
