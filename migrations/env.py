"""Alembic 迁移环境。

数据库连接串取自 `cqu_affairs.core.config`（即 .env），不使用 alembic.ini 中的 sqlalchemy.url。
表结构来源为 `cqu_affairs.models.Base.metadata`。新模型要在 `models/__init__.py` 中导入。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from cqu_affairs.core.config import get_settings
from cqu_affairs.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
DB_URL = get_settings().mysql.url


def run_migrations_offline() -> None:
    """离线模式：只输出 SQL，不连接数据库（alembic upgrade head --sql）。"""
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(DB_URL, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
