# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""FastAPI 应用入口。

启动：`uv run cqu-affairs`，或 `uv run uvicorn cqu_affairs.main:app --reload`
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cqu_affairs.api import api_router
from cqu_affairs.core.config import get_settings
from cqu_affairs.core.logging import setup_logging
from cqu_affairs.infra.milvus import close_milvus, init_milvus
from cqu_affairs.infra.mysql import close_mysql, init_mysql
from cqu_affairs.infra.redis import close_redis, init_redis


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.app.upload_dir.mkdir(parents=True, exist_ok=True)
    # 这里只创建客户端，不主动建立连接；外部服务不可用时应用仍可启动，由 /health/ready 报告状态
    init_mysql(settings.mysql)
    init_redis(settings.redis)
    init_milvus(settings.milvus)
    try:
        yield
    finally:
        await close_milvus()
        await close_redis()
        await close_mysql()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.app.log_level)
    app = FastAPI(title=settings.app.name, debug=settings.app.debug, lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
