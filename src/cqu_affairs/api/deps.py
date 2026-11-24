# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""FastAPI 公共依赖。路由中用类型别名声明依赖，例如 `async def f(session: SessionDep)`。"""

from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from cqu_affairs.core.config import Settings, get_settings
from cqu_affairs.infra.mysql import get_session
from cqu_affairs.infra.redis import get_redis

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]
