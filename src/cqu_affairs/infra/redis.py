# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""Redis 连接：用于缓存、会话短期记忆、限流和提醒队列。"""

from redis.asyncio import Redis

from cqu_affairs.core.config import RedisSettings

_client: Redis | None = None


def init_redis(settings: RedisSettings) -> Redis:
    global _client
    _client = Redis.from_url(settings.url, decode_responses=True)
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


def get_redis() -> Redis:
    if _client is None:
        raise RuntimeError("Redis 未初始化，请先调用 init_redis()")
    return _client


async def ping_redis() -> None:
    await get_redis().ping()
