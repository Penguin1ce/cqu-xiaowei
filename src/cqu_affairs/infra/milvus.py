# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""Milvus 连接。

P0 只提供客户端和健康检查。集合结构（schema）和检索逻辑放在 P1 的 `rag/` 模块。
"""

from pymilvus import AsyncMilvusClient

from cqu_affairs.core.config import MilvusSettings

_client: AsyncMilvusClient | None = None


def init_milvus(settings: MilvusSettings) -> AsyncMilvusClient:
    global _client
    token = settings.token.get_secret_value() if settings.token else ""
    _client = AsyncMilvusClient(uri=settings.uri, token=token)
    return _client


async def close_milvus() -> None:
    global _client
    if _client is not None:
        await _client.close()
    _client = None


def get_milvus() -> AsyncMilvusClient:
    if _client is None:
        raise RuntimeError("Milvus 未初始化，请先调用 init_milvus()")
    return _client


async def ping_milvus() -> None:
    await get_milvus().list_collections()
