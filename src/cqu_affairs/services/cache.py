# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""Redis 缓存：查询向量与问答结果。

- 查询向量：key 含 Embedding 模型名和维度，与知识库内容无关，TTL 默认 7 天。
  主要解决硅基流动 8B Embedding 单次 0.14–7s 的延迟波动
- 问答结果：key 含知识库版本（kb_document 的文档数 + 最近入库时间），重新入库后自动失效

Redis 不可用时只记日志、直接走原逻辑，不影响回答。
"""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cqu_affairs.models import KbDocument
from cqu_affairs.rag.answer import QAResult

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


class CachedEmbedder:
    def __init__(self, inner: Embedder, redis: Redis, model: str, dim: int, ttl: int) -> None:
        self.inner = inner
        self.redis = redis
        self.prefix = f"emb:{_digest(model, str(dim))[:12]}:"
        self.ttl = ttl
        self.hits = 0

    def key(self, text: str) -> str:
        return self.prefix + _digest(text)

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        keys = [self.key(t) for t in texts]
        try:
            cached = await self.redis.mget(keys)
        except Exception:  # noqa: BLE001
            logger.warning("Redis 读取向量缓存失败，直接调用 Embedding", exc_info=True)
            return await self.inner(texts)

        out: list[list[float] | None] = [json.loads(v) if v else None for v in cached]
        self.hits += sum(v is not None for v in out)
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            vectors = await self.inner([texts[i] for i in missing])
            for i, v in zip(missing, vectors, strict=True):
                out[i] = v
            try:
                async with self.redis.pipeline(transaction=False) as pipe:
                    for i in missing:
                        pipe.set(keys[i], json.dumps(out[i]), ex=self.ttl)
                    await pipe.execute()
            except Exception:  # noqa: BLE001
                logger.warning("Redis 写入向量缓存失败", exc_info=True)
        return out  # type: ignore[return-value]


async def kb_version(session_factory: async_sessionmaker[AsyncSession]) -> str:
    """知识库版本：文档数 + 最近一次入库时间。每次运行入库脚本都会变化。"""
    async with session_factory() as s:
        count, latest = (
            await s.execute(select(func.count(KbDocument.id), func.max(KbDocument.ingested_at)))
        ).one()
    return _digest(str(count), str(latest))[:12]


class AnswerCache:
    def __init__(self, redis: Redis, ttl: int) -> None:
        self.redis = redis
        self.ttl = ttl

    @staticmethod
    def key(version: str, question: str, filters: dict) -> str:
        return f"qa:{version}:" + _digest(question, json.dumps(filters, sort_keys=True))

    async def get(self, key: str) -> QAResult | None:
        try:
            raw = await self.redis.get(key)
        except Exception:  # noqa: BLE001
            logger.warning("Redis 读取问答缓存失败", exc_info=True)
            return None
        return QAResult.model_validate_json(raw) if raw else None

    async def set(self, key: str, result: QAResult) -> None:
        try:
            await self.redis.set(key, result.model_dump_json(), ex=self.ttl)
        except Exception:  # noqa: BLE001
            logger.warning("Redis 写入问答缓存失败", exc_info=True)
