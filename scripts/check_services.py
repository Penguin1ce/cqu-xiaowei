# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""连通性自检：依次检查 MySQL / Redis / Milvus 以及 DeepSeek / 通义 Embedding / 通义 Rerank。

用法：
    uv run python scripts/check_services.py            # 全部检查
    uv run python scripts/check_services.py --only api # 只检查模型 API
    uv run python scripts/check_services.py --only infra

模型 API 会产生少量费用（每项约几十个 token）。未配置 API Key 的项目会跳过。
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable

from agentscope.message import Msg, TextBlock

from cqu_affairs.core.config import Settings, get_settings
from cqu_affairs.infra import milvus, mysql, redis
from cqu_affairs.llm import DashScopeReranker, build_chat_model, build_embedding_model


class Skip(Exception):
    """该项未配置，跳过检查。"""


async def check_mysql(s: Settings) -> str:
    mysql.init_mysql(s.mysql)
    try:
        await mysql.ping_mysql()
    finally:
        await mysql.close_mysql()
    return f"{s.mysql.host}:{s.mysql.port}/{s.mysql.database}"


async def check_redis(s: Settings) -> str:
    redis.init_redis(s.redis)
    try:
        await redis.ping_redis()
    finally:
        await redis.close_redis()
    return s.redis.url


async def check_milvus(s: Settings) -> str:
    client = milvus.init_milvus(s.milvus)
    try:
        collections = await client.list_collections()
    finally:
        await milvus.close_milvus()
    return f"{s.milvus.uri}，已有集合 {len(collections)} 个"


async def check_deepseek(s: Settings) -> str:
    if s.deepseek.api_key is None:
        raise Skip("DEEPSEEK_API_KEY 未配置")
    model = build_chat_model(s.deepseek, stream=False)
    msg = Msg(name="user", role="user", content=[TextBlock(text="用一句话介绍你自己。")])
    resp = await model([msg])
    text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
    return f"{s.deepseek.model} 回复：{text[:40]}"


async def check_embedding(s: Settings) -> str:
    if s.dashscope.api_key is None:
        raise Skip("DASHSCOPE_API_KEY 未配置")
    model = build_embedding_model(s.dashscope)
    resp = await model(["学生请假需要哪些材料？"])
    dim = len(resp.embeddings[0])
    if dim != s.dashscope.embedding_dim:
        raise RuntimeError(f"向量维度 {dim} 与配置 {s.dashscope.embedding_dim} 不一致")
    return f"{s.dashscope.embedding_model}，维度 {dim}"


async def check_rerank(s: Settings) -> str:
    if s.dashscope.api_key is None:
        raise Skip("DASHSCOPE_API_KEY 未配置")
    reranker = DashScopeReranker(s.dashscope)
    docs = ["食堂营业时间为早上六点半至晚上九点。", "学生请病假需提交医院诊断证明并经辅导员审批。"]
    try:
        results = await reranker.rerank("请病假要交什么材料", docs)
    finally:
        await reranker.aclose()
    if results[0].index != 1:
        raise RuntimeError(f"排序结果不符合预期：{results}")
    return f"{s.dashscope.rerank_model}，最高分 {results[0].score:.3f}"


INFRA_CHECKS: dict[str, Callable[[Settings], Awaitable[str]]] = {
    "MySQL": check_mysql,
    "Redis": check_redis,
    "Milvus": check_milvus,
}
API_CHECKS: dict[str, Callable[[Settings], Awaitable[str]]] = {
    "DeepSeek": check_deepseek,
    "Embedding": check_embedding,
    "Rerank": check_rerank,
}


async def main(only: str | None) -> int:
    settings = get_settings()
    checks: dict[str, Callable[[Settings], Awaitable[str]]] = {}
    if only in (None, "infra"):
        checks |= INFRA_CHECKS
    if only in (None, "api"):
        checks |= API_CHECKS

    failed = 0
    for name, check in checks.items():
        start = time.perf_counter()
        try:
            detail = await asyncio.wait_for(check(settings), timeout=30)
            print(f"  ✓ {name:<10} {detail}  ({time.perf_counter() - start:.1f}s)")
        except Skip as e:
            print(f"  - {name:<10} 跳过：{e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name:<10} {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["infra", "api"])
    sys.exit(asyncio.run(main(parser.parse_args().only)))
