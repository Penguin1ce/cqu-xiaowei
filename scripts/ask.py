# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""命令行政策问答：流式打印回答，再列出引用与提示。

用法：
    uv run python scripts/ask.py "请病假需要什么材料"
    uv run python scripts/ask.py "国家奖学金多少钱" --no-draft
    uv run python scripts/ask.py "请病假需要什么材料" --no-cache   # 不读写 Redis 缓存
    uv run python scripts/ask.py "转专业几次" --sources            # 同时列出全部参考资料

每次调用 Embedding、Rerank 和 DeepSeek（少量费用）；命中缓存时不调用。
"""

import argparse
import asyncio
import logging
import sys
import time

from cqu_affairs.core.config import get_settings
from cqu_affairs.infra import milvus, mysql, redis
from cqu_affairs.infra.mysql import get_session_factory
from cqu_affairs.rag.retriever import SearchFilters
from cqu_affairs.services.qa import build_qa_service


async def main(args: argparse.Namespace) -> int:
    s = get_settings()
    mysql.init_mysql(s.mysql)
    client = milvus.init_milvus(s.milvus)
    r = None if args.no_cache else redis.init_redis(s.redis)
    svc = build_qa_service(s, milvus_client=client, redis=r, session_factory=get_session_factory())
    filters = SearchFilters(affairs=args.affairs, include_draft=not args.no_draft)

    start = time.perf_counter()
    first: float | None = None
    result = None
    streamed = ""
    try:
        print(f"Q: {args.question}\n")
        async for ev in svc.stream(args.question, filters):
            if ev.type == "delta":
                if first is None:
                    first = time.perf_counter() - start
                streamed += ev.data["text"]
                print(ev.data["text"], end="", flush=True)
            elif ev.type == "done":
                result = ev.data
            elif ev.type == "error":
                print(f"\n出错：{ev.data['message']}")
                return 1
    finally:
        await svc.aclose()
        await milvus.close_milvus()
        await mysql.close_mysql()
        if r is not None:
            await redis.close_redis()

    total = time.perf_counter() - start
    if streamed.strip() != result.answer:
        print(f"\n\n→ 最终回答：{result.answer}")
    print()
    for c in result.citations:
        print(f"  [{c.index}] {c.citation}")
    for n in result.notices:
        print(f"  ⚠ {n}")
    if args.sources:
        print("\n参考资料：")
        for c in result.sources:
            score = f"{c.score:.2f}" if c.score is not None else "-"
            print(f"  [{c.index}] ({score}) {c.citation}")
    tag = "命中缓存" if result.cached else f"首字 {first:.2f}s" if first else ""
    print(f"\n（{tag}，总 {total:.2f}s）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--affairs")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sources", action="store_true")
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main(parser.parse_args())))
