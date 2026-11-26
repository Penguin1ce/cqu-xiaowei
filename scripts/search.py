# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""知识库检索调试：混合召回 + Rerank，打印命中条款、分数和引用。

用法：
    uv run python scripts/search.py "请病假需要什么材料"
    uv run python scripts/search.py "国家奖学金多少钱" --affairs scholarship --no-draft
    uv run python scripts/search.py "转专业几次" --top-k 3 --full     # 打印完整正文
    uv run python scripts/search.py "转专业几次" --no-rerank          # 只看混合召回

每次检索调用一次 Embedding 和一次 Rerank（少量费用）。
"""

import argparse
import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from cqu_affairs.core.config import get_settings
from cqu_affairs.infra import milvus, mysql
from cqu_affairs.infra.mysql import get_engine
from cqu_affairs.llm import SiliconFlowReranker, build_embedder
from cqu_affairs.rag.retriever import PolicyRetriever, SearchFilters


async def main(args: argparse.Namespace) -> int:
    s = get_settings()
    mysql.init_mysql(s.mysql)
    client = milvus.init_milvus(s.milvus)
    reranker = None if args.no_rerank else SiliconFlowReranker(s.siliconflow)
    retriever = PolicyRetriever(
        client=client,
        collection=s.milvus.collection,
        embed=build_embedder(s.siliconflow),
        reranker=reranker,
        session_factory=async_sessionmaker(get_engine(), expire_on_commit=False),
        settings=s.rag,
    )
    filters = SearchFilters(
        affairs=args.affairs, category=args.category, include_draft=not args.no_draft
    )
    try:
        result = await retriever.search(args.query, filters, top_k=args.top_k)
    finally:
        if reranker:
            await reranker.aclose()
        await milvus.close_milvus()
        await mysql.close_mysql()

    mode = "Rerank" if result.reranked else "仅混合召回（无相关性分数）"
    print(f"Q: {result.query}    [{mode}，候选 {result.candidates} 个]\n")
    if not result.found:
        print(f"未找到依据（没有 Rerank 分数 ≥ {s.rag.min_score} 的条款）")
        return 0
    for i, h in enumerate(result.hits, 1):
        score = f"{h.score:.3f}" if h.score is not None else "  -  "
        print(f"{i}. [{score}] 召回第{h.recall_rank:2d}  {h.citation}")
        body = h.text if args.full else h.text[:160].replace("\n", " ")
        print(f"   {body}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--affairs", help="leave / scholarship / reimbursement / ...")
    parser.add_argument("--category", help="enrollment / leave / scholarship / ...")
    parser.add_argument("--no-draft", action="store_true", help="排除讨论稿")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--full", action="store_true", help="打印完整正文")
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main(parser.parse_args())))
