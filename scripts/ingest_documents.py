# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""把 data/parsed/ 下的文档分块、向量化，写入 Milvus（向量）和 MySQL（元数据与引用）。

用法：
    uv run python scripts/ingest_documents.py            # 增量入库，向量未变的文档不调 Embedding
    uv run python scripts/ingest_documents.py --dry-run  # 只统计块数，不写库、不调 API
    uv run python scripts/ingest_documents.py --force    # 全部重新向量化
    uv run python scripts/ingest_documents.py --rebuild  # 重建 Milvus 集合（换模型或维度时用）

前置：docker compose 已启动，且已执行 `uv run alembic upgrade head`。
会调用硅基流动 Embedding API（少量费用）。库中存在但 data/parsed 已没有的文档会被删除。
"""

import argparse
import asyncio
import json
import sys
import time

from sqlalchemy.ext.asyncio import async_sessionmaker

from cqu_affairs.core.config import get_settings
from cqu_affairs.infra import milvus, mysql
from cqu_affairs.infra.mysql import get_engine
from cqu_affairs.llm import build_embedder
from cqu_affairs.rag.chunking import chunk_document
from cqu_affairs.rag.ingest import DocumentInput, ingest_documents
from cqu_affairs.rag.parsing import PARSED_DIR
from cqu_affairs.rag.store import KnowledgeStore


def load_documents() -> list[DocumentInput]:
    docs = []
    for meta_path in sorted(PARSED_DIR.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        cl = json.loads((meta_path.parent / meta["outputs"]["content_list"]).read_text("utf-8"))
        docs.append(DocumentInput(meta=meta, chunks=chunk_document(meta, cl)))
    return docs


async def main(args: argparse.Namespace) -> int:
    docs = load_documents()
    if not docs:
        print("data/parsed 下没有解析结果，先运行 scripts/parse_documents.py")
        return 1
    n_chunks = sum(len(d.chunks) for d in docs)
    n_chars = sum(len(c.embedding_text()) for d in docs for c in d.chunks)
    print(f"共 {len(docs)} 份文档、{n_chunks} 个块、约 {n_chars} 字")
    if args.dry_run:
        return 0

    s = get_settings()
    mysql.init_mysql(s.mysql)
    client = milvus.init_milvus(s.milvus)
    session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    store = KnowledgeStore(
        client,
        s.milvus.collection,
        dim=s.siliconflow.embedding_dim,
        embedding_model=s.siliconflow.embedding_model,
    )
    embedder = build_embedder(s.siliconflow)
    start = time.perf_counter()
    try:
        if await store.ensure_collection(rebuild=args.rebuild):
            print(f"已创建 Milvus 集合 {s.milvus.collection}")
        report = await ingest_documents(
            docs,
            session_factory=session_factory,
            store=store,
            embed=embedder,
            force=args.force or args.rebuild,
        )
        total = await store.count()
    finally:
        await milvus.close_milvus()
        await mysql.close_mysql()

    print(
        f"\n向量化 {len(report.embedded)} 份（{report.chunks_embedded} 块，"
        f"{embedder.total_tokens} tokens），仅刷新元数据 {len(report.refreshed)} 份，"
        f"移除 {len(report.removed)} 份；Milvus 现有 {total} 块；"
        f"用时 {time.perf_counter() - start:.1f}s"
    )
    if total != n_chunks:
        print(f"⚠ Milvus 块数 {total} 与分块结果 {n_chunks} 不一致")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    sys.exit(asyncio.run(main(parser.parse_args())))
