# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""知识库入库：分块 → 向量化 → Milvus（向量）+ MySQL（元数据与引用）。

增量规则（按文档）：

- 向量指纹 = Embedding 模型 + 每个块的 chunk_id、向量化文本和过滤字段。指纹不变且 Milvus
  行数对得上时，不再调用 Embedding，只刷新 MySQL（发布日期、引用文字等元数据可能改了）
- 指纹变化：先向量化（失败则库中数据保持原样），再按 doc_id 替换 Milvus 与 MySQL 中的数据
- prune=True 时，删除库中存在、但 data/parsed 已没有的文档

MySQL 每份文档一个事务。Milvus 与 MySQL 不在同一事务里，中途失败时重跑即可修复：
指纹或行数对不上的文档会被重新入库。
"""

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cqu_affairs.models import KbChunk, KbDocument
from cqu_affairs.rag.chunking import Chunk
from cqu_affairs.rag.store import ChunkRow, KnowledgeStore

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass(frozen=True)
class DocumentInput:
    meta: dict
    """parsing 写出的 meta.json。"""
    chunks: list[Chunk]

    @property
    def doc_id(self) -> str:
        return self.meta["doc_id"]


@dataclass
class IngestReport:
    embedded: list[str] = field(default_factory=list)
    refreshed: list[str] = field(default_factory=list)
    """向量未变化，只刷新了 MySQL 元数据。"""
    removed: list[str] = field(default_factory=list)
    chunks_embedded: int = 0


def _row_filters(c: Chunk) -> dict:
    return {
        "kind": c.kind,
        "status": c.status,
        "level": c.level,
        "category": c.category,
        "affairs": c.affairs,
    }


def vector_hash(chunks: Sequence[Chunk], embedding_model: str) -> str:
    payload = [embedding_model] + [
        [c.chunk_id, c.embedding_text(), _row_filters(c)] for c in chunks
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


async def _write_mysql(
    session: AsyncSession, doc: DocumentInput, vhash: str, embedding_model: str
) -> None:
    src = doc.meta["source"]
    await session.execute(delete(KbChunk).where(KbChunk.doc_id == doc.doc_id))
    row = await session.scalar(select(KbDocument).where(KbDocument.doc_id == doc.doc_id))
    if row is None:
        row = KbDocument(doc_id=doc.doc_id)
        session.add(row)
    publish = doc.chunks[0].publish_date if doc.chunks else None
    row.source_path = src["path"]
    row.title = src["title"]
    row.issuer = src.get("issuer", "")
    row.level = src["level"]
    row.category = src["path"].split("/", 1)[0]
    row.doc_no = src.get("doc_no", "")
    row.publish_date = publish
    row.status = src["status"]
    row.supersedes = list(src.get("supersedes", []))
    row.structure = src["structure"]
    row.affairs = list(src.get("affairs", []))
    row.sha256 = doc.meta["file"]["sha256"]
    row.chunks_hash = vhash
    row.embedding_model = embedding_model
    row.chunk_count = len(doc.chunks)
    row.ingested_at = datetime.now()
    await session.flush()  # 先落 kb_document，满足 kb_chunk 外键
    session.add_all(
        KbChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            seq=c.seq,
            kind=c.kind,
            text=c.text,
            heading_path=c.heading_path,
            article_no=c.article_no,
            part=c.part,
            pages=c.pages,
            citation=c.citation,
            char_count=c.char_count,
        )
        for c in doc.chunks
    )


async def ingest_documents(
    docs: Sequence[DocumentInput],
    *,
    session_factory: async_sessionmaker[AsyncSession],
    store: KnowledgeStore,
    embed: Embedder,
    force: bool = False,
    prune: bool = True,
    log: Callable[[str], None] = print,
) -> IngestReport:
    report = IngestReport()
    model = store.embedding_model

    for doc in docs:
        vhash = vector_hash(doc.chunks, model)
        async with session_factory() as session:
            old = await session.scalar(
                select(KbDocument.chunks_hash).where(KbDocument.doc_id == doc.doc_id)
            )
        unchanged = not force and old == vhash and await store.count(doc.doc_id) == len(doc.chunks)
        title = doc.meta["source"]["title"]
        if not unchanged:
            vectors = await embed([c.embedding_text() for c in doc.chunks]) if doc.chunks else []
            if len(vectors) != len(doc.chunks):
                raise RuntimeError(f"{title}：向量数量 {len(vectors)} ≠ 块数量 {len(doc.chunks)}")
            await store.delete_document(doc.doc_id)
            await store.insert(
                [
                    ChunkRow(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        text=c.embedding_text(),
                        dense=v,
                        **_row_filters(c),
                    )
                    for c, v in zip(doc.chunks, vectors, strict=True)
                ]
            )
            report.embedded.append(doc.doc_id)
            report.chunks_embedded += len(doc.chunks)
            log(f"  ✓ 向量化 {len(doc.chunks):3d} 块  {title}")
        else:
            report.refreshed.append(doc.doc_id)
        async with session_factory() as session, session.begin():
            await _write_mysql(session, doc, vhash, model)

    if prune:
        keep = {d.doc_id for d in docs}
        async with session_factory() as session, session.begin():
            in_mysql = set(await session.scalars(select(KbDocument.doc_id)))
            stale = (in_mysql | await store.doc_ids()) - keep
            if stale:
                await session.execute(delete(KbChunk).where(KbChunk.doc_id.in_(stale)))
                await session.execute(delete(KbDocument).where(KbDocument.doc_id.in_(stale)))
        for doc_id in sorted(stale):
            await store.delete_document(doc_id)
            log(f"  - 移除已不存在的文档 {doc_id}")
        report.removed = sorted(stale)
    return report
