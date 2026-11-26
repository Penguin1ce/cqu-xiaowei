# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""政策检索：稠密 + BM25 混合召回 → RRF 融合 → Qwen3-Reranker 精排 → 阈值过滤。

    query ─┬─ Embedding ── dense  ANN (recall_k) ─┐
           └──────────────  BM25  ANN (recall_k) ─┴─ RRF → recall_k 个候选
                                                          │ MySQL 取正文与引用
                                                          ▼
                                       Rerank → score ≥ min_score 的前 top_k 条

- 结果为空（`found=False`）时，上层应回答“未找到依据”，不能让 LLM 自由发挥
- Rerank 调用失败时退回 RRF 排序（`reranked=False`），这时没有分数，无法判断相关性，
  上层应提示结果仅供参考
- 默认包含讨论稿（命中时引用里会标注“讨论稿”），`include_draft=False` 可排除
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from pydantic import BaseModel
from pymilvus import AnnSearchRequest, AsyncMilvusClient, RRFRanker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cqu_affairs.core.config import RagSettings
from cqu_affairs.llm.rerank import RerankResult
from cqu_affairs.models import KbChunk, KbDocument

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]: ...


@dataclass(frozen=True)
class SearchFilters:
    affairs: str | None = None
    """事务类型，如 leave / scholarship；对应 manifest 的 affairs。"""
    category: str | None = None
    """分类目录，如 enrollment。"""
    include_draft: bool = True
    kinds: Sequence[str] = field(default_factory=tuple)
    """限定块类型：notice / article / section / table，空表示不限。"""


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_filter_expr(f: SearchFilters) -> str:
    parts = []
    if f.affairs:
        parts.append(f"array_contains(affairs, {_quote(f.affairs)})")
    if f.category:
        parts.append(f"category == {_quote(f.category)}")
    if not f.include_draft:
        parts.append('status != "draft"')
    if f.kinds:
        parts.append(f"kind in [{', '.join(_quote(k) for k in f.kinds)}]")
    return " and ".join(parts)


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    citation: str
    location: str
    kind: str
    text: str
    status: str
    doc_no: str
    publish_date: date | None
    pages: list[int]
    score: float | None
    """Rerank 分数（0–1）；Rerank 失败退回 RRF 时为 None。"""
    recall_rank: int
    """在混合召回结果中的名次（1 起），便于调试召回质量。"""


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    reranked: bool
    candidates: int
    """混合召回得到的候选数。"""

    @property
    def found(self) -> bool:
        return bool(self.hits)


def _location(chunk: KbChunk) -> str:
    parts = list(chunk.heading_path)
    if chunk.article_no and chunk.article_no not in parts:
        parts.append(chunk.article_no)
    return " ".join(parts)


def rerank_text(chunk: KbChunk, doc: KbDocument, max_chars: int) -> str:
    """送入 Rerank 的文本：文件名 + 位置 + 正文（超长截断）。"""
    head = f"{doc.title} {_location(chunk)}".strip()
    return f"{head}\n{chunk.text[:max_chars]}"


class PolicyRetriever:
    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection: str,
        embed: Embedder,
        reranker: Reranker | None,
        session_factory: async_sessionmaker[AsyncSession],
        settings: RagSettings,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embed = embed
        self.reranker = reranker
        self.session_factory = session_factory
        self.s = settings

    async def _recall(self, query: str, vector: list[float], expr: str) -> list[str]:
        k = self.s.recall_k
        reqs = [
            AnnSearchRequest(
                data=[vector],
                anns_field="dense",
                param={"metric_type": "COSINE"},
                limit=k,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[query], anns_field="sparse", param={"metric_type": "BM25"}, limit=k, expr=expr
            ),
        ]
        res = await self.client.hybrid_search(
            self.collection, reqs, ranker=RRFRanker(self.s.rrf_k), limit=k
        )
        return [h["chunk_id"] for h in res[0]] if res else []

    async def _load(self, ids: list[str]) -> list[tuple[KbChunk, KbDocument]]:
        if not ids:
            return []
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(KbChunk, KbDocument)
                    .join(KbDocument, KbDocument.doc_id == KbChunk.doc_id)
                    .where(KbChunk.chunk_id.in_(ids))
                )
            ).all()
        by_id = {c.chunk_id: (c, d) for c, d in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            logger.warning(
                "Milvus 中有 %d 个块在 MySQL 里找不到，需重新入库：%s", len(missing), missing
            )
        return [by_id[i] for i in ids if i in by_id]

    async def search(
        self, query: str, filters: SearchFilters | None = None, *, top_k: int | None = None
    ) -> SearchResult:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        top_k = top_k or self.s.top_k
        expr = build_filter_expr(filters or SearchFilters())

        vector = (await self.embed([query]))[0]
        ids = await self._recall(query, vector, expr)
        candidates = await self._load(ids)

        def hit(i: int, chunk: KbChunk, doc: KbDocument, score: float | None) -> SearchHit:
            return SearchHit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=doc.title,
                citation=chunk.citation,
                location=_location(chunk),
                kind=chunk.kind,
                text=chunk.text,
                status=doc.status,
                doc_no=doc.doc_no,
                publish_date=doc.publish_date,
                pages=list(chunk.pages),
                score=score,
                recall_rank=i + 1,
            )

        if not candidates:
            return SearchResult(query=query, hits=[], reranked=False, candidates=0)

        if self.reranker is not None:
            docs = [rerank_text(c, d, self.s.rerank_max_chars) for c, d in candidates]
            try:
                ranked = await self.reranker.rerank(query, docs, top_n=len(docs))
            except Exception:  # noqa: BLE001
                logger.exception("Rerank 失败，退回 RRF 排序")
            else:
                hits = [
                    hit(r.index, *candidates[r.index], r.score)
                    for r in ranked
                    if r.score >= self.s.min_score
                ]
                return SearchResult(
                    query=query, hits=hits[:top_k], reranked=True, candidates=len(candidates)
                )

        hits = [hit(i, c, d, None) for i, (c, d) in enumerate(candidates[:top_k])]
        return SearchResult(query=query, hits=hits, reranked=False, candidates=len(candidates))
