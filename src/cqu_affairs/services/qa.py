# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""RAG 问答服务：检索 → 时效过滤 → DeepSeek 生成 → 引用回查 → 缓存。

`stream()` 产出事件，供 SSE 接口逐条推送；`answer()` 收集事件返回最终结果。

    sources  {"found", "reranked", "sources": [Citation...]}   检索完成，前端可先展示参考资料
    delta    {"text": "..."}                                    回答的增量文本（含 [n] 编号）
    done     QAResult                                           最终结果：清理后的回答、引用、提示
    error    {"message": "..."}                                 出错，流结束

检索结果为空时不调用模型，直接返回“未找到依据”。命中问答缓存时一次性回放三个事件，
done 中 `cached=True`。
"""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from agentscope.message import TextBlock

from cqu_affairs.rag.answer import (
    QAResult,
    build_messages,
    drop_superseded,
    finalize,
    not_found_result,
    to_citation,
)
from cqu_affairs.rag.retriever import PolicyRetriever, SearchFilters
from cqu_affairs.services.cache import AnswerCache, kb_version

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAEvent:
    type: str
    data: Any


class QAService:
    def __init__(
        self,
        *,
        retriever: PolicyRetriever,
        chat_model: Any,
        cache: AnswerCache | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.retriever = retriever
        self.chat_model = chat_model
        self.cache = cache
        self.today = today
        self._closers: list[Callable[[], Any]] = []

    async def aclose(self) -> None:
        for close in self._closers:
            await close()

    async def _cache_key(self, question: str, filters: SearchFilters) -> str | None:
        if self.cache is None:
            return None
        try:
            version = await kb_version(self.retriever.session_factory)
        except Exception:  # noqa: BLE001
            logger.warning("读取知识库版本失败，跳过问答缓存", exc_info=True)
            return None
        f = asdict(filters) | {"kinds": list(filters.kinds), "today": self.today().isoformat()}
        return self.cache.key(version, question, f)

    async def stream(
        self, question: str, filters: SearchFilters | None = None
    ) -> AsyncIterator[QAEvent]:
        question = question.strip()
        filters = filters or SearchFilters()
        try:
            key = await self._cache_key(question, filters)
            if key and (hit := await self.cache.get(key)):  # type: ignore[union-attr]
                hit = hit.model_copy(update={"cached": True})
                yield QAEvent("sources", _sources_payload(hit))
                yield QAEvent("delta", {"text": hit.answer})
                yield QAEvent("done", hit)
                return

            res = await self.retriever.search(question, filters)
            hits, dropped = drop_superseded(res.hits)
            if dropped:
                logger.info("过滤已废止文件：%s", [h.doc_no for h in dropped])
            sources = [to_citation(i, h) for i, h in enumerate(hits, 1)]

            if not hits:
                result = not_found_result(question, reranked=res.reranked)
                yield QAEvent("sources", _sources_payload(result))
                yield QAEvent("delta", {"text": result.answer})
            else:
                yield QAEvent(
                    "sources",
                    {"found": True, "reranked": res.reranked, "sources": sources},
                )
                messages = build_messages(question, hits, self.today())
                raw = ""
                async for chunk in await self.chat_model(messages):
                    text = "".join(b.text for b in chunk.content if isinstance(b, TextBlock))
                    if chunk.is_last:
                        # 最后一个 chunk 是完整内容；以它为准，补发流中缺失的尾部
                        if text.startswith(raw) and len(text) > len(raw):
                            yield QAEvent("delta", {"text": text[len(raw) :]})
                        raw = text or raw
                    elif text:
                        raw += text
                        yield QAEvent("delta", {"text": text})
                result = finalize(question, raw, sources, res.reranked)

            if key:
                await self.cache.set(key, result)  # type: ignore[union-attr]
            yield QAEvent("done", result)
        except Exception as e:  # noqa: BLE001
            logger.exception("问答失败：%s", question)
            yield QAEvent("error", {"message": f"问答服务出错：{type(e).__name__}"})

    async def answer(self, question: str, filters: SearchFilters | None = None) -> QAResult:
        async for ev in self.stream(question, filters):
            if ev.type == "done":
                return ev.data
            if ev.type == "error":
                raise RuntimeError(ev.data["message"])
        raise RuntimeError("问答流程未产生结果")


def _sources_payload(r: QAResult) -> dict:
    return {"found": r.found, "reranked": r.reranked, "sources": r.sources}


def build_qa_service(settings, *, milvus_client, redis, session_factory) -> QAService:
    """按配置装配问答服务。缺少 DeepSeek / 硅基流动密钥时抛 RuntimeError。

    `redis=None` 时不启用缓存。返回的服务持有 Rerank 的 HTTP 客户端，
    关闭时调用 `await service.aclose()`。
    """
    from cqu_affairs.llm import SiliconFlowReranker, build_chat_model, build_embedder
    from cqu_affairs.services.cache import CachedEmbedder

    sf, rag = settings.siliconflow, settings.rag
    embed = build_embedder(sf)
    if redis is not None:
        embed = CachedEmbedder(
            embed, redis, sf.embedding_model, sf.embedding_dim, rag.embedding_cache_ttl
        )
    chat_model = build_chat_model(settings.deepseek, stream=True)
    reranker = SiliconFlowReranker(sf)
    retriever = PolicyRetriever(
        client=milvus_client,
        collection=settings.milvus.collection,
        embed=embed,
        reranker=reranker,
        session_factory=session_factory,
        settings=rag,
    )
    svc = QAService(
        retriever=retriever,
        chat_model=chat_model,
        cache=AnswerCache(redis, rag.answer_cache_ttl) if redis is not None else None,
    )
    svc._closers.append(reranker.aclose)
    return svc
