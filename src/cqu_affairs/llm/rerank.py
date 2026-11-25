# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""重排序模型：硅基流动 Rerank。

AgentScope 没有提供 Rerank 封装，这里直接调用硅基流动原生接口：

    POST {base_url}/rerank
    {"model": ..., "query": ..., "documents": [...], "top_n": ..., "return_documents": false}
    -> {"results": [{"index": 0, "relevance_score": 0.93}, ...]}
"""

from dataclasses import dataclass

import httpx

from cqu_affairs.core.config import SiliconFlowSettings


@dataclass(frozen=True)
class RerankResult:
    index: int
    """在输入 documents 中的下标。"""
    score: float


class SiliconFlowReranker:
    def __init__(
        self,
        settings: SiliconFlowSettings,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.api_key is None:
            raise RuntimeError("未配置 SILICONFLOW_API_KEY")
        self._url = settings.rerank_url
        self._model = settings.rerank_model
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=timeout,
            transport=transport,
        )

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """返回按相关度从高到低排序的结果。"""
        if not documents:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        results = resp.json()["results"]
        parsed = [RerankResult(index=r["index"], score=r["relevance_score"]) for r in results]
        return sorted(parsed, key=lambda r: r.score, reverse=True)

    async def aclose(self) -> None:
        await self._client.aclose()
