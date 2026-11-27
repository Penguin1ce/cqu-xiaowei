# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""政策问答接口。

- `POST /api/qa`：一次性返回 `QAResult`
- `POST /api/qa/stream`：SSE，依次推送 `sources` → 多个 `delta` → `done`（或 `error`）。
  `delta` 是模型的原始增量文本，最终展示以 `done.answer` 为准（越界编号已清理，
  “未找到依据”已替换为标准话术）
"""

from collections.abc import AsyncIterable

from fastapi import APIRouter, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from cqu_affairs.api.deps import QAServiceDep
from cqu_affairs.rag.answer import QAResult
from cqu_affairs.rag.retriever import SearchFilters

router = APIRouter(prefix="/api/qa", tags=["qa"])


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, examples=["请病假需要什么材料"])
    affairs: str | None = Field(None, description="事务类型过滤：leave / scholarship / ...")
    category: str | None = Field(None, description="分类过滤：enrollment / leave / ...")
    include_draft: bool = Field(True, description="是否包含讨论稿")

    def filters(self) -> SearchFilters:
        return SearchFilters(
            affairs=self.affairs, category=self.category, include_draft=self.include_draft
        )


@router.post("", response_model=QAResult)
async def ask(req: QARequest, svc: QAServiceDep) -> QAResult:
    try:
        return await svc.answer(req.question, req.filters())
    except RuntimeError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e


@router.post("/stream", response_class=EventSourceResponse)
async def ask_stream(req: QARequest, svc: QAServiceDep) -> AsyncIterable[ServerSentEvent]:
    async for ev in svc.stream(req.question, req.filters()):
        yield ServerSentEvent(data=ev.data, event=ev.type)
