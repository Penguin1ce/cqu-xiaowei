# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""健康检查。

- `GET /health`：进程存活检查，不访问外部服务
- `GET /health/ready`：逐个检查 MySQL、Redis、Milvus，任一失败返回 503
"""

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from cqu_affairs.infra.milvus import ping_milvus
from cqu_affairs.infra.mysql import ping_mysql
from cqu_affairs.infra.redis import ping_redis

router = APIRouter(prefix="/health", tags=["health"])

CHECKS: dict[str, Callable[[], Awaitable[None]]] = {
    "mysql": ping_mysql,
    "redis": ping_redis,
    "milvus": ping_milvus,
}
CHECK_TIMEOUT = 3.0


class ReadyResponse(BaseModel):
    ok: bool
    services: dict[str, str]
    """服务名 -> "ok" 或错误信息。"""


@router.get("")
async def live() -> dict[str, str]:
    return {"status": "ok"}


async def _run(check: Callable[[], Awaitable[None]]) -> str:
    try:
        await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT)
        return "ok"
    except Exception as e:  # noqa: BLE001 - 健康检查需要把任何异常都报告出来
        return f"error: {type(e).__name__}: {e}"


@router.get("/ready", response_model=ReadyResponse)
async def ready(response: Response) -> ReadyResponse:
    results = await asyncio.gather(*(_run(c) for c in CHECKS.values()))
    services = dict(zip(CHECKS, results, strict=True))
    ok = all(r == "ok" for r in results)
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(ok=ok, services=services)
