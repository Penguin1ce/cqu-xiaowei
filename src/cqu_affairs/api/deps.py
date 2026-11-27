# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""FastAPI 公共依赖。路由中用类型别名声明依赖，例如 `async def f(session: SessionDep)`。"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from cqu_affairs.core.config import Settings, get_settings
from cqu_affairs.infra.milvus import get_milvus
from cqu_affairs.infra.mysql import get_session, get_session_factory
from cqu_affairs.infra.redis import get_redis
from cqu_affairs.services.qa import QAService, build_qa_service

_qa_service: QAService | None = None


def get_qa_service() -> QAService:
    """问答服务单例，首次请求时装配；缺少模型密钥时返回 503，不影响应用其他接口。"""
    global _qa_service
    if _qa_service is None:
        try:
            _qa_service = build_qa_service(
                get_settings(),
                milvus_client=get_milvus(),
                redis=get_redis(),
                session_factory=get_session_factory(),
            )
        except RuntimeError as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    return _qa_service


async def close_qa_service() -> None:
    global _qa_service
    if _qa_service is not None:
        await _qa_service.aclose()
    _qa_service = None


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
