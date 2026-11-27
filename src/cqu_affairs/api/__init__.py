# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP 接口层。新增路由模块后在 `api_router` 中注册。"""

from fastapi import APIRouter

from cqu_affairs.api.routes import health, qa

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(qa.router)
