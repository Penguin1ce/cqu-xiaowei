# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""向量模型：通义 text-embedding（AgentScope `DashScopeEmbeddingModel`）。"""

from agentscope.credential import DashScopeCredential
from agentscope.embedding import DashScopeEmbeddingModel

from cqu_affairs.core.config import DashScopeSettings


def build_embedding_model(settings: DashScopeSettings) -> DashScopeEmbeddingModel:
    if settings.api_key is None:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    return DashScopeEmbeddingModel(
        credential=DashScopeCredential(api_key=settings.api_key, base_url=settings.base_url),
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
