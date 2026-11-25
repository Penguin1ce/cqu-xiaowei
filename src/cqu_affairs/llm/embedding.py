# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""向量模型：硅基流动 Embedding（OpenAI 兼容接口，AgentScope `OpenAIEmbeddingModel`）。"""

from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel

from cqu_affairs.core.config import SiliconFlowSettings


def build_embedding_model(settings: SiliconFlowSettings) -> OpenAIEmbeddingModel:
    if settings.api_key is None:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY")
    return OpenAIEmbeddingModel(
        credential=OpenAICredential(api_key=settings.api_key, base_url=settings.base_url),
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        # 非 Qwen3-Embedding 模型不接受 dimensions 参数，传了会报 400
        pass_dimensions=settings.embedding_supports_dim,
    )
