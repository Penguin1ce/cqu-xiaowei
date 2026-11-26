# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""向量模型：硅基流动 Embedding（OpenAI 兼容接口，AgentScope `OpenAIEmbeddingModel`）。"""

from collections.abc import Callable

from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel

from cqu_affairs.core.config import SiliconFlowSettings

# 单次请求的文本条数。AgentScope 默认 2048 条一批，硅基流动对单次输入总量有限制，取保守值
EMBED_BATCH_SIZE = 16


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


class Embedder:
    """分批调用 Embedding，累计 token 用量。实例可直接作为 `rag.ingest.Embedder` 使用。"""

    def __init__(
        self,
        model: OpenAIEmbeddingModel,
        batch_size: int = EMBED_BATCH_SIZE,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.on_batch = on_batch
        self.total_tokens = 0

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = await self.model(texts[i : i + self.batch_size])
            out.extend(resp.embeddings)
            if resp.usage is not None:
                self.total_tokens += resp.usage.tokens or 0
            if self.on_batch:
                self.on_batch(len(out), len(texts))
        return out


def build_embedder(settings: SiliconFlowSettings, **kwargs) -> Embedder:
    return Embedder(build_embedding_model(settings), **kwargs)
