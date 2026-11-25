# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""模型接入层：对话（DeepSeek）、向量与重排序（硅基流动 Embedding / Rerank）。

上层（rag / agent）只通过这里的工厂函数获取模型实例，不直接拼装凭证。
"""

from cqu_affairs.llm.chat import build_chat_model
from cqu_affairs.llm.embedding import build_embedding_model
from cqu_affairs.llm.rerank import RerankResult, SiliconFlowReranker

__all__ = ["RerankResult", "SiliconFlowReranker", "build_chat_model", "build_embedding_model"]
