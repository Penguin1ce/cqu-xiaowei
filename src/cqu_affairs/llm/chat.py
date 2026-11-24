# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""对话模型：DeepSeek（AgentScope `DeepSeekChatModel`）。"""

from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel

from cqu_affairs.core.config import DeepSeekSettings


def build_chat_model(settings: DeepSeekSettings, *, stream: bool = True) -> DeepSeekChatModel:
    if settings.api_key is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    return DeepSeekChatModel(
        credential=DeepSeekCredential(api_key=settings.api_key, base_url=settings.base_url),
        model=settings.model,
        stream=stream,
    )
