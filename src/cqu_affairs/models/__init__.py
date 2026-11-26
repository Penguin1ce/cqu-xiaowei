# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""ORM 模型。

新增表模型后务必在这里导入，Alembic 才能通过 `Base.metadata` 发现它。
"""

from cqu_affairs.models.base import Base, IdMixin, TimestampMixin
from cqu_affairs.models.knowledge import KbChunk, KbDocument

__all__ = ["Base", "IdMixin", "KbChunk", "KbDocument", "TimestampMixin"]
