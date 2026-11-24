# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""基础设施连接：MySQL、Redis、Milvus。

每个模块对外提供相同的四个函数：`init_*`、`close_*`、`get_*`、`ping_*`。
"""
