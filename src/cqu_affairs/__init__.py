# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""校园事务智能办理系统。"""


def main() -> None:
    """命令行入口 `cqu-affairs`：启动 API 服务。"""
    import uvicorn

    from cqu_affairs.core.config import get_settings

    app = get_settings().app
    uvicorn.run("cqu_affairs.main:app", host=app.host, port=app.port, reload=app.env == "dev")
