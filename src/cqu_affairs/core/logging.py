# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""日志配置：统一格式，在应用启动时调用一次 `setup_logging`。"""

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    # 第三方库日志较多，调高级别
    for noisy in ("httpx", "httpcore", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
