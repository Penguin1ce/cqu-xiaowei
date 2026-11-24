# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""应用配置。

所有配置从环境变量或项目根目录的 `.env` 读取，按服务分组，每组用独立前缀：

    APP_*        应用自身
    MYSQL_*      MySQL
    REDIS_*      Redis
    MILVUS_*     Milvus
    DEEPSEEK_*   DeepSeek 对话模型
    DASHSCOPE_*  通义 Embedding / Rerank

业务代码统一通过 `get_settings()` 获取配置，不要直接读环境变量。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _EnvSettings(BaseSettings):
    """公共基类：从项目根目录的 .env 读取，忽略未声明的变量。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `KEY=` 视为未设置，走默认值
        extra="ignore",
    )


class AppSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")

    name: str = "cqu-affairs"
    env: str = "dev"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"


class MySQLSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="MYSQL_")

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "cqu"
    password: SecretStr = SecretStr("cqu")
    database: str = "cqu_affairs"
    pool_size: int = 10
    echo: bool = False

    @property
    def url(self) -> str:
        """SQLAlchemy 异步连接串（asyncmy 驱动）。"""
        return (
            f"mysql+asyncmy://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )


class RedisSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = "redis://127.0.0.1:6379/0"


class MilvusSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    uri: str = "http://127.0.0.1:19530"
    token: SecretStr | None = None
    collection: str = "campus_kb"


class DeepSeekSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_")

    api_key: SecretStr | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


class DashScopeSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="DASHSCOPE_")

    api_key: SecretStr | None = None
    # OpenAI 兼容地址，Embedding 使用
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 原生 Rerank 接口地址
    rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    rerank_model: str = "gte-rerank-v2"


class Settings:
    """聚合各分组配置。"""

    def __init__(self) -> None:
        self.app = AppSettings()
        self.mysql = MySQLSettings()
        self.redis = RedisSettings()
        self.milvus = MilvusSettings()
        self.deepseek = DeepSeekSettings()
        self.dashscope = DashScopeSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
