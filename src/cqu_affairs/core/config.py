# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""应用配置。

所有配置从环境变量或项目根目录的 `.env` 读取，按服务分组，每组用独立前缀：

    APP_*        应用自身
    MYSQL_*      MySQL
    REDIS_*      Redis
    MILVUS_*     Milvus
    DEEPSEEK_*   DeepSeek 对话模型
    SILICONFLOW_* 硅基流动 Embedding / Rerank
    MINERU_      MinerU 文档解析

业务代码统一通过 `get_settings()` 获取配置，不要直接读环境变量。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


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


class SiliconFlowSettings(_EnvSettings):
    """硅基流动：Embedding（OpenAI 兼容）与 Rerank（原生 /rerank 接口）共用一个密钥。"""

    model_config = SettingsConfigDict(env_prefix="SILICONFLOW_")

    api_key: SecretStr | None = None
    base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    # 仅 Qwen3-Embedding 系列支持自定义维度：8B ≤ 4096，4B ≤ 2560，0.6B ≤ 1024
    embedding_dim: int = 1024
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"

    @property
    def embedding_supports_dim(self) -> bool:
        return self.embedding_model.startswith("Qwen/Qwen3-Embedding")

    @property
    def rerank_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/rerank"


class MinerUSettings(_EnvSettings):
    """MinerU 云端精准解析 API（https://mineru.net/apiManage）。"""

    model_config = SettingsConfigDict(env_prefix="MINERU_")

    api_key: SecretStr | None = None
    base_url: str = "https://mineru.net/api/v4"
    # pipeline | vlm（官方推荐）
    model_version: str = "vlm"
    language: str = "ch"
    is_ocr: bool = False
    enable_table: bool = True
    enable_formula: bool = False
    poll_interval: float = 5.0
    # 整批任务的最长等待时间（秒）
    timeout: float = 1800.0


class Settings:
    """聚合各分组配置。"""

    def __init__(self) -> None:
        self.app = AppSettings()
        self.mysql = MySQLSettings()
        self.redis = RedisSettings()
        self.milvus = MilvusSettings()
        self.deepseek = DeepSeekSettings()
        self.siliconflow = SiliconFlowSettings()
        self.mineru = MinerUSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
