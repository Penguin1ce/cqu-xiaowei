# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""知识库原始文件清单 `data/raw/manifest.toml` 的读取与校验。

清单里的文档级元数据（标题、发文字号、发布日期、生效状态等）会随解析产物写入
`meta.json`，再一路带到分块和检索结果里，用于回答时展示引用来源。
"""

import hashlib
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from cqu_affairs.core.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.toml"


class DocumentMeta(BaseModel):
    """一份原始文件的文档级元数据，字段含义见 manifest.toml 文件头。"""

    path: str
    title: str
    issuer: str = ""
    level: Literal["school", "college"]
    doc_no: str = ""
    publish_date: date | None = None
    status: Literal["effective", "draft"]
    supersedes: list[str] = Field(default_factory=list)
    structure: Literal["chapter_article", "numbered", "form", "steps"]
    affairs: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("publish_date", mode="before")
    @classmethod
    def _empty_date(cls, v: object) -> object:
        return None if v == "" else v

    @property
    def category(self) -> str:
        """分类目录名，如 `scholarship`。"""
        return Path(self.path).parts[0]

    @property
    def stem(self) -> str:
        return Path(self.path).stem


def load_manifest(path: Path = MANIFEST_PATH) -> list[DocumentMeta]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    docs = [DocumentMeta.model_validate(d) for d in raw.get("documents", [])]
    dup = {d.path for d in docs if sum(x.path == d.path for x in docs) > 1}
    if dup:
        raise ValueError(f"manifest 中路径重复：{sorted(dup)}")
    return docs


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
