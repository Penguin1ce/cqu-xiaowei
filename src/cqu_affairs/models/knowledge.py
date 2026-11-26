# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""知识库：文档与分块的元数据。

向量存在 Milvus（集合见 `rag/store.py`），两边用 `chunk_id` / `doc_id` 关联。
MySQL 是引用展示的数据源：检索拿到 chunk_id 后，从这里取正文、章条号、页码和引用文字。
"""

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cqu_affairs.models.base import Base, IdMixin, TimestampMixin


class KbDocument(IdMixin, TimestampMixin, Base):
    __tablename__ = "kb_document"

    doc_id: Mapped[str] = mapped_column(String(32), unique=True, comment="原文件 sha256 前 16 位")
    source_path: Mapped[str] = mapped_column(String(255), comment="相对 data/raw/ 的路径")
    title: Mapped[str] = mapped_column(String(255))
    issuer: Mapped[str] = mapped_column(String(128), default="")
    level: Mapped[str] = mapped_column(String(16), comment="school / college")
    category: Mapped[str] = mapped_column(String(32), index=True)
    doc_no: Mapped[str] = mapped_column(String(64), default="", comment="发文字号")
    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(16), comment="effective / draft")
    supersedes: Mapped[list[str]] = mapped_column(JSON, default=list, comment="废止的旧文件字号")
    structure: Mapped[str] = mapped_column(String(32))
    affairs: Mapped[list[str]] = mapped_column(JSON, default=list)

    sha256: Mapped[str] = mapped_column(String(64), comment="原文件指纹")
    chunks_hash: Mapped[str] = mapped_column(
        String(64), comment="向量指纹（模型 + 向量化文本 + 过滤字段），未变化时不重新向量化"
    )
    embedding_model: Mapped[str] = mapped_column(String(128))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime)

    chunks: Mapped[list["KbChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="KbChunk.seq",
    )


class KbChunk(IdMixin, TimestampMixin, Base):
    __tablename__ = "kb_chunk"
    __table_args__ = (Index("ix_kb_chunk_doc_seq", "doc_id", "seq"),)

    chunk_id: Mapped[str] = mapped_column(String(64), unique=True)
    doc_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("kb_document.doc_id", ondelete="CASCADE")
    )
    seq: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), comment="notice / article / section / table")
    text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    article_no: Mapped[str | None] = mapped_column(String(32), nullable=True)
    part: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pages: Mapped[list[int]] = mapped_column(JSON, default=list)
    citation: Mapped[str] = mapped_column(String(512), comment="回答中展示的引用文字")
    char_count: Mapped[int] = mapped_column(Integer)

    document: Mapped[KbDocument] = relationship(back_populates="chunks")
