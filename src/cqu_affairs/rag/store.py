# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""Milvus 知识库集合：稠密向量 + BM25 稀疏向量，供混合检索使用。

不用 AgentScope 的 `MilvusLiteStore`：它只支持稠密检索，也不能设一致性级别。

集合字段：

    chunk_id   VARCHAR 主键，与 MySQL kb_chunk.chunk_id 对应
    doc_id     VARCHAR，按文档删除 / 过滤
    text       VARCHAR，中文分词（analyzer type=chinese），由 BM25 函数生成 sparse
    dense      FLOAT_VECTOR，Qwen3-Embedding，COSINE
    sparse     SPARSE_FLOAT_VECTOR，BM25 函数自动生成，写入时不传
    kind / status / level / category  VARCHAR 过滤字段
    affairs    ARRAY<VARCHAR>，用 array_contains(affairs, "leave") 过滤

text 存的是 `Chunk.embedding_text()`（标题路径 + 正文），BM25 也能匹配到文件名和章名里的词。
集合使用 Strong 一致性：写入后立即可检索。行数统计用 `count(*)`，
`get_collection_stats` 在 flush 前会返回 0。
"""

from dataclasses import dataclass

from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus.milvus_client.index import IndexParams

TEXT_MAX_LENGTH = 16384  # VARCHAR 按字节计，3000 汉字约 9000 字节
EMBEDDING_MODEL_KEY = "embedding_model"


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    text: str
    dense: list[float]
    kind: str
    status: str
    level: str
    category: str
    affairs: list[str]


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class KnowledgeStore:
    def __init__(self, client: AsyncMilvusClient, collection: str, dim: int, embedding_model: str):
        self.client = client
        self.collection = collection
        self.dim = dim
        self.embedding_model = embedding_model

    def _schema(self):
        s = AsyncMilvusClient.create_schema(
            auto_id=False,
            description=f"{EMBEDDING_MODEL_KEY}={self.embedding_model}",
        )
        s.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
        s.add_field("doc_id", DataType.VARCHAR, max_length=32)
        s.add_field(
            "text",
            DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_field("kind", DataType.VARCHAR, max_length=16)
        s.add_field("status", DataType.VARCHAR, max_length=16)
        s.add_field("level", DataType.VARCHAR, max_length=16)
        s.add_field("category", DataType.VARCHAR, max_length=32)
        s.add_field(
            "affairs", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16, max_length=32
        )
        s.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        return s

    async def ensure_collection(self, *, rebuild: bool = False) -> bool:
        """确保集合存在且向量维度、模型一致。新建返回 True。

        已有集合的维度或 Embedding 模型不同时抛错，需要 rebuild=True 重建（会清空向量）。
        """
        exists = await self.client.has_collection(self.collection)
        if exists and rebuild:
            await self.client.drop_collection(self.collection)
            exists = False
        if exists:
            desc = await self.client.describe_collection(self.collection)
            dim = next(
                (f["params"].get("dim") for f in desc["fields"] if f["name"] == "dense"), None
            )
            model = desc.get("description", "").removeprefix(f"{EMBEDDING_MODEL_KEY}=")
            if int(dim or 0) != self.dim or model != self.embedding_model:
                raise RuntimeError(
                    f"集合 {self.collection} 是 {model}（{dim} 维），与当前配置 "
                    f"{self.embedding_model}（{self.dim} 维）不一致，请用 --rebuild 重建"
                )
            await self.client.load_collection(self.collection)
            return False

        index = IndexParams()
        index.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        await self.client.create_collection(
            self.collection,
            schema=self._schema(),
            index_params=index,
            consistency_level="Strong",
        )
        return True

    async def delete_document(self, doc_id: str) -> None:
        await self.client.delete(self.collection, filter=f"doc_id == {_quote(doc_id)}")

    async def insert(self, rows: list[ChunkRow]) -> int:
        if not rows:
            return 0
        for r in rows:
            if len(r.dense) != self.dim:
                raise ValueError(f"{r.chunk_id} 向量维度 {len(r.dense)} ≠ {self.dim}")
        res = await self.client.insert(self.collection, [r.__dict__ for r in rows])
        return int(res["insert_count"])

    async def count(self, doc_id: str | None = None) -> int:
        flt = f"doc_id == {_quote(doc_id)}" if doc_id else ""
        res = await self.client.query(self.collection, filter=flt, output_fields=["count(*)"])
        return int(res[0]["count(*)"])

    async def doc_ids(self) -> set[str]:
        """集合中现有的全部 doc_id（文档数量很少，直接全量查询）。"""
        res = await self.client.query(
            self.collection, filter='doc_id != ""', output_fields=["doc_id"], limit=16384
        )
        return {r["doc_id"] for r in res}
