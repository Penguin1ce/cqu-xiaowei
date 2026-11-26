"""KnowledgeStore 集成测试：需要本地 Milvus（docker compose），连不上时跳过。"""

import uuid

import pytest
from pymilvus import AsyncMilvusClient

from cqu_affairs.core.config import MilvusSettings
from cqu_affairs.rag.store import ChunkRow, KnowledgeStore

DIM = 4


@pytest.fixture
async def store():
    uri = MilvusSettings(_env_file=None).uri
    client = AsyncMilvusClient(uri=uri, timeout=3)
    try:
        await client.list_collections()
    except Exception:  # noqa: BLE001
        await client.close()
        pytest.skip(f"Milvus 不可用：{uri}")
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    s = KnowledgeStore(client, name, dim=DIM, embedding_model="fake-model")
    yield s
    await client.drop_collection(name)
    await client.close()


def row(cid, doc, text, vec, **kw):
    base = dict(kind="article", status="effective", level="school", category="leave", affairs=[])
    return ChunkRow(chunk_id=cid, doc_id=doc, text=text, dense=vec, **(base | kw))


async def test_create_insert_count_delete_and_bm25(store):
    assert await store.ensure_collection() is True
    await store.insert(
        [
            row("d1-1", "d1", "学生因病请假的，应当出具校医院诊断证明", [1, 0, 0, 0]),
            row("d1-2", "d1", "国家励志奖学金每人每年6000元", [0, 1, 0, 0]),
            row("d2-1", "d2", "食堂营业时间为早上六点半", [0, 0, 1, 0], affairs=["life"]),
        ]
    )
    assert await store.count() == 3
    assert await store.count("d1") == 2
    assert await store.doc_ids() == {"d1", "d2"}

    hits = await store.client.search(
        store.collection,
        data=["请病假 诊断证明"],
        anns_field="sparse",
        limit=3,
        search_params={"metric_type": "BM25"},
    )
    assert hits[0][0]["chunk_id"] == "d1-1"

    await store.delete_document("d1")
    assert await store.count() == 1


async def test_existing_collection_with_other_model_is_rejected(store):
    await store.ensure_collection()
    assert await store.ensure_collection() is False  # 同配置可复用
    other = KnowledgeStore(store.client, store.collection, dim=DIM, embedding_model="other")
    with pytest.raises(RuntimeError, match="--rebuild"):
        await other.ensure_collection()
    assert await other.ensure_collection(rebuild=True) is True


async def test_insert_rejects_wrong_dimension(store):
    await store.ensure_collection()
    with pytest.raises(ValueError, match="维度"):
        await store.insert([row("x", "d", "t", [1.0, 0.0])])
