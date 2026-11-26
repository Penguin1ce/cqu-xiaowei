import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cqu_affairs.models import Base, KbChunk, KbDocument
from cqu_affairs.rag.chunking import chunk_document
from cqu_affairs.rag.ingest import DocumentInput, ingest_documents
from cqu_affairs.rag.store import ChunkRow

MODEL = "fake-embed"
DIM = 3


class FakeStore:
    """内存版 KnowledgeStore。"""

    def __init__(self):
        self.embedding_model = MODEL
        self.rows: dict[str, ChunkRow] = {}

    async def delete_document(self, doc_id):
        self.rows = {k: v for k, v in self.rows.items() if v.doc_id != doc_id}

    async def insert(self, rows):
        for r in rows:
            self.rows[r.chunk_id] = r
        return len(rows)

    async def count(self, doc_id=None):
        return sum(1 for r in self.rows.values() if doc_id is None or r.doc_id == doc_id)

    async def doc_ids(self):
        return {r.doc_id for r in self.rows.values()}


class FakeEmbed:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def __call__(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding api down")
        return [[float(len(t)), 0.0, 1.0] for t in texts]


def make_doc(doc_id: str, articles: list[str], **src) -> DocumentInput:
    meta = {
        "doc_id": doc_id,
        "file": {"sha256": doc_id * 4, "ext": ".pdf"},
        "source": {
            "path": f"leave/{doc_id}.pdf",
            "title": f"测试办法{doc_id}",
            "level": "school",
            "status": "effective",
            "structure": "chapter_article",
            "affairs": ["leave"],
            "doc_no": "X〔2021〕1号",
            "publish_date": "2021-05-17",
        }
        | src,
    }
    content = [{"type": "text", "text": "第一章 总则", "page_idx": 0}] + [
        {"type": "text", "text": a, "page_idx": i} for i, a in enumerate(articles)
    ]
    return DocumentInput(meta=meta, chunks=chunk_document(meta, content))


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _run(docs, sf, store, embed, **kw):
    return await ingest_documents(
        docs, session_factory=sf, store=store, embed=embed, log=lambda *_: None, **kw
    )


async def test_first_ingest_writes_milvus_and_mysql(session_factory):
    store, embed = FakeStore(), FakeEmbed()
    doc = make_doc("aaaa000011112222", ["第一条 学生请假应事先办理。", "第二条 病假需诊断证明。"])
    report = await _run([doc], session_factory, store, embed)

    assert report.embedded == ["aaaa000011112222"] and report.chunks_embedded == 2
    assert await store.count() == 2
    row = store.rows["aaaa000011112222-0001"]
    assert row.text.startswith("测试办法aaaa000011112222 > 第一章 总则\n第一条")
    assert row.status == "effective" and row.affairs == ["leave"] and row.kind == "article"

    async with session_factory() as s:
        d = await s.scalar(select(KbDocument))
        chunks = (await s.scalars(select(KbChunk).order_by(KbChunk.seq))).all()
    assert d.chunk_count == 2 and d.embedding_model == MODEL and d.doc_no == "X〔2021〕1号"
    assert [c.article_no for c in chunks] == ["第一条", "第二条"]
    assert (
        chunks[1].citation
        == "《测试办法aaaa000011112222》第二条（X〔2021〕1号，2021-05-17，第2页）"
    )


async def test_unchanged_doc_skips_embedding_but_refreshes_metadata(session_factory):
    store, embed = FakeStore(), FakeEmbed()
    articles = ["第一条 学生请假应事先办理。"]
    await _run([make_doc("aaaa000011112222", articles)], session_factory, store, embed)
    # 只改发布日期：向量文本不变，不应再调用 Embedding，但 MySQL 引用要更新
    doc2 = make_doc("aaaa000011112222", articles, publish_date="2022-01-01")
    report = await _run([doc2], session_factory, store, embed)

    assert embed.calls == 1
    assert report.refreshed == ["aaaa000011112222"] and report.embedded == []
    async with session_factory() as s:
        c = await s.scalar(select(KbChunk))
    assert "2022-01-01" in c.citation


async def test_changed_status_reembeds(session_factory):
    store, embed = FakeStore(), FakeEmbed()
    articles = ["第一条 学生请假应事先办理。"]
    await _run([make_doc("aaaa000011112222", articles)], session_factory, store, embed)
    await _run(
        [make_doc("aaaa000011112222", articles, status="draft")], session_factory, store, embed
    )
    assert embed.calls == 2
    assert store.rows["aaaa000011112222-0001"].status == "draft"


async def test_prune_removes_missing_documents(session_factory):
    store, embed = FakeStore(), FakeEmbed()
    a = make_doc("aaaa000011112222", ["第一条 甲。"])
    b = make_doc("bbbb000011112222", ["第一条 乙。"])
    await _run([a, b], session_factory, store, embed)
    report = await _run([a], session_factory, store, embed)

    assert report.removed == ["bbbb000011112222"]
    assert await store.doc_ids() == {"aaaa000011112222"}
    async with session_factory() as s:
        assert set(await s.scalars(select(KbDocument.doc_id))) == {"aaaa000011112222"}
        assert {c.doc_id for c in await s.scalars(select(KbChunk))} == {"aaaa000011112222"}


async def test_embedding_failure_leaves_existing_data(session_factory):
    store = FakeStore()
    await _run([make_doc("aaaa000011112222", ["第一条 旧。"])], session_factory, store, FakeEmbed())
    with pytest.raises(RuntimeError, match="embedding api down"):
        await _run(
            [make_doc("aaaa000011112222", ["第一条 新。"])],
            session_factory,
            store,
            FakeEmbed(fail=True),
        )
    assert "旧" in store.rows["aaaa000011112222-0001"].text
    async with session_factory() as s:
        assert "旧" in (await s.scalar(select(KbChunk))).text


async def test_missing_milvus_rows_trigger_reingest(session_factory):
    store, embed = FakeStore(), FakeEmbed()
    doc = make_doc("aaaa000011112222", ["第一条 甲。"])
    await _run([doc], session_factory, store, embed)
    await store.delete_document("aaaa000011112222")  # 模拟上次中途失败
    await _run([doc], session_factory, store, embed)
    assert embed.calls == 2 and await store.count() == 1
