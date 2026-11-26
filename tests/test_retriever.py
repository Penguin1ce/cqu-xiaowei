import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cqu_affairs.core.config import RagSettings
from cqu_affairs.llm.rerank import RerankResult
from cqu_affairs.models import Base
from cqu_affairs.rag.chunking import chunk_document
from cqu_affairs.rag.ingest import DocumentInput, ingest_documents
from cqu_affairs.rag.retriever import PolicyRetriever, SearchFilters, build_filter_expr


class FakeStore:
    embedding_model = "fake"

    def __init__(self):
        self.rows = {}

    async def delete_document(self, doc_id):
        self.rows = {k: v for k, v in self.rows.items() if v.doc_id != doc_id}

    async def insert(self, rows):
        self.rows |= {r.chunk_id: r for r in rows}

    async def count(self, doc_id=None):
        return sum(doc_id in (None, r.doc_id) for r in self.rows.values())

    async def doc_ids(self):
        return {r.doc_id for r in self.rows.values()}


class FakeMilvus:
    """hybrid_search 按预设顺序返回 chunk_id，并记录收到的过滤表达式。"""

    def __init__(self, order):
        self.order = order
        self.exprs = []

    async def hybrid_search(self, collection, reqs, ranker=None, limit=10, **kw):
        self.exprs = [r.expr for r in reqs]
        return [[{"chunk_id": cid, "distance": 0.0} for cid in self.order[:limit]]]


class FakeReranker:
    def __init__(self, scores=None, fail=False):
        self.scores = scores or {}
        self.fail = fail
        self.docs = []

    async def rerank(self, query, documents, top_n=None):
        if self.fail:
            raise RuntimeError("rerank down")
        self.docs = documents
        res = [
            RerankResult(index=i, score=next((v for k, v in self.scores.items() if k in d), 0.0))
            for i, d in enumerate(documents)
        ]
        return sorted(res, key=lambda r: r.score, reverse=True)


async def fake_embed(texts):
    return [[0.0, 1.0] for _ in texts]


def make_doc(doc_id, title, articles, status="effective"):
    meta = {
        "doc_id": doc_id,
        "file": {"sha256": doc_id * 4, "ext": ".pdf"},
        "source": {
            "path": f"enrollment/{doc_id}.pdf",
            "title": title,
            "level": "school",
            "status": status,
            "structure": "chapter_article",
            "affairs": ["enrollment"],
            "doc_no": "重大校发〔2021〕61号",
            "publish_date": "2021-05-22",
        },
    }
    content = [{"type": "text", "text": "第一章 总则", "page_idx": 0}] + [
        {"type": "text", "text": a, "page_idx": 1} for a in articles
    ]
    return DocumentInput(meta=meta, chunks=chunk_document(meta, content))


A = "aaaa000011112222"
B = "bbbb000011112222"


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    docs = [
        make_doc(
            A, "转专业管理办法", ["第五条 学生只能转一次专业。", "第十三条 只允许填报一个专业。"]
        ),
        make_doc(B, "奖学金评定细则（讨论稿）", ["第一条 每人每年10000元。"], status="draft"),
    ]
    await ingest_documents(
        docs, session_factory=sf, store=FakeStore(), embed=fake_embed, log=lambda *_: None
    )
    yield sf
    await engine.dispose()


def make_retriever(sf, order, reranker, **settings):
    milvus = FakeMilvus(order)
    r = PolicyRetriever(
        client=milvus,
        collection="kb",
        embed=fake_embed,
        reranker=reranker,
        session_factory=sf,
        settings=RagSettings(_env_file=None, **settings),
    )
    return r, milvus


async def test_rerank_reorders_and_returns_citations(session_factory):
    rr = FakeReranker({"只能转一次": 0.98, "填报一个": 0.6})
    r, _ = make_retriever(session_factory, [f"{A}-0002", f"{A}-0001"], rr)
    res = await r.search("转专业能转几次")

    assert res.reranked and res.found and res.candidates == 2
    assert [h.chunk_id for h in res.hits] == [f"{A}-0001", f"{A}-0002"]
    top = res.hits[0]
    assert top.score == 0.98 and top.recall_rank == 2
    assert top.citation == "《转专业管理办法》第五条（重大校发〔2021〕61号，2021-05-22，第2页）"
    assert top.location == "第一章 总则 第五条"
    # 送入 Rerank 的文本带文件名和位置
    assert rr.docs[1].startswith("转专业管理办法 第一章 总则 第五条\n")


async def test_below_threshold_means_not_found(session_factory):
    r, _ = make_retriever(session_factory, [f"{A}-0001"], FakeReranker({"只能转一次": 0.05}))
    res = await r.search("食堂几点开门")
    assert not res.found and res.reranked


async def test_top_k_limits_hits(session_factory):
    rr = FakeReranker({"只能转一次": 0.9, "填报一个": 0.8, "10000": 0.7})
    r, _ = make_retriever(session_factory, [f"{A}-0001", f"{A}-0002", f"{B}-0001"], rr, top_k=2)
    assert len((await r.search("q")).hits) == 2


async def test_draft_is_marked_in_citation(session_factory):
    r, _ = make_retriever(session_factory, [f"{B}-0001"], FakeReranker({"10000": 0.95}))
    hit = (await r.search("奖学金多少钱")).hits[0]
    assert hit.status == "draft" and "讨论稿" in hit.citation


async def test_rerank_failure_falls_back_to_recall_order(session_factory):
    r, _ = make_retriever(
        session_factory, [f"{A}-0002", f"{A}-0001"], FakeReranker(fail=True), top_k=1
    )
    res = await r.search("q")
    assert not res.reranked
    assert [h.chunk_id for h in res.hits] == [f"{A}-0002"] and res.hits[0].score is None


async def test_missing_mysql_rows_are_skipped(session_factory):
    rr = FakeReranker({"只能转一次": 0.9})
    r, _ = make_retriever(session_factory, ["ghost-0001", f"{A}-0001"], rr)
    res = await r.search("q")
    assert res.candidates == 1 and res.hits[0].chunk_id == f"{A}-0001"


async def test_filters_passed_to_both_recall_paths(session_factory):
    r, milvus = make_retriever(session_factory, [], FakeReranker())
    res = await r.search("q", SearchFilters(affairs="leave", include_draft=False))
    assert not res.found
    expected = 'array_contains(affairs, "leave") and status != "draft"'
    assert milvus.exprs == [expected, expected]


async def test_empty_query_rejected(session_factory):
    r, _ = make_retriever(session_factory, [], FakeReranker())
    with pytest.raises(ValueError):
        await r.search("   ")


def test_build_filter_expr_quotes_values():
    expr = build_filter_expr(SearchFilters(category='a"b', kinds=["article", "table"]))
    assert expr == 'category == "a\\"b" and kind in ["article", "table"]'
    assert build_filter_expr(SearchFilters()) == ""
