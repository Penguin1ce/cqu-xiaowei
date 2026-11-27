import json
from datetime import date

import pytest
from agentscope.message import TextBlock
from fastapi.testclient import TestClient
from test_answer import hit  # tests/ 无 __init__.py，pytest 会把该目录加入 sys.path

from cqu_affairs.rag.answer import NOT_FOUND_ANSWER
from cqu_affairs.rag.retriever import SearchFilters, SearchResult
from cqu_affairs.services.cache import AnswerCache, CachedEmbedder
from cqu_affairs.services.qa import QAService


class Chunk:
    def __init__(self, text, is_last=False):
        self.content = [TextBlock(text=text)]
        self.is_last = is_last


class FakeChat:
    """流式模型：逐段返回增量，最后一个 chunk 是完整内容。"""

    def __init__(self, parts, fail=False):
        self.parts = parts
        self.fail = fail
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("deepseek down")

        async def gen():
            for p in self.parts:
                yield Chunk(p)
            yield Chunk("".join(self.parts), is_last=True)

        return gen()


class FakeRetriever:
    def __init__(self, hits, reranked=True):
        self.hits = hits
        self.reranked = reranked
        self.calls = 0
        self.session_factory = None

    async def search(self, question, filters=None, top_k=None):
        self.calls += 1
        return SearchResult(
            query=question, hits=self.hits, reranked=self.reranked, candidates=len(self.hits)
        )


class FakeRedis:
    def __init__(self, broken=False):
        self.data = {}
        self.broken = broken

    def _check(self):
        if self.broken:
            raise ConnectionError("redis down")

    async def get(self, k):
        self._check()
        return self.data.get(k)

    async def mget(self, keys):
        self._check()
        return [self.data.get(k) for k in keys]

    async def set(self, k, v, ex=None):
        self._check()
        self.data[k] = v

    def pipeline(self, transaction=False):
        redis = self

        class P:
            ops = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def set(self, k, v, ex=None):
                self.ops.append((k, v))

            async def execute(self):
                redis._check()
                for k, v in self.ops:
                    redis.data[k] = v

        return P()


def make_service(hits, parts, *, cache=None, reranked=True, fail=False, monkeypatch=None):
    svc = QAService(
        retriever=FakeRetriever(hits, reranked),
        chat_model=FakeChat(parts, fail),
        cache=cache,
        today=lambda: date(2026, 11, 27),
    )
    if cache is not None and monkeypatch is not None:

        async def fake_version(_):
            return "v1"

        monkeypatch.setattr("cqu_affairs.services.qa.kb_version", fake_version)
    return svc


async def collect(svc, q="请病假要什么材料", filters=None):
    return [ev async for ev in svc.stream(q, filters)]


async def test_stream_event_order_and_final_result():
    svc = make_service([hit(1), hit(2)], ["需提交", "诊断证明[1]", "。"])
    evs = await collect(svc)

    assert [e.type for e in evs] == ["sources", "delta", "delta", "delta", "done"]
    assert len(evs[0].data["sources"]) == 2 and evs[0].data["found"]
    assert "".join(e.data["text"] for e in evs if e.type == "delta") == "需提交诊断证明[1]。"
    done = evs[-1].data
    assert done.found and [c.index for c in done.citations] == [1] and not done.cached


async def test_last_chunk_fills_missing_tail():
    svc = make_service([hit(1)], ["需提交"])
    svc.chat_model.parts = ["需提交"]

    async def call(messages):
        async def gen():
            yield Chunk("需提交")
            yield Chunk("需提交证明[1]。", is_last=True)

        return gen()

    svc.chat_model = call
    evs = await collect(svc)
    deltas = "".join(e.data["text"] for e in evs if e.type == "delta")
    assert deltas == "需提交证明[1]。" and evs[-1].data.answer == "需提交证明[1]。"


async def test_no_hits_short_circuits_without_calling_model():
    svc = make_service([], ["不应调用"])
    evs = await collect(svc, "食堂几点开门")
    assert svc.chat_model.calls == []
    assert evs[0].data["found"] is False
    assert evs[-1].type == "done" and evs[-1].data.answer == NOT_FOUND_ANSWER


async def test_superseded_document_removed_before_prompt():
    old = hit(1, doc_no="X〔2024〕16号")
    new = hit(2, doc_no="X〔2026〕17号", supersedes=["X〔2024〕16号"])
    svc = make_service([old, new], ["以新文件为准[1]。"])
    evs = await collect(svc)
    assert [s.chunk_id for s in evs[0].data["sources"]] == ["d2-0001"]
    prompt = svc.chat_model.calls[0][1].content[0].text
    assert "《文件1》" not in prompt and "《文件2》" in prompt


async def test_model_error_yields_error_event():
    svc = make_service([hit(1)], [], fail=True)
    evs = await collect(svc)
    assert evs[-1].type == "error" and "RuntimeError" in evs[-1].data["message"]
    with pytest.raises(RuntimeError):
        await svc.answer("q")


async def test_answer_cache_hit_skips_retrieval_and_model(monkeypatch):
    cache = AnswerCache(FakeRedis(), ttl=60)
    svc = make_service([hit(1)], ["答案[1]。"], cache=cache, monkeypatch=monkeypatch)
    first = await svc.answer("q")
    second_events = await collect(svc, "q")

    assert svc.retriever.calls == 1 and len(svc.chat_model.calls) == 1
    assert [e.type for e in second_events] == ["sources", "delta", "done"]
    second = second_events[-1].data
    assert second.cached and second.answer == first.answer


async def test_cache_key_depends_on_filters(monkeypatch):
    cache = AnswerCache(FakeRedis(), ttl=60)
    svc = make_service([hit(1)], ["答案[1]。"], cache=cache, monkeypatch=monkeypatch)
    await svc.answer("q")
    await svc.answer("q", SearchFilters(include_draft=False))
    assert svc.retriever.calls == 2


async def test_broken_redis_does_not_break_answer(monkeypatch):
    cache = AnswerCache(FakeRedis(broken=True), ttl=60)
    svc = make_service([hit(1)], ["答案[1]。"], cache=cache, monkeypatch=monkeypatch)
    assert (await svc.answer("q")).found


async def test_cached_embedder_reuses_vectors_and_survives_redis_failure():
    calls = []

    async def inner(texts):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    redis = FakeRedis()
    emb = CachedEmbedder(inner, redis, "m", 1, ttl=60)
    assert await emb(["甲", "乙乙"]) == [[1.0], [2.0]]
    assert await emb(["乙乙", "丙丙丙"]) == [[2.0], [3.0]]
    assert calls == [["甲", "乙乙"], ["丙丙丙"]] and emb.hits == 1

    redis.broken = True
    assert await emb(["甲"]) == [[1.0]]


# ── HTTP / SSE ──────────────────────────────────────────────────


@pytest.fixture
def client():
    from cqu_affairs.api.deps import get_qa_service
    from cqu_affairs.main import app

    svc = make_service([hit(1)], ["需提交", "证明[1]。"])
    app.dependency_overrides[get_qa_service] = lambda: svc
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_http_qa_returns_result(client):
    r = client.post("/api/qa", json={"question": "请病假要什么材料"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["citations"][0]["citation"].startswith("《文件1》")


def test_http_qa_validates_question(client):
    assert client.post("/api/qa", json={"question": ""}).status_code == 422


def test_sse_stream_events(client):
    with client.stream("POST", "/api/qa/stream", json={"question": "请病假"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [
        (blk.split("event: ")[1].split("\n")[0], json.loads(blk.split("data: ", 1)[1]))
        for blk in raw.strip().split("\n\n")
        if "event: " in blk
    ]
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    assert events[-1][1]["answer"] == "需提交证明[1]。"
