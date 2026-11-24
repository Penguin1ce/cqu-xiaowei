import json

import httpx
import pytest

from cqu_affairs.core.config import DashScopeSettings
from cqu_affairs.llm.rerank import DashScopeReranker


@pytest.fixture
def settings() -> DashScopeSettings:
    return DashScopeSettings(api_key="sk-test", rerank_model="gte-rerank-v2")


async def test_rerank_sorts_by_score_and_sends_payload(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.1},
                        {"index": 1, "relevance_score": 0.9},
                    ]
                }
            },
        )

    reranker = DashScopeReranker(settings, transport=httpx.MockTransport(handler))
    results = await reranker.rerank("q", ["a", "b"], top_n=2)
    await reranker.aclose()

    assert [r.index for r in results] == [1, 0]
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["input"] == {"query": "q", "documents": ["a", "b"]}
    assert captured["body"]["parameters"]["top_n"] == 2


async def test_rerank_empty_documents_skips_request(settings):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("不应发起请求")

    reranker = DashScopeReranker(settings, transport=httpx.MockTransport(handler))
    assert await reranker.rerank("q", []) == []
    await reranker.aclose()


def test_rerank_requires_api_key():
    with pytest.raises(RuntimeError):
        DashScopeReranker(DashScopeSettings(api_key=None))
