import json

import httpx
import pytest

from cqu_affairs.core.config import SiliconFlowSettings
from cqu_affairs.llm.rerank import SiliconFlowReranker


@pytest.fixture
def settings() -> SiliconFlowSettings:
    return SiliconFlowSettings(api_key="sk-test", rerank_model="Qwen/Qwen3-Reranker-8B")


async def test_rerank_sorts_by_score_and_sends_payload(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                ],
            },
        )

    reranker = SiliconFlowReranker(settings, transport=httpx.MockTransport(handler))
    results = await reranker.rerank("q", ["a", "b"], top_n=2)
    await reranker.aclose()

    assert [r.index for r in results] == [1, 0]
    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["query"] == "q"
    assert captured["body"]["documents"] == ["a", "b"]
    assert captured["body"]["top_n"] == 2
    assert captured["body"]["model"] == "Qwen/Qwen3-Reranker-8B"


async def test_rerank_empty_documents_skips_request(settings):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("不应发起请求")

    reranker = SiliconFlowReranker(settings, transport=httpx.MockTransport(handler))
    assert await reranker.rerank("q", []) == []
    await reranker.aclose()


def test_rerank_requires_api_key():
    with pytest.raises(RuntimeError):
        SiliconFlowReranker(SiliconFlowSettings(api_key=None, _env_file=None))


def test_embedding_dimensions_only_for_qwen3():
    assert SiliconFlowSettings(embedding_model="Qwen/Qwen3-Embedding-4B").embedding_supports_dim
    assert not SiliconFlowSettings(embedding_model="BAAI/bge-m3").embedding_supports_dim
