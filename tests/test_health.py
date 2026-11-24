from fastapi.testclient import TestClient

from cqu_affairs.api.routes import health
from cqu_affairs.main import create_app


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionError("refused")


def test_live():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_ready_all_ok(monkeypatch):
    monkeypatch.setattr(health, "CHECKS", {"mysql": _ok, "redis": _ok})
    with TestClient(create_app()) as client:
        resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "services": {"mysql": "ok", "redis": "ok"}}


def test_ready_reports_failure(monkeypatch):
    monkeypatch.setattr(health, "CHECKS", {"mysql": _ok, "milvus": _fail})
    with TestClient(create_app()) as client:
        resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["services"]["milvus"].startswith("error: ConnectionError")
