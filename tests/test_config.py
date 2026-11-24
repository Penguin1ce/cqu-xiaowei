from cqu_affairs.core.config import get_settings


def test_env_prefix_groups(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "db.local")
    monkeypatch.setenv("MYSQL_PASSWORD", "p@ss")
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_DIM", "768")

    s = get_settings()

    assert s.mysql.host == "db.local"
    assert s.mysql.url.startswith("mysql+asyncmy://")
    assert "db.local" in s.mysql.url
    assert s.dashscope.embedding_dim == 768


def test_empty_api_key_treated_as_unset(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    assert get_settings().deepseek.api_key is None
