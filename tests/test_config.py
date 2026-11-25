from cqu_affairs.core.config import DeepSeekSettings, get_settings


def test_env_prefix_groups(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "db.local")
    monkeypatch.setenv("MYSQL_PASSWORD", "p@ss")
    monkeypatch.setenv("SILICONFLOW_EMBEDDING_DIM", "768")
    monkeypatch.setenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")

    s = get_settings()

    assert s.mysql.host == "db.local"
    assert s.mysql.url.startswith("mysql+asyncmy://")
    assert "db.local" in s.mysql.url
    assert s.siliconflow.embedding_dim == 768
    assert s.siliconflow.embedding_model == "Qwen/Qwen3-Embedding-4B"


def test_empty_api_key_treated_as_unset(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    # 不读项目 .env，避免本地已填写的真实密钥干扰断言
    assert DeepSeekSettings(_env_file=None).api_key is None
