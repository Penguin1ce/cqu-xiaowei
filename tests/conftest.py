import pytest

from cqu_affairs.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """每个测试前后清空配置缓存，避免 monkeypatch 的环境变量互相影响。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
