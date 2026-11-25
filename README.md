# 校园事务智能办理系统（cqu-affairs）

基于大语言模型智能体的校园事务办理助手，覆盖请假、奖学金申请、竞赛报销、证明办理等场景。

## 技术栈

AgentScope · DeepSeek（对话）· 硅基流动 Embedding / Rerank · FastAPI · MySQL · Redis · Milvus · Vue 3

## 快速开始

环境要求：Python 3.13、[uv](https://docs.astral.sh/uv/)、Docker。

```bash
# 1. 安装依赖
uv sync

# 2. 配置：复制模板并填写 DEEPSEEK_API_KEY、SILICONFLOW_API_KEY
cp .env.example .env

# 3. 启动 MySQL / Redis / Milvus
docker compose -f docker/docker-compose.yml up -d

# 4. 连通性自检（基础服务 + 三个模型 API）
uv run python scripts/check_services.py

# 5. 数据库迁移
uv run alembic upgrade head

# 6. 启动后端（http://127.0.0.1:8000/docs）
uv run cqu-affairs
```

健康检查：`GET /health`（进程存活）、`GET /health/ready`（逐个检查 MySQL、Redis、Milvus）。

## 开发

```bash
uv run pytest                    # 运行全部测试
uv run pytest tests/test_rerank.py::test_rerank_requires_api_key   # 运行单个测试
uv run ruff check . && uv run ruff format .                         # lint 与格式化
uv run alembic revision --autogenerate -m "add xxx table"   # 生成迁移
```

## 作者与许可证

Copyright (c) 2026 Penguin1ce

本项目以 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0-only）发布。使用、修改或通过网络提供本项目（含衍生作品）时，须保留原作者署名与版权声明，并以相同许可证公开完整源代码。
