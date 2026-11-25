# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""用 MinerU 解析 data/raw/manifest.toml 中登记的文件，产物写入 data/parsed/。

用法：
    uv run python scripts/parse_documents.py                 # 解析新增或有变动的文件
    uv run python scripts/parse_documents.py --only leave    # 只处理某个分类
    uv run python scripts/parse_documents.py --force         # 忽略缓存全部重解析
    uv run python scripts/parse_documents.py --dry-run       # 只列出将要处理的文件

会消耗 MinerU 每日解析页数额度。
"""

import argparse
import asyncio
import sys

from cqu_affairs.core.config import get_settings
from cqu_affairs.rag.manifest import load_manifest
from cqu_affairs.rag.mineru import MinerUClient
from cqu_affairs.rag.parsing import parse_documents


async def main(args: argparse.Namespace) -> int:
    docs = load_manifest()
    if args.only:
        docs = [d for d in docs if d.category == args.only]
    if args.dry_run:
        for d in docs:
            print(f"  {d.path}")
        print(f"共 {len(docs)} 个文件")
        return 0

    s = get_settings().mineru
    async with MinerUClient(s) as client:
        report = await parse_documents(docs, client, s.model_version, force=args.force)

    n_ok, n_skip, n_fail = len(report.parsed), len(report.skipped), len(report.failed)
    print(f"\n解析 {n_ok} 个，跳过（未变动）{n_skip} 个，失败 {n_fail} 个")
    for path, err in report.failed.items():
        print(f"  ✗ {path}：{err}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="只处理指定分类目录，如 scholarship")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    sys.exit(asyncio.run(main(parser.parse_args())))
