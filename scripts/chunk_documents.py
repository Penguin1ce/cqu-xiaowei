# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""对 data/parsed/ 下的解析结果做结构化分块，写入 data/chunks/<分类>/<文件名>.jsonl。

用法：
    uv run python scripts/chunk_documents.py                # 全部
    uv run python scripts/chunk_documents.py --only leave   # 只处理某个分类
    uv run python scripts/chunk_documents.py --show 学籍     # 打印标题含“学籍”的文件的全部块

不调用任何外部 API。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from cqu_affairs.core.config import DATA_DIR
from cqu_affairs.rag.chunking import chunk_parsed_dir
from cqu_affairs.rag.parsing import PARSED_DIR

CHUNKS_DIR = DATA_DIR / "chunks"


def main(args: argparse.Namespace) -> int:
    doc_dirs = sorted(p.parent for p in PARSED_DIR.glob("*/*/meta.json"))
    if args.only:
        doc_dirs = [d for d in doc_dirs if d.parent.name == args.only]
    if not doc_dirs:
        print("没有找到解析结果，先运行 scripts/parse_documents.py")
        return 1

    total = 0
    for d in doc_dirs:
        chunks = chunk_parsed_dir(d)
        out = CHUNKS_DIR / d.parent.name / f"{d.name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        total += len(chunks)

        kinds = Counter(c.kind for c in chunks)
        lens = [c.char_count for c in chunks]
        articles = len({c.article_no for c in chunks if c.article_no})
        print(
            f"{chunks[0].title[:30]:30s} 块 {len(chunks):3d} "
            f"{dict(kinds)} 条 {articles:2d} 长度 {min(lens)}–{max(lens)}"
        )
        if args.show and args.show in chunks[0].title:
            for c in chunks:
                print(f"\n── {c.chunk_id} [{c.kind}] {c.citation}")
                print(f"   path={c.heading_path} part={c.part} {c.char_count}字")
                print("   " + c.text[:300].replace("\n", "\n   "))
    print(f"\n共 {len(doc_dirs)} 份文件，{total} 个块，输出到 {Path(CHUNKS_DIR)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="只处理指定分类目录")
    parser.add_argument("--show", help="打印标题包含该关键词的文件的全部块")
    sys.exit(main(parser.parse_args()))
