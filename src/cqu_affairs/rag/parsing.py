# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""用 MinerU 把 data/raw/ 下的原始文件解析成 Markdown，并保留文档元数据。

每份文件的产物放在 `data/parsed/<分类>/<文件名>/`：

    full.md                  MinerU 生成的 Markdown（表格为 HTML <table>）
    <uuid>_content_list.json 扁平内容块列表，按阅读顺序排列，分块用它：
                             {"type": "text" | "table" | "image" | "list" | "header" | ...,
                              "text": ..., "text_level": 1, "page_idx": 0（从 0 起）, "bbox": [...]}
    <uuid>_content_list_v2.json  按页分组的结构化版本（list[list[dict]]）
    <uuid>_origin.pdf        MinerU 使用的 PDF（doc / docx 会先转换成 PDF）
    layout.json、<uuid>_model.json、images/
    meta.json                本项目写入：清单元数据 + 文件指纹 + 解析信息

meta.json 是后续分块、入库、引用展示的唯一元数据来源。结构：

    {
      "doc_id": "<sha256 前 16 位>",
      "source": {...manifest 中该文件的全部字段...},
      "file":   {"sha256": ..., "size": ..., "ext": ".pdf"},
      "parser": {"name": "mineru", "model_version": ..., "batch_id": ..., "parsed_at": ...},
      "outputs": {"markdown": "full.md", "content_list": "xxx_content_list.json", "files": [...]}
    }

原文件 sha256 和 model_version 都没变时跳过重新解析（节省 MinerU 额度），
但仍会用最新的 manifest 刷新 meta.json 的 source 字段。
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cqu_affairs.core.config import DATA_DIR
from cqu_affairs.rag.manifest import RAW_DIR, DocumentMeta, file_sha256
from cqu_affairs.rag.mineru import MAX_BATCH_FILES, MinerUClient, UploadItem

PARSED_DIR = DATA_DIR / "parsed"
META_FILE = "meta.json"


def parsed_dir(doc: DocumentMeta, root: Path = PARSED_DIR) -> Path:
    return root / doc.category / doc.stem


def doc_id_of(sha256: str) -> str:
    return sha256[:16]


def read_meta(doc: DocumentMeta, root: Path = PARSED_DIR) -> dict | None:
    p = parsed_dir(doc, root) / META_FILE
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def _write_meta(dest: Path, meta: dict) -> None:
    (dest / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", "utf-8")


def _find_outputs(dest: Path) -> dict:
    files = sorted(
        str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and p.name != META_FILE
    )
    md = next((f for f in files if f == "full.md"), None) or next(
        (f for f in files if f.endswith(".md") and "/" not in f), None
    )

    def pick(suffix: str) -> str | None:
        return next((f for f in files if f.endswith(suffix)), None)

    return {
        "markdown": md,
        "content_list": pick("_content_list.json"),
        "content_list_v2": pick("_content_list_v2.json"),
        "origin_pdf": pick("_origin.pdf"),
        "files": files,
    }


@dataclass
class ParseReport:
    parsed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


async def parse_documents(
    docs: list[DocumentMeta],
    client: MinerUClient,
    model_version: str,
    *,
    raw_root: Path = RAW_DIR,
    out_root: Path = PARSED_DIR,
    force: bool = False,
    log=print,
) -> ParseReport:
    report = ParseReport()
    pending: dict[str, tuple[DocumentMeta, str, Path]] = {}

    for doc in docs:
        src = raw_root / doc.path
        if not src.exists():
            report.failed[doc.path] = "原始文件不存在"
            continue
        sha = file_sha256(src)
        old = read_meta(doc, out_root)
        if (
            not force
            and old
            and old["file"]["sha256"] == sha
            and old["parser"]["model_version"] == model_version
            and old["outputs"].get("markdown")
        ):
            dest = parsed_dir(doc, out_root)
            old["source"] = doc.model_dump(mode="json")
            old["outputs"] = _find_outputs(dest)
            _write_meta(dest, old)
            report.skipped.append(doc.path)
            continue
        pending[doc_id_of(sha)] = (doc, sha, src)

    ids = list(pending)
    for i in range(0, len(ids), MAX_BATCH_FILES):
        chunk = ids[i : i + MAX_BATCH_FILES]
        items = [UploadItem(path=pending[d][2], data_id=d) for d in chunk]
        log(f"上传 {len(items)} 个文件到 MinerU …")
        batch_id = await client.submit_batch(items)
        log(f"批次 {batch_id} 已提交，等待解析")

        total = len(items)

        def _progress(results, total=total):
            done = sum(r.state in ("done", "failed") for r in results)
            log(f"  进度 {done}/{total}")

        results = await client.wait_batch(batch_id, expected=len(items), on_progress=_progress)
        by_id = {r.data_id: r for r in results}

        for d in chunk:
            doc, sha, src = pending[d]
            r = by_id.get(d)
            if r is None or r.state != "done" or not r.full_zip_url:
                report.failed[doc.path] = r.err_msg if r else "结果中缺少该文件"
                continue
            dest = parsed_dir(doc, out_root)
            if dest.exists():
                shutil.rmtree(dest)
            await client.download_result(r.full_zip_url, dest)
            outputs = _find_outputs(dest)
            if outputs["markdown"] is None:
                report.failed[doc.path] = "解析结果中没有 Markdown 文件"
                continue
            _write_meta(
                dest,
                {
                    "doc_id": d,
                    "source": doc.model_dump(mode="json"),
                    "file": {"sha256": sha, "size": src.stat().st_size, "ext": src.suffix.lower()},
                    "parser": {
                        "name": "mineru",
                        "model_version": model_version,
                        "batch_id": batch_id,
                        "parsed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    },
                    "outputs": outputs,
                },
            )
            report.parsed.append(doc.path)
    return report
