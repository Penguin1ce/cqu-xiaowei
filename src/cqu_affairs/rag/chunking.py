# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""结构化分块：把 MinerU 的 content_list 按文件原有的章、条、节切成检索单元。

输入是 `data/parsed/<分类>/<文件名>/` 下的 `meta.json` 和 `*_content_list.json`，
输出 `Chunk` 列表。每个块都带完整的来源元数据，`Chunk.citation` 直接生成引用文字。

切分规则按 manifest 的 `structure` 区分：

- chapter_article：一“条”一块，记录所属“章”；条前的印发通知单独成块（kind=notice）
- 其余结构：按标题切节。“一、”或带 text_level 的短标题为节标题，“（一）”“1.”为小节；
  节超长时按小节或段落再切，同一节的子块共用 heading_path

公共规则：

- 丢弃 header / footer / page_number / aside_text 和没有说明文字的图片
- 表格单独成块（kind=table），超长按行拆分并重复表头；跨页表格 MinerU 已合并到首页，
  后续页留下的空表直接丢弃
- 标题识别以文字模式为主，text_level 只作辅助（MinerU 对同级标题的 text_level 标注不稳定）
- doc / docx 解析结果的 page_idx 全为 0，这类文件不记录页码
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 正文块超过该长度时继续切分；表格块的上限单独设置
MAX_CHARS = 1200
MAX_TABLE_CHARS = 3000

_CN_NUM = "零〇一二三四五六七八九十百"
_RE_CHAPTER = re.compile(rf"^第\s*[{_CN_NUM}]+\s*章")
_RE_ARTICLE = re.compile(rf"^第\s*([{_CN_NUM}]+)\s*条")
_RE_SEC1 = re.compile(rf"^[{_CN_NUM}]+\s*[、.．]")
_RE_SEC2 = re.compile(rf"^[（(]\s*[{_CN_NUM}\d]+\s*[)）]|^\d+(\.\d+)*\s*[、.．\s]")
_RE_LETTERHEAD = re.compile(r"^.{2,20}文件$")
_DROP_TYPES = {"header", "footer", "page_number", "aside_text"}
_NO_PAGE_EXTS = {".doc", ".docx"}
_HEADING_MAX = 40

ChunkKind = Literal["notice", "article", "section", "table"]


# ── 数据结构 ─────────────────────────────────────────────────────


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    seq: int
    kind: ChunkKind
    text: str
    heading_path: list[str] = Field(default_factory=list)
    article_no: str | None = None
    part: int | None = None
    """超长条款 / 节 / 表格被拆分时的序号（从 1 起），未拆分为 None。"""
    pages: list[int] = Field(default_factory=list)
    """1 起的页码；doc / docx 文件为空。"""

    # 文档级元数据（来自 meta.json 的 source）
    title: str
    issuer: str = ""
    doc_no: str = ""
    publish_date: date | None = None
    status: Literal["effective", "draft"]
    level: Literal["school", "college"]
    category: str
    affairs: list[str] = Field(default_factory=list)
    source_path: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def location(self) -> str:
        """文内位置，如“第三章 考核与成绩记载 第三十条”。"""
        parts = list(self.heading_path)
        if self.article_no and self.article_no not in parts:
            parts.append(self.article_no)
        return " ".join(parts)

    @property
    def citation(self) -> str:
        """回答中展示的引用，如：

        《重庆大学…学籍管理办法（2021年修订）》第三十条（重大校发〔2021〕57号，2021-05-17，第11–12页）
        """
        loc = self.article_no or (self.heading_path[-1] if self.heading_path else "")
        extra = [x for x in (self.doc_no, self.publish_date and self.publish_date.isoformat()) if x]
        if self.pages:
            a, b = self.pages[0], self.pages[-1]
            extra.append(f"第{a}页" if a == b else f"第{a}–{b}页")
        if self.status == "draft":
            extra.append("讨论稿")
        return f"《{self.title}》{loc}" + (f"（{'，'.join(extra)}）" if extra else "")

    def embedding_text(self) -> str:
        """向量化用的文本：拼上文件标题和标题路径，让没有出现关键词的条款也能被检索到。"""
        head = " > ".join([self.title, *self.heading_path])
        return f"{head}\n{self.text}"


@dataclass
class Block:
    """归一化后的内容块。"""

    kind: Literal["text", "table", "list", "equation", "image"]
    text: str
    page: int  # 0 起
    text_level: int | None = None
    page_end: int | None = None
    """跨页表格的末页（0 起），MinerU 把跨页表格合并到首页，续页只留空表占位。"""


# ── 预处理 ───────────────────────────────────────────────────────


def _norm_title(text: str) -> str:
    """比较标题用：去空白、统一括号、去掉“重庆大学”前缀。"""
    s = re.sub(r"\s+", "", text).replace("(", "（").replace(")", "）")
    return s.removeprefix("重庆大学")


def _is_doc_title(text: str, doc_title: str) -> bool:
    a, b = _norm_title(text), _norm_title(doc_title)
    return len(a) >= 8 and (a in b or b in a)


def _clean(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"[ \t　]+", " ", text).strip()


def normalize(content_list: list[dict]) -> list[Block]:
    out: list[Block] = []
    for x in content_list:
        t, page = x.get("type"), x.get("page_idx", 0)
        if t in _DROP_TYPES:
            continue
        if t == "table":
            body = (x.get("table_body") or "").strip()
            if "<tr" not in body:
                # 跨页表格的续页占位：用它的页码延长前一个表格的页码范围
                if out and out[-1].kind == "table":
                    out[-1].page_end = page
                continue
            cap = _clean(" ".join(x.get("table_caption") or []))
            foot = _clean(" ".join(x.get("table_footnote") or []))
            text = "\n".join(s for s in (cap, body, foot) if s)
            out.append(Block("table", text, page))
        elif t == "list":
            items = [_clean(i) for i in x.get("list_items") or []]
            text = "\n".join(i for i in items if i)
            if text:
                out.append(Block("list", text, page))
        elif t == "equation":
            if text := (x.get("text") or "").strip():
                out.append(Block("equation", text, page))
        elif t == "image":
            cap = _clean(" ".join((x.get("image_caption") or []) + (x.get("image_footnote") or [])))
            if cap:
                out.append(Block("image", cap, page))
        else:
            if text := _clean(x.get("text") or ""):
                out.append(Block("text", text, page, x.get("text_level")))
    return out


def heading_level(b: Block) -> int | None:
    """标题层级：2 = 节（“一、”或短标题），3 = 小节（“（一）”“1.”），None = 正文。"""
    if b.kind != "text" or len(b.text) > _HEADING_MAX:
        return None
    if _RE_ARTICLE.match(b.text):
        return None
    if _RE_SEC1.match(b.text):
        return 2
    if b.text_level is not None:
        if _RE_SEC2.match(b.text):
            return 3
        return min(b.text_level, 2)
    return None


# ── 切分 ─────────────────────────────────────────────────────────


@dataclass
class _Piece:
    kind: ChunkKind
    blocks: list[Block]
    heading_path: list[str]
    article_no: str | None = None


def _split_text_blocks(blocks: list[Block], limit: int) -> list[list[Block]]:
    """把一组正文块按长度切成若干组，优先在小节标题处断开。"""
    groups: list[list[Block]] = [[]]
    size = 0
    for b in blocks:
        n = len(b.text)
        boundary = heading_level(b) == 3 or bool(_RE_SEC2.match(b.text))
        if groups[-1] and size + n > limit and (boundary or size >= limit):
            groups.append([])
            size = 0
        groups[-1].append(b)
        size += n
    return [g for g in groups if g]


def _split_table(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    rows = re.findall(r"<tr.*?</tr>", text, flags=re.S)
    if len(rows) < 2:
        return [text]
    # 表头行数 = 首行单元格的最大 rowspan（多行表头要整体重复）
    spans = [int(s) for s in re.findall(r'rowspan="(\d+)"', rows[0])]
    n_head = min(max(spans, default=1), len(rows) - 1)
    head, rest = "".join(rows[:n_head]), rows[n_head:]
    out, cur = [], []
    for r in rest:
        if cur and len(head) + sum(map(len, cur)) + len(r) > limit:
            out.append(cur)
            cur = []
        cur.append(r)
    out.append(cur)
    return [f"<table>{head}{''.join(g)}</table>" for g in out]


def _pieces_chapter_article(blocks: list[Block]) -> list[_Piece]:
    pieces: list[_Piece] = []
    chapter: str | None = None
    cur: _Piece | None = _Piece("notice", [], [])
    for b in blocks:
        if b.kind == "text" and _RE_CHAPTER.match(b.text) and len(b.text) <= _HEADING_MAX:
            chapter = re.sub(r"\s+", " ", b.text)
            # 章标题不单独成块，挂到后续条款的 heading_path 上
            if cur and cur.kind == "notice":
                pieces.append(cur)
                cur = None
            continue
        m = _RE_ARTICLE.match(b.text) if b.kind == "text" else None
        if m:
            if cur:
                pieces.append(cur)
            no = f"第{m.group(1)}条"
            cur = _Piece("article", [b], [chapter] if chapter else [], no)
            continue
        if cur is None:  # 章标题之后、第一条之前的零散文字
            cur = _Piece("notice", [], [chapter] if chapter else [])
        cur.blocks.append(b)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p.blocks]


def _pieces_sections(blocks: list[Block], doc_title: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    major: list[str] = []
    # 第一个标题之前的内容（红头、印发通知、文件名）视为文件说明
    cur = _Piece("notice", [], [])

    def flush() -> None:
        nonlocal cur
        if cur.blocks:
            pieces.append(cur)
        cur = _Piece("section", [], list(major))

    for b in blocks:
        lv = heading_level(b)
        if b.kind == "text" and lv is not None and _is_doc_title(b.text, doc_title):
            # 正文里重复的文件标题（含“关于印发《…》的通知”）：仍在文件说明阶段就并入说明，
            # 否则视为正文（或附件）开始，清空标题栈
            if cur.kind == "notice":
                cur.blocks.append(b)
                continue
            flush()
            stack, major = [], []
            cur.heading_path = []
            continue
        if lv is not None and _RE_LETTERHEAD.match(b.text):
            cur.blocks.append(b)  # 红头“××文件”不是标题
            continue
        if lv == 2 or (lv is not None and lv < 2):
            flush()
            stack = [(level, t) for level, t in stack if level < lv]
            stack.append((lv, b.text))
            major = [t for _, t in stack]
            cur.heading_path = list(major)
            cur.blocks.append(b)
            continue
        cur.blocks.append(b)
    flush()
    # 全文没有节标题（如简短的操作说明）：内容本身就是正文
    if all(p.kind == "notice" for p in pieces):
        for p in pieces:
            p.kind = "section"
    return pieces


def _finalize(
    pieces: list[_Piece], use_pages: bool
) -> list[tuple[_Piece, str, int | None, list[int]]]:
    """把 piece 展开成最终块：表格独立、超长切分。返回 (piece, text, part, pages)。"""
    out = []

    def pages_of(bs: list[Block]) -> list[int]:
        if not use_pages:
            return []
        pages = set()
        for b in bs:
            pages.update(range(b.page + 1, (b.page_end or b.page) + 2))
        return sorted(pages)

    for p in pieces:
        # 1. 按表格把 piece 拆成交替的 text / table 段；紧挨表格前的小标题（如“2.竞赛成果类”）
        #    从正文段移出，作为该表格的 heading_path 末项
        segments: list[tuple[_Piece, list[Block]]] = []
        text_buf: list[Block] = []
        for b in p.blocks:
            if b.kind != "table":
                text_buf.append(b)
                continue
            path = list(p.heading_path)
            if text_buf and heading_level(text_buf[-1]) is not None:
                caption = text_buf.pop().text
                if caption not in path:
                    path.append(caption)
            if text_buf:
                segments.append((p, text_buf))
                text_buf = []
            segments.append((_Piece("table", [b], path, p.article_no), [b]))
        if text_buf:
            segments.append((p, text_buf))

        # 2. 展开成块
        # (piece, text, pages, 是否只有一个标题块)
        produced: list[tuple[_Piece, str, list[int], bool]] = []
        for kp, bs in segments:
            if kp.kind == "table":
                for t in _split_table(bs[0].text, MAX_TABLE_CHARS):
                    produced.append((kp, t, pages_of(bs), False))
            else:
                for g in _split_text_blocks(bs, MAX_CHARS):
                    bare = len(g) == 1 and heading_level(g[0]) is not None
                    produced.append((kp, "\n".join(b.text for b in g), pages_of(g), bare))

        # 3. 只有一个标题的正文块：并入紧随其后的同节正文块，否则丢弃（无检索价值）
        merged: list[tuple[_Piece, str, list[int]]] = []
        for i, (kp, text, pgs, bare) in enumerate(produced):
            if bare:
                nxt = produced[i + 1] if i + 1 < len(produced) else None
                if nxt and nxt[0] is kp:
                    produced[i + 1] = (kp, text + "\n" + nxt[1], sorted(set(pgs + nxt[2])), False)
                continue
            merged.append((kp, text, pgs))

        # 4. 同一来源被拆成多块时编号
        counts: dict[int, int] = {}
        for kp, _, _ in merged:
            counts[id(kp)] = counts.get(id(kp), 0) + 1
        seen: dict[int, int] = {}
        for kp, text, pgs in merged:
            seen[id(kp)] = seen.get(id(kp), 0) + 1
            out.append((kp, text, seen[id(kp)] if counts[id(kp)] > 1 else None, pgs))
    return out


# ── 入口 ─────────────────────────────────────────────────────────


def chunk_document(meta: dict, content_list: list[dict]) -> list[Chunk]:
    src = meta["source"]
    blocks = normalize(content_list)
    if src["structure"] == "chapter_article":
        pieces = _pieces_chapter_article(blocks)
    else:
        pieces = _pieces_sections(blocks, src["title"])
    use_pages = meta["file"]["ext"] not in _NO_PAGE_EXTS

    chunks: list[Chunk] = []
    for seq, (p, text, part, pages) in enumerate(_finalize(pieces, use_pages), start=1):
        chunks.append(
            Chunk(
                chunk_id=f"{meta['doc_id']}-{seq:04d}",
                doc_id=meta["doc_id"],
                seq=seq,
                kind=p.kind,
                text=text,
                heading_path=[h for h in p.heading_path if h],
                article_no=p.article_no,
                part=part,
                pages=pages,
                title=src["title"],
                issuer=src.get("issuer", ""),
                doc_no=src.get("doc_no", ""),
                publish_date=src.get("publish_date"),
                status=src["status"],
                level=src["level"],
                category=Path(src["path"]).parts[0],
                affairs=src.get("affairs", []),
                source_path=src["path"],
            )
        )
    return chunks


def chunk_parsed_dir(doc_dir: Path) -> list[Chunk]:
    meta = json.loads((doc_dir / "meta.json").read_text("utf-8"))
    cl_name = meta["outputs"].get("content_list")
    if not cl_name:
        raise FileNotFoundError(f"{doc_dir} 缺少 content_list.json")
    content_list = json.loads((doc_dir / cl_name).read_text("utf-8"))
    return chunk_document(meta, content_list)
