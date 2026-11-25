from cqu_affairs.rag import chunking
from cqu_affairs.rag.chunking import chunk_document


def _meta(structure: str, *, ext: str = ".pdf", status: str = "effective", **src) -> dict:
    source = {
        "path": "leave/测试办法.pdf",
        "title": "重庆大学测试管理办法",
        "issuer": "重庆大学",
        "level": "school",
        "doc_no": "重大校发〔2021〕1号",
        "publish_date": "2021-05-17",
        "status": status,
        "structure": structure,
        "affairs": ["leave"],
    } | src
    return {"doc_id": "abcd1234abcd1234", "source": source, "file": {"ext": ext}}


def t(text: str, page: int = 0, level: int | None = None, type_: str = "text") -> dict:
    d = {"type": type_, "text": text, "page_idx": page, "bbox": [0, 0, 1, 1]}
    if level is not None:
        d["text_level"] = level
    return d


def table(body: str, page: int = 0) -> dict:
    return {
        "type": "table",
        "table_body": body,
        "table_caption": [],
        "table_footnote": [],
        "page_idx": page,
    }


# ── 章 / 条 ──────────────────────────────────────────────────────

CHAPTER_DOC = [
    t("重庆大学文件", 0, 1),
    t("重大校发〔2021〕1号", 0),
    t("关于印发《重庆大学测试管理办法》的通知", 0, 1),
    t("学校各单位：现印发给你们，请遵照执行。", 0),
    t("1", 0, type_="page_number"),
    t("重庆大学测试管理办法", 1, 1),
    t("第一章 总 则", 1),
    t("第一条 为规范请假管理，制定本办法。", 1),
    t("第二条 学生请假应当事先办理手续，", 1),
    t("2", 1, type_="page_number"),
    t("页眉", 2, type_="header"),
    t("并经辅导员批准。", 2),
    t("第二章 请假审批", 2, 2),
    t("第三条 请假三日以下由辅导员审批。", 2),
]


def test_chapter_article_one_chunk_per_article():
    chunks = chunk_document(_meta("chapter_article"), CHAPTER_DOC)
    kinds = [c.kind for c in chunks]
    assert kinds == ["notice", "article", "article", "article"]
    assert [c.article_no for c in chunks[1:]] == ["第一条", "第二条", "第三条"]
    assert chunks[1].heading_path == ["第一章 总 则"]
    assert chunks[3].heading_path == ["第二章 请假审批"]


def test_article_spanning_pages_is_merged_and_noise_dropped():
    art2 = chunk_document(_meta("chapter_article"), CHAPTER_DOC)[2]
    assert art2.text == "第二条 学生请假应当事先办理手续，\n并经辅导员批准。"
    assert art2.pages == [2, 3]  # 1 起
    assert "页眉" not in art2.text and "\n2\n" not in art2.text


def test_citation_and_embedding_text():
    c = chunk_document(_meta("chapter_article"), CHAPTER_DOC)[2]
    assert (
        c.citation == "《重庆大学测试管理办法》第二条（重大校发〔2021〕1号，2021-05-17，第2–3页）"
    )
    assert c.location == "第一章 总 则 第二条"
    assert c.embedding_text().startswith("重庆大学测试管理办法 > 第一章 总 则\n第二条")
    assert c.chunk_id == "abcd1234abcd1234-0003"


def test_long_article_split_into_parts(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHARS", 60)
    items = "".join(f"（{n}）情形{n}的说明文字较长需要单独成段。" for n in "一二三四五")
    content = [
        t("第一章 总则"),
        t("第一条 有下列情形之一的，给予警告处分："),
        *[t(f"（{n}）情形{n}的说明文字较长需要单独成段。") for n in "一二三四五"],
    ]
    chunks = [c for c in chunk_document(_meta("chapter_article"), content) if c.kind == "article"]
    assert len(chunks) > 1
    assert {c.article_no for c in chunks} == {"第一条"}
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert "".join(c.text.replace("\n", "") for c in chunks).endswith(items[-20:])


# ── 一、（一）编号 ─────────────────────────────────────────────

NUMBERED_DOC = [
    t("**大数据与软件学院测试细则**"),
    t("**（讨论稿）**"),
    t("根据有关文件，制定本细则。"),
    t("**一、评定范围**"),
    t("本科二年级以上学生。"),
    t("**二、评定条件**"),
    t("（一）基本条件"),
    t("1.遵守校规校纪；"),
    t("（二）具体条件"),
    t("2.成绩排名前10%。"),
]


def test_numbered_sections_and_bold_stripped():
    meta = _meta("numbered", ext=".doc", status="draft", title="大数据与软件学院测试细则（讨论稿）")
    chunks = chunk_document(meta, NUMBERED_DOC)
    assert chunks[0].kind == "notice"
    sections = [c for c in chunks if c.kind == "section"]
    assert [c.heading_path for c in sections] == [["一、评定范围"], ["二、评定条件"]]
    assert sections[1].text.startswith("二、评定条件\n（一）基本条件")
    assert "**" not in "".join(c.text for c in chunks)
    # doc 文件不记录页码；讨论稿在引用中标注
    assert sections[0].pages == []
    assert (
        sections[0].citation.endswith("一、评定范围（2021-05-17，讨论稿）")
        or "讨论稿" in sections[0].citation
    )


def test_letterhead_and_repeated_title_are_not_headings():
    content = [
        t("重庆大学大数据与软件学院文件", 0, 1),
        t("关于印发《测试细则（2026年修订）》的通知", 0, 1),
        t("各系、室：现印发给你们。", 0),
        t("测试细则 (2026年修订)", 1, 1),
        t("一、评定范围", 1, 2),
        t("我院在读学生。", 1),
    ]
    chunks = chunk_document(_meta("numbered", title="重庆大学测试细则（2026年修订）"), content)
    assert [c.kind for c in chunks] == ["notice", "section"]
    assert chunks[1].heading_path == ["一、评定范围"]
    assert all("文件" not in h for c in chunks for h in c.heading_path)


# ── 表格 ─────────────────────────────────────────────────────────


def _rows(n: int) -> str:
    return "".join(f"<tr><td>{i}</td><td>课程{i}</td></tr>" for i in range(n))


def test_table_chunk_with_caption_heading_and_page_range(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_TABLE_CHARS", 200)
    head = '<tr><td rowspan="2">代码</td><td>名称</td></tr><tr><td>子表头</td></tr>'
    content = [
        t("课程设置一览表", 0, 2),
        t("2.竞赛成果类", 0, 2),
        table(f"<table>{head}{_rows(12)}</table>", 0),
        table("", 1),  # 续页占位
        table("", 2),
        t("备注", 3, 2),
        t("至少修读8学分。", 3),
    ]
    chunks = chunk_document(_meta("form"), content)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) > 1
    assert all(c.heading_path == ["课程设置一览表", "2.竞赛成果类"] for c in tables)
    assert all(c.pages == [1, 2, 3] for c in tables)
    # 两行表头在每个拆分块中重复
    assert all(c.text.startswith(f"<table>{head}") for c in tables)
    # 只有标题的块被丢弃，备注节保留
    assert [c.text for c in chunks if c.kind != "table"] == ["备注\n至少修读8学分。"]


def test_images_without_caption_dropped_and_list_kept():
    content = [
        t("一、登录系统", 0, 2),
        {
            "type": "image",
            "img_path": "a.jpg",
            "image_caption": [],
            "image_footnote": [],
            "page_idx": 0,
        },
        {"type": "list", "list_items": ["• 打开网站", "• 点击报销"], "page_idx": 0},
    ]
    chunks = chunk_document(_meta("steps"), content)
    assert len(chunks) == 1
    assert chunks[0].text == "一、登录系统\n• 打开网站\n• 点击报销"


def test_document_without_headings_is_a_section():
    content = [t("登录智能财务服务大厅"), t("点击网上投递报销")]
    chunks = chunk_document(_meta("steps", ext=".docx"), content)
    assert [c.kind for c in chunks] == ["section"]
    assert chunks[0].citation == "《重庆大学测试管理办法》（重大校发〔2021〕1号，2021-05-17）"
