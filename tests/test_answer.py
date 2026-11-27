from datetime import date

from cqu_affairs.rag.answer import (
    NOT_FOUND_ANSWER,
    build_messages,
    clean_citations,
    drop_superseded,
    finalize,
    format_source,
    to_citation,
)
from cqu_affairs.rag.retriever import SearchHit


def hit(n: int, **kw) -> SearchHit:
    base = dict(
        chunk_id=f"d{n}-0001",
        doc_id=f"d{n}",
        title=f"文件{n}",
        citation=f"《文件{n}》第{n}条（X〔2021〕{n}号，2021-05-17，第1页）",
        location=f"第一章 第{n}条",
        kind="article",
        text=f"第{n}条 正文{n}。",
        status="effective",
        level="school",
        issuer="重庆大学",
        doc_no=f"X〔2021〕{n}号",
        publish_date=date(2021, 5, 17),
        supersedes=[],
        pages=[1],
        score=0.9,
        recall_rank=n,
    )
    return SearchHit(**(base | kw))


def test_clean_citations_drops_out_of_range_and_dedupes():
    text, used = clean_citations("需提交证明[2]，并审批[1][2]。另见[7][0]。", n_sources=2)
    assert text == "需提交证明[2]，并审批[1][2]。另见。"
    assert used == [2, 1]


def test_finalize_maps_used_numbers_to_citations():
    sources = [to_citation(i, hit(i)) for i in (1, 2, 3)]
    r = finalize("q", "要点甲[3]；要点乙[1]。", sources, reranked=True)
    assert r.found and [c.index for c in r.citations] == [3, 1]
    assert r.citations[0].citation.startswith("《文件3》")
    assert len(r.sources) == 3 and r.notices == []


def test_model_saying_not_found_becomes_standard_answer():
    sources = [to_citation(1, hit(1))]
    r = finalize("食堂几点开门", "未找到依据。", sources, reranked=True)
    assert not r.found and r.answer == NOT_FOUND_ANSWER and r.citations == []


def test_draft_and_degraded_notices():
    sources = [to_citation(1, hit(1, status="draft", title="国奖细则（讨论稿）"))]
    r = finalize("q", "每人每年10000元[1]（讨论稿）。", sources, reranked=False)
    assert any("讨论稿" in n for n in r.notices)
    assert any("仅供参考" in n for n in r.notices)


def test_answer_without_citations_gets_notice():
    r = finalize("q", "需要提交证明。", [to_citation(1, hit(1))], reranked=True)
    assert r.found and r.citations == []
    assert any("没有标注出处" in n for n in r.notices)


def test_drop_superseded_by_doc_no_ignoring_spaces():
    old = hit(1, doc_no="重大软件〔2024〕16 号")
    new = hit(2, doc_no="重大软件〔2026〕17号", supersedes=["重大软件〔2024〕16号"])
    kept, dropped = drop_superseded([old, new])
    assert [h.doc_id for h in kept] == ["d2"] and [h.doc_id for h in dropped] == ["d1"]


def test_source_block_shows_level_date_status_and_number():
    s = format_source(2, hit(1, level="college", status="draft", publish_date=None, doc_no=""))
    assert s.startswith("[2] 《文件1》第一章 第1条\n（院级，发布日期未标注，讨论稿）\n")


def test_messages_contain_today_sources_and_question():
    msgs = build_messages("请病假要什么材料", [hit(1), hit(2)], date(2026, 11, 27))
    assert msgs[0].role == "system" and "今天是 2026-11-27" in msgs[0].content[0].text
    user = msgs[1].content[0].text
    assert "[1] 《文件1》" in user and "[2] 《文件2》" in user
    assert user.endswith("【问题】\n请病假要什么材料")
