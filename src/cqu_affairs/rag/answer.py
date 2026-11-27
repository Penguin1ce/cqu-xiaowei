# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""带来源引用的问答：提示词、上下文组织、引用回查与时效处理（纯函数，不调用外部服务）。

引用的生成方式：检索结果按 [1]…[n] 编号放进上下文，要求模型在每个要点后写 [n]；
回答完成后由代码解析 [n] 回查对应条款，生成引用列表。模型只写编号，不写出处文字，
因此不可能编造文件名或条款号；资料中不存在的编号会从回答里删除。

时效处理（P2.2）：

1. 代码过滤：命中结果中，若某文件的字号出现在另一份命中文件的 `supersedes` 里，
   视为已废止，不放进上下文
2. 上下文标注每份资料的级别（校级 / 院级）、发布日期、生效状态，并告知模型今天的日期
3. 提示词规定冲突时以较新、较具体的为准；讨论稿、带届次的通知要说明适用范围
4. 代码追加提示：引用了讨论稿、检索降级（无 Rerank）等情况
"""

import re
from datetime import date

from agentscope.message import Msg, TextBlock
from pydantic import BaseModel

from cqu_affairs.rag.retriever import SearchHit

NOT_FOUND_ANSWER = (
    "未找到依据：知识库中没有与这个问题相关的政策条款。建议咨询辅导员或学院教务办公室。"
)

# 提示词里的句子不折行，便于阅读和修改
SYSTEM_PROMPT = """你是重庆大学校园事务助手，依据给定的【参考资料】回答学生关于学校政策和办事流程的问题。

规则：
1. 只依据【参考资料】回答。不使用资料以外的知识，不推测、不补充资料没有写明的内容（如具体天数、金额、部门、地点、电话）。
2. 每个要点后用方括号编号标注出处，如“需提交校医院诊断证明[1]”；一个要点有多处依据时写 [1][3]。只能使用资料中存在的编号。
3. 资料只能回答问题的一部分时，回答能回答的部分，并明确写出“关于……，现有资料未找到依据”。
4. 资料与问题都不相关时，只回答“未找到依据。”这几个字。
5. 时效与冲突：
   - 每份资料都标注了级别、发布日期和状态。多份资料规定不一致时，以发布日期较新的为准，并指出差异。
   - 学院细则是对学校规定的细化：两者不冲突时一并说明，学院细则写得更具体的部分优先。
   - 状态为“讨论稿”的资料尚未正式定稿，引用时在句末注明“（讨论稿）”。
   - 带具体年份或届次的通知（如“2027届”）只适用于对应对象，引用时说明适用范围。
6. 用简体中文直接回答，条理清晰；办理步骤用编号列表。不要写“根据参考资料”之类的开场白，结尾不要罗列参考文献（系统会自动附上）。

今天是 {today}。"""  # noqa: E501

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_LEVEL = {"school": "校级", "college": "院级"}
_STATUS = {"effective": "现行", "draft": "讨论稿"}


class Citation(BaseModel):
    index: int
    """回答中使用的编号 [index]。"""
    chunk_id: str
    citation: str
    title: str
    location: str
    status: str
    publish_date: date | None
    score: float | None
    text: str


class QAResult(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[Citation]
    """回答中实际引用到的条款，按首次出现的顺序。"""
    sources: list[Citation]
    """提供给模型的全部参考资料（编号与回答中的 [n] 一致）。"""
    notices: list[str]
    reranked: bool
    cached: bool = False


def _norm_doc_no(s: str) -> str:
    return re.sub(r"\s+", "", s)


def drop_superseded(hits: list[SearchHit]) -> tuple[list[SearchHit], list[SearchHit]]:
    """去掉被同批命中的其他文件明确废止的文件，返回 (保留, 去掉)。"""
    revoked = {_norm_doc_no(x) for h in hits for x in h.supersedes}
    kept, dropped = [], []
    for h in hits:
        (dropped if h.doc_no and _norm_doc_no(h.doc_no) in revoked else kept).append(h)
    return kept, dropped


def to_citation(index: int, h: SearchHit) -> Citation:
    return Citation(
        index=index,
        chunk_id=h.chunk_id,
        citation=h.citation,
        title=h.title,
        location=h.location,
        status=h.status,
        publish_date=h.publish_date,
        score=h.score,
        text=h.text,
    )


def format_source(index: int, h: SearchHit) -> str:
    meta = [
        _LEVEL.get(h.level, h.level),
        f"发布日期 {h.publish_date.isoformat()}" if h.publish_date else "发布日期未标注",
        _STATUS.get(h.status, h.status),
    ]
    if h.doc_no:
        meta.append(h.doc_no)
    head = f"[{index}] 《{h.title}》{h.location}".rstrip()
    return f"{head}\n（{'，'.join(meta)}）\n{h.text}"


def build_messages(question: str, hits: list[SearchHit], today: date) -> list[Msg]:
    sources = "\n\n".join(format_source(i, h) for i, h in enumerate(hits, 1))
    user = f"【参考资料】\n{sources}\n\n【问题】\n{question}"
    return [
        Msg(
            name="system",
            role="system",
            content=[TextBlock(text=SYSTEM_PROMPT.format(today=today.isoformat()))],
        ),
        Msg(name="user", role="user", content=[TextBlock(text=user)]),
    ]


def clean_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """删除越界编号，返回 (清理后的回答, 按首次出现顺序去重的有效编号)。"""
    used: list[int] = []

    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        if 1 <= k <= n_sources:
            if k not in used:
                used.append(k)
            return m.group(0)
        return ""

    return _CITE_RE.sub(repl, answer).strip(), used


def is_not_found(answer: str) -> bool:
    return answer.strip().rstrip("。.").strip() in {"未找到依据", "未找到依据："}


def build_notices(cited: list[Citation], reranked: bool, found: bool) -> list[str]:
    notices = []
    drafts = list(dict.fromkeys(c.title for c in cited if c.status == "draft"))
    for t in drafts:
        notices.append(f"《{t}》是讨论稿，尚未正式定稿，请以学院正式发布的文件为准。")
    if found and not reranked:
        notices.append("检索精排服务暂时不可用，参考资料按粗排结果提供，回答仅供参考。")
    if found and not cited:
        notices.append("回答没有标注出处，请对照下方参考资料核实。")
    return notices


def finalize(question: str, raw_answer: str, sources: list[Citation], reranked: bool) -> QAResult:
    """把模型的原始回答整理成最终结果：清理编号、回查引用、判断是否找到依据、追加提示。"""
    answer, used = clean_citations(raw_answer, len(sources))
    if is_not_found(answer):
        return not_found_result(question, sources, reranked)
    cited = [sources[i - 1] for i in used]
    return QAResult(
        question=question,
        answer=answer,
        found=True,
        citations=cited,
        sources=sources,
        notices=build_notices(cited, reranked, found=True),
        reranked=reranked,
    )


def not_found_result(
    question: str, sources: list[Citation] | None = None, reranked: bool = True
) -> QAResult:
    return QAResult(
        question=question,
        answer=NOT_FOUND_ANSWER,
        found=False,
        citations=[],
        sources=sources or [],
        notices=[],
        reranked=reranked,
    )
