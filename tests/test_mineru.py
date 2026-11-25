import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from cqu_affairs.core.config import MinerUSettings
from cqu_affairs.rag.manifest import MANIFEST_PATH, DocumentMeta, load_manifest
from cqu_affairs.rag.mineru import MinerUClient, MinerUError, safe_extract
from cqu_affairs.rag.parsing import parse_documents, parsed_dir, read_meta

UPLOAD = "https://oss.example/upload/"
ZIP = "https://cdn.example/result.zip"


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _settings() -> MinerUSettings:
    return MinerUSettings(api_key="tok", poll_interval=0, timeout=5, _env_file=None)


async def _no_sleep(_):
    return None


class FakeMinerU:
    """模拟 MinerU：申请链接 -> PUT 上传 -> 两次轮询后完成 -> 下载 zip。"""

    def __init__(self, fail_ids: set[str] = frozenset()):
        self.fail_ids = fail_ids
        self.files: list[dict] = []
        self.puts: list[httpx.Request] = []
        self.polls = 0
        self.submits = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url.endswith("/file-urls/batch"):
            assert req.headers["Authorization"] == "Bearer tok"
            body = json.loads(req.content)
            self.files = body["files"]
            self.submits += 1
            urls = [f"{UPLOAD}{i}" for i in range(len(self.files))]
            return httpx.Response(
                200, json={"code": 0, "data": {"batch_id": "b1", "file_urls": urls}}
            )
        if url.startswith(UPLOAD):
            self.puts.append(req)
            return httpx.Response(200)
        if "/extract-results/batch/b1" in url:
            self.polls += 1
            state = "running" if self.polls < 2 else None
            res = []
            for f in self.files:
                s = state or ("failed" if f["data_id"] in self.fail_ids else "done")
                res.append(
                    {
                        "file_name": f["name"],
                        "data_id": f["data_id"],
                        "state": s,
                        "err_msg": "boom" if s == "failed" else "",
                        "full_zip_url": ZIP if s == "done" else "",
                    }
                )
            return httpx.Response(
                200, json={"code": 0, "data": {"batch_id": "b1", "extract_result": res}}
            )
        if url == ZIP:
            return httpx.Response(
                200,
                content=_zip(
                    {
                        "full.md": "# 第一章 总则\n",
                        "x_content_list.json": "[]",
                        "x_content_list_v2.json": "[]",
                        "x_origin.pdf": "%PDF",
                        "images/a.jpg": "img",
                    }
                ),
            )
        raise AssertionError(f"unexpected {req.method} {url}")


def _doc(path: str, **kw) -> DocumentMeta:
    base = dict(
        path=path, title=Path(path).stem, level="school", status="effective", structure="numbered"
    )
    return DocumentMeta.model_validate(base | kw)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    (root / "leave").mkdir(parents=True)
    (root / "leave" / "请假.pdf").write_bytes(b"%PDF-a")
    (root / "leave" / "留校.pdf").write_bytes(b"%PDF-b")
    return root


async def test_parse_writes_outputs_and_meta(raw, tmp_path):
    fake = FakeMinerU()
    docs = [
        _doc("leave/请假.pdf", doc_no="X〔2021〕1号", publish_date="2021-05-17"),
        _doc("leave/留校.pdf"),
    ]
    out = tmp_path / "parsed"
    async with MinerUClient(_settings(), transport=httpx.MockTransport(fake), sleep=_no_sleep) as c:
        report = await parse_documents(
            docs, c, "vlm", raw_root=raw, out_root=out, log=lambda *_: None
        )

    assert set(report.parsed) == {"leave/请假.pdf", "leave/留校.pdf"}
    # 上传请求不能带 Content-Type（OSS 签名要求）
    assert all("content-type" not in r.headers for r in fake.puts)
    assert [f["name"] for f in fake.files] == ["请假.pdf", "留校.pdf"]

    dest = parsed_dir(docs[0], out)
    assert (dest / "full.md").read_text() == "# 第一章 总则\n"
    meta = read_meta(docs[0], out)
    assert meta["source"]["doc_no"] == "X〔2021〕1号"
    assert meta["source"]["publish_date"] == "2021-05-17"
    assert meta["parser"] == meta["parser"] | {
        "name": "mineru",
        "model_version": "vlm",
        "batch_id": "b1",
    }
    assert meta["outputs"]["markdown"] == "full.md"
    assert meta["outputs"]["content_list"] == "x_content_list.json"
    assert meta["outputs"]["content_list_v2"] == "x_content_list_v2.json"
    assert meta["outputs"]["origin_pdf"] == "x_origin.pdf"
    assert len(meta["doc_id"]) == 16


async def test_unchanged_files_are_skipped_but_meta_refreshed(raw, tmp_path):
    out = tmp_path / "parsed"
    doc = _doc("leave/请假.pdf")
    fake = FakeMinerU()
    async with MinerUClient(_settings(), transport=httpx.MockTransport(fake), sleep=_no_sleep) as c:
        await parse_documents([doc], c, "vlm", raw_root=raw, out_root=out, log=lambda *_: None)
        updated = _doc("leave/请假.pdf", publish_date="2022-01-01")
        report = await parse_documents(
            [updated], c, "vlm", raw_root=raw, out_root=out, log=lambda *_: None
        )

    assert report.skipped == ["leave/请假.pdf"] and report.parsed == []
    assert fake.submits == 1
    assert read_meta(doc, out)["source"]["publish_date"] == "2022-01-01"


async def test_failed_file_reported(raw, tmp_path):
    from cqu_affairs.rag.manifest import file_sha256
    from cqu_affairs.rag.parsing import doc_id_of

    bad = doc_id_of(file_sha256(raw / "leave" / "留校.pdf"))
    fake = FakeMinerU(fail_ids={bad})
    docs = [_doc("leave/请假.pdf"), _doc("leave/留校.pdf")]
    async with MinerUClient(_settings(), transport=httpx.MockTransport(fake), sleep=_no_sleep) as c:
        report = await parse_documents(
            docs, c, "vlm", raw_root=raw, out_root=tmp_path / "p", log=lambda *_: None
        )
    assert report.parsed == ["leave/请假.pdf"]
    assert report.failed == {"leave/留校.pdf": "boom"}


async def test_api_error_code_raises():
    def handler(req):
        return httpx.Response(200, json={"code": "A0211", "msg": "Token 过期", "data": None})

    async with MinerUClient(_settings(), transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(MinerUError, match="A0211"):
            await c.get_batch("b1")


def test_safe_extract_rejects_zip_slip(tmp_path):
    with pytest.raises(MinerUError):
        safe_extract(_zip({"../evil.txt": "x"}), tmp_path / "out")


def test_requires_api_key():
    with pytest.raises(RuntimeError):
        MinerUClient(MinerUSettings(api_key=None, _env_file=None))


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="校内材料不入库，仅本地校验")
def test_project_manifest_is_valid():
    docs = load_manifest()
    assert docs, "manifest 为空"
    for d in docs:
        assert (Path("data/raw") / d.path).exists(), d.path


async def test_http_401_shows_mineru_message():
    def handler(req):
        return httpx.Response(401, json={"msgCode": "A0202", "msg": "user authenticate failed"})

    async with MinerUClient(_settings(), transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(MinerUError, match="A0202.*apiManage"):
            await c.get_batch("b1")
