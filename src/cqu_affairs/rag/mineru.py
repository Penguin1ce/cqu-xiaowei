# Copyright (c) 2026 Penguin1ce <https://github.com/Penguin1ce>
# SPDX-License-Identifier: AGPL-3.0-only
"""MinerU 云端精准解析 API 客户端（本地文件批量上传）。

流程（https://mineru.net/apiManage/docs）：

1. POST {base_url}/file-urls/batch 申请上传链接 -> data.batch_id、data.file_urls
2. 逐个 PUT 文件到 file_urls[i]（不能设置 Content-Type），上传完成后系统自动提交解析
3. 轮询 GET {base_url}/extract-results/batch/{batch_id}，
   state: waiting-file / pending / running / converting / done / failed
4. 下载 full_zip_url（含 full.md、*_content_list.json、images/ 等）并解压

接口统一返回 {"code": 0, "data": ..., "msg": "ok"}，code 非 0 即失败。
"""

import asyncio
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from cqu_affairs.core.config import MinerUSettings

# 单次申请上传链接的上限（官方文档两处分别写 50 / 200，取保守值）
MAX_BATCH_FILES = 50
_FINAL_STATES = {"done", "failed"}


class MinerUError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadItem:
    path: Path
    data_id: str
    """回传用的业务 ID，仅允许字母、数字、_ - .，不超过 128 字符。"""


@dataclass(frozen=True)
class ExtractResult:
    data_id: str
    file_name: str
    state: str
    err_msg: str = ""
    full_zip_url: str | None = None


class MinerUClient:
    def __init__(
        self,
        settings: MinerUSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        if settings.api_key is None:
            raise RuntimeError("未配置 MINERU_API_KEY")
        self._s = settings
        self._sleep = sleep
        # API 请求带鉴权；上传（OSS 签名链接）和下载走不带鉴权的客户端
        self._api = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=60.0,
            transport=transport,
        )
        self._raw = httpx.AsyncClient(timeout=300.0, transport=transport, follow_redirects=True)

    async def aclose(self) -> None:
        await self._api.aclose()
        await self._raw.aclose()

    async def __aenter__(self) -> "MinerUClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @staticmethod
    def _unwrap(resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.is_error:
            # 鉴权失败等情况返回 {"msgCode": "A0202", "msg": ...}
            code = body.get("code") or body.get("msgCode")
            hint = "（Token 无效或已过期，请到 https://mineru.net/apiManage 重新申请）"
            raise MinerUError(
                f"MinerU HTTP {resp.status_code} code={code}：{body.get('msg', resp.text[:200])}"
                + (hint if resp.status_code == 401 else "")
            )
        if body.get("code") != 0:
            raise MinerUError(f"MinerU 返回错误 code={body.get('code')}：{body.get('msg')}")
        return body["data"]

    async def submit_batch(self, items: list[UploadItem]) -> str:
        """申请上传链接并上传文件，返回 batch_id。上传完成后 MinerU 自动开始解析。"""
        if not items:
            raise ValueError("items 不能为空")
        if len(items) > MAX_BATCH_FILES:
            raise ValueError(f"单批最多 {MAX_BATCH_FILES} 个文件")
        payload = {
            "files": [
                {"name": it.path.name, "data_id": it.data_id, "is_ocr": self._s.is_ocr}
                for it in items
            ],
            "model_version": self._s.model_version,
            "language": self._s.language,
            "enable_table": self._s.enable_table,
            "enable_formula": self._s.enable_formula,
        }
        data = self._unwrap(await self._api.post("/file-urls/batch", json=payload))
        urls: list[str] = data["file_urls"]
        if len(urls) != len(items):
            raise MinerUError(f"上传链接数量 {len(urls)} 与文件数量 {len(items)} 不一致")
        for it, url in zip(items, urls, strict=True):
            # 传 bytes 时 httpx 不会自动加 Content-Type，符合 OSS 签名要求
            resp = await self._raw.put(url, content=it.path.read_bytes())
            if resp.status_code != 200:
                raise MinerUError(f"上传 {it.path.name} 失败：HTTP {resp.status_code}")
        return data["batch_id"]

    async def get_batch(self, batch_id: str) -> list[ExtractResult]:
        data = self._unwrap(await self._api.get(f"/extract-results/batch/{batch_id}"))
        return [
            ExtractResult(
                data_id=r.get("data_id", ""),
                file_name=r.get("file_name", ""),
                state=r.get("state", ""),
                err_msg=r.get("err_msg", "") or "",
                full_zip_url=r.get("full_zip_url") or None,
            )
            for r in data.get("extract_result", [])
        ]

    async def wait_batch(
        self, batch_id: str, expected: int, on_progress=None
    ) -> list[ExtractResult]:
        """轮询直到 expected 个文件都进入 done / failed，超时抛 TimeoutError。"""
        deadline = time.monotonic() + self._s.timeout
        while True:
            results = await self.get_batch(batch_id)
            if on_progress:
                on_progress(results)
            if len(results) >= expected and all(r.state in _FINAL_STATES for r in results):
                return results
            if time.monotonic() > deadline:
                raise TimeoutError(f"MinerU 批次 {batch_id} 在 {self._s.timeout}s 内未完成")
            await self._sleep(self._s.poll_interval)

    async def download_result(self, zip_url: str, dest: Path) -> list[Path]:
        """下载结果 zip 并解压到 dest，返回解压出的文件列表。"""
        resp = await self._raw.get(zip_url)
        resp.raise_for_status()
        return safe_extract(resp.content, dest)


def safe_extract(zip_bytes: bytes, dest: Path) -> list[Path]:
    """解压 zip，拒绝写出 dest 之外的路径（防 zip slip）。"""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(dest):
                raise MinerUError(f"压缩包包含非法路径：{info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            out.append(target)
    return out
