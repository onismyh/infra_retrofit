"""`scripts/isimip_fetch.fetch` 对本机 HTTP 服务：中途断开时按 Range 续传，416 视作已完整，服务器不理 Range 时从头写，
截断的 200 响应不算完成，一直没有进展时报错且不留下成品文件。"""
from __future__ import annotations

import importlib.util
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PAYLOAD = bytes(range(256)) * 40  # 10 240 字节


class _Handler(BaseHTTPRequestHandler):
    honour_range = True
    cut_after: int | None = None  # 每个响应只发这么多字节就断开（Content-Length 照报全长）
    status: int | None = None  # 不为 None 时一律回这个状态码、不带正文

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的接口名
        if self.status is not None:
            self.send_response(self.status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        match = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range", ""))
        start = int(match.group(1)) if match and self.honour_range else 0
        if start >= len(PAYLOAD):
            self.send_response(416)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = PAYLOAD[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body if self.cut_after is None else body[: self.cut_after])
        self.close_connection = True

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("Handler", (_Handler,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.nc", handler
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def fetcher(monkeypatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("isimip_fetch_under_test", SCRIPTS / "isimip_fetch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STALL_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(module, "MAX_STALLS", 3)
    monkeypatch.setattr(module, "CHUNK_BYTES", 1000)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    return module


def test_resumes_after_dropped_connections(server, fetcher, tmp_path) -> None:
    url, handler = server
    handler.cut_after = 3000
    final = tmp_path / "file.nc"
    assert fetcher.fetch(url, final) == len(PAYLOAD)
    assert final.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.nc.part").exists()


def test_416_on_a_full_part_file_is_complete(server, fetcher, tmp_path) -> None:
    url, _ = server
    (tmp_path / "file.nc.part").write_bytes(PAYLOAD)
    final = tmp_path / "file.nc"
    assert fetcher.fetch(url, final) == len(PAYLOAD)
    assert final.read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts_the_part_file(server, fetcher, tmp_path) -> None:
    url, handler = server
    handler.honour_range = False
    (tmp_path / "file.nc.part").write_bytes(b"x" * 500)
    final = tmp_path / "file.nc"
    assert fetcher.fetch(url, final) == len(PAYLOAD)
    assert final.read_bytes() == PAYLOAD


def test_truncated_200_responses_never_count_as_complete(server, fetcher, tmp_path) -> None:
    url, handler = server
    handler.honour_range, handler.cut_after = False, 3000
    final = tmp_path / "file.nc"
    with pytest.raises(RuntimeError, match="no progress"):
        fetcher.fetch(url, final)
    assert not final.exists()


def test_http_errors_end_as_no_progress(server, fetcher, tmp_path) -> None:
    url, handler = server
    handler.status = 503
    with pytest.raises(RuntimeError, match="no progress"):
        fetcher.fetch(url, tmp_path / "file.nc")
    assert fetcher.download_all([("file.nc", url)], tmp_path) == ["file.nc"]
