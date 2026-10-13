"""ISIMIP3b 文件的断点续传下载，`download_isimip_qtot.py` 与 `download_isimip_water_use.py` 共用（2026-10-13 合并，
此前两份各写一个 `fetch()`，判完成的条件不同：耗水那份对 HTTP 200 不核对长度、读完流就改名，连接提前断开而没被当成
错误时——urllib3 1.x 缺省不核对 Content-Length，2.x 才报 IncompleteRead——截断的文件会被当成完整文件）。

用 requests（OpenSSL）而不用 curl：作者那台机器的 curl 是 mingw 版，走 Schannel，与 files.isimip.org 握手失败。
files.isimip.org 会在长 TLS 传输中途断开（261 MB 的文件第一轮都停在约 170 MB，第二轮约 252 MB），所以按已写入的字节
续传（Range 请求），重试预算花在没有进展的尝试上：一次尝试只要写进了字节就不计数，连续 `MAX_STALLS` 次零进展才算
失败；`ATTEMPT_CAP` 兜底服务器永远只给几个字节的情形。完成只认一条：本地字节数等于服务器给的总长（200 为
content-length，206 为偏移 + content-length，416 为已有全部字节）。两路并发：四路时只有前 4 个文件写进过字节，
其余立刻 SSLError，像是服务器拒绝同一客户端的第 4、5 个并发连接。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MAX_STALLS = 6
ATTEMPT_CAP = 200
MAX_WORKERS = 2
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 180
STALL_BACKOFF_SECONDS = 5.0
# 断线时还在读缓冲里、不满一块的字节会丢掉，续传从已写盘的字节开始。
CHUNK_BYTES = 1 << 20


def _local_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def fetch(url: str, final: Path) -> int:
    """把 `url` 下到 `final`（先写 `final.part`，完整后改名），返回字节数；连续 `MAX_STALLS` 次没有进展或到
    `ATTEMPT_CAP` 次时报 RuntimeError。"""
    partial = final.with_name(final.name + ".part")
    stalls = 0
    for attempt in range(1, ATTEMPT_CAP + 1):
        offset = _local_size(partial)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        total = 0
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
                if response.status_code in (200, 206):
                    # 200 是服务器没理 Range、从头发：.part 从头写。进展仍与本次之前的字节数比，服务器一直不理 Range
                    # 又一直中途断开时按没有进展计，不会耗到 ATTEMPT_CAP。
                    resumed = response.status_code == 206
                    total = int(response.headers.get("content-length") or 0) + (offset if resumed else 0)
                    with open(partial, "ab" if resumed else "wb") as handle:
                        for chunk in response.iter_content(CHUNK_BYTES):
                            handle.write(chunk)
                elif response.status_code == 416 and offset:
                    total = offset  # Range 从服务器的文件长度开始：.part 已有全部字节
                else:
                    print(f"  [{final.name} a{attempt}] HTTP {response.status_code}", flush=True)
        except requests.RequestException as exc:
            print(f"  [{final.name} a{attempt}] {type(exc).__name__}", flush=True)

        got = _local_size(partial)
        if total and got == total:
            partial.replace(final)
            return got
        if got > offset:
            stalls = 0
            share = f"{100.0 * got / total:.1f}%" if total else "?"
            print(f"  [{final.name} a{attempt}] +{(got - offset) / 1e6:.1f} MB -> {got / 1e6:.1f} MB ({share})", flush=True)
        else:
            stalls += 1
            if stalls >= MAX_STALLS:
                raise RuntimeError(f"{MAX_STALLS} consecutive attempts made no progress at {got} bytes: {final.name}")
        time.sleep(STALL_BACKOFF_SECONDS * stalls)  # 只在没有进展时退避，有进展的尝试立刻重连
    raise RuntimeError(f"hit the {ATTEMPT_CAP}-attempt cap: {final.name}")


def download_all(urls: list[tuple[str, str]], dest: Path) -> list[str]:
    """把 `urls`（[(文件名, url)]）里 `dest` 下还没有（或为空）的文件下完，返回失败的文件名。"""
    dest.mkdir(parents=True, exist_ok=True)
    todo = [(name, url) for name, url in urls if _local_size(dest / name) == 0]
    print(f"{len(urls)} files needed, {len(todo)} missing", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch, url, dest / name): name for name, url in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                size = future.result()
                print(f"OK   {size / 1e6:7.1f} MB  {name}", flush=True)
            except (RuntimeError, OSError, ValueError) as exc:
                failures.append(name)
                print(f"FAIL {name}: {exc}", flush=True)
    print(f"ok={len(todo) - len(failures)} failed={len(failures)}")
    if failures:
        print("failed:", *failures, sep="\n  ")
    return failures
