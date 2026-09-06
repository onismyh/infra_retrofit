#!/usr/bin/env python3
"""Download the 16 ISIMIP3b qtot files reliably on Windows.

Why Python+requests instead of curl: this box's curl is a mingw build using the
Schannel TLS backend, which fails the handshake against files.isimip.org
("schannel: server closed abruptly"). Python's `requests` uses OpenSSL (via
certifi) and connects cleanly (verified: HTTP 200, content-length matches,
byte-range supported). The original data set was also downloaded this way.

Two packages, 24 files, ~8.2 GB total:
  future      (cwatm + watergap2-2e) x 4 extra GCMs x {ssp126, ssp370} = 16, ~4.2 GB
  historical  (cwatm + watergap2-2e) x 4 extra GCMs                    =  8, ~4.0 GB

The historical half is a prerequisite, not a bonus: bias correction is estimated per
(hydrology model, GCM) pair against that pair's own historical run, so a member whose
historical file is absent stays uncorrected while its siblings are corrected -- a
correction-on/off contrast that would masquerade as GCM spread in the ensemble.

Naming follows the existing convention so `builders/water.py`'s WATER_PATTERN parses
the future files. Resume-safe: on a broken connection we restart from the byte offset
already on disk (Range request); the retry budget is spent on stalls, not attempts.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEST = "data/water"
GCMS = ["ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
SSPS = ["ssp126", "ssp370"]
HYD = {"cwatm": "CWatM", "watergap2-2e": "WaterGAP2-2e"}
# files.isimip.org drops long TLS transfers well before a 261 MB file completes -- the first
# run got every file to ~170 MB and the second to ~252 MB, then hit a fixed 5-attempt cap and
# threw the remaining 9 MB away. So the retry budget is spent on *stalls*, not on attempts:
# an attempt that moved the byte counter forward is progress and costs nothing, and only
# MAX_STALLS consecutive zero-progress attempts end the file. STALL_CAP bounds the pathological
# case where the server hands back a few bytes at a time forever.
MAX_STALLS = 6
ATTEMPT_CAP = 200
# Two workers, not four: on the second run only the first 4 files ever wrote a byte and the
# other 12 got instant SSLErrors, which reads as the server refusing a 4th/5th concurrent
# connection from one client after the earlier hammering.
MAX_WORKERS = 2
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 180

# The local name is byte-identical to the remote one -- builders/water.py's WATER_PATTERN keys
# off the `w5e5` token, so it must survive into the filename on disk.
def _basename(h: str, g: str, s: str) -> str:
    return f"{h}_{g}_w5e5_{s}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"


def _historical_basename(h: str, g: str) -> str:
    return f"{h}_{g}_w5e5_historical_histsoc_default_qtot_global_monthly_1850_2014.nc"


# 16 future files (hydro x gcm x ssp), ~4.2 GB.
FUTURE_URLS = [
    (
        _basename(h, g, s),
        f"https://files.isimip.org/ISIMIP3b/OutputData/water_global/{HYD[h]}/{g}/future/"
        f"{_basename(h, g, s)}",
    )
    for h in HYD
    for g in GCMS
    for s in SSPS
]

# 8 historical files (hydro x gcm), ~4.0 GB. Not optional: `builders/water.basin_bias_factors`
# estimates each member's basin bias against ITS OWN (hydrology model, GCM) historical run, so a
# missing historical file silently leaves that member uncorrected at factor 1.0. Shipping an
# ensemble where gfdl-esm4 is bias-corrected and the other four GCMs are raw would be worse than
# no correction at all -- WaterGAP2-2e runs 2.43x too wet in Hai, so the uncorrected members
# would separate from gfdl mostly through the correction rather than through their hydrology,
# and Fig 3(c) would report that artefact as GCM variance.
HISTORICAL_URLS = [
    (
        _historical_basename(h, g),
        f"https://files.isimip.org/ISIMIP3b/OutputData/water_global/{HYD[h]}/{g}/historical/"
        f"{_historical_basename(h, g)}",
    )
    for h in HYD
    for g in GCMS
]

URLS = FUTURE_URLS + HISTORICAL_URLS


def _local_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _finish(partial: str, final: str) -> None:
    os.replace(partial, final)


def fetch(url: str, fname: str) -> tuple[str, int]:
    """Download one file with byte-range resume; returns (fname, final_bytes).

    Retries are budgeted against progress rather than attempt count: a dropped connection
    that still advanced the `.part` file resets the stall counter, so a file that needs
    twenty reconnections to walk through 261 MB succeeds. Only MAX_STALLS consecutive
    attempts that move zero bytes are treated as a real failure.
    """
    final = os.path.join(DEST, fname)
    partial = final + ".part"
    stalls = 0
    for attempt in range(1, ATTEMPT_CAP + 1):
        offset = _local_size(partial)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        total = None
        try:
            with requests.get(
                url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers=headers
            ) as r:
                if r.status_code in (200, 206):
                    total = int(r.headers.get("content-length") or 0) + (offset if r.status_code == 206 else 0)
                    mode = "ab" if (offset and r.status_code == 206) else "wb"
                    with open(partial, mode) as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                elif r.status_code == 416 and offset:
                    # Already have every byte the server has; treat as complete.
                    total = offset
                else:
                    print(f"  [{fname} a{attempt}] HTTP {r.status_code}", flush=True)
        except requests.RequestException as e:
            print(f"  [{fname} a{attempt}] {e.__class__.__name__}", flush=True)

        got = _local_size(partial)
        if total and got == total:
            _finish(partial, final)
            return fname, total
        if got > offset:
            stalls = 0
            pct = f"{100.0 * got / total:.1f}%" if total else "?"
            print(f"  [{fname} a{attempt}] +{(got - offset) / 1e6:.1f} MB -> {got / 1e6:.1f} MB ({pct})", flush=True)
        else:
            stalls += 1
            if stalls >= MAX_STALLS:
                raise RuntimeError(f"{MAX_STALLS} consecutive attempts made no progress at {got} bytes: {fname}")
        # Back off on stalls only; a productive attempt reconnects immediately.
        time.sleep(5 * stalls)
    raise RuntimeError(f"hit the {ATTEMPT_CAP}-attempt cap: {fname}")


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    # Keep (fname, url) ordering identical to URLS all the way to ex.submit -- swapping the
    # two here is what made every request fail with MissingSchema on the first attempt.
    todo = [
        (fn, u)
        for fn, u in URLS
        if not (os.path.exists(os.path.join(DEST, fn)) and os.path.getsize(os.path.join(DEST, fn)) > 0)
    ]
    if not todo:
        print("all files already present")
        return
    print(f"{len(todo)} files to fetch", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch, u, fn): fn for fn, u in todo}
        for fut in as_completed(futs):
            fn = futs[fut]
            try:
                fname, nbytes = fut.result()
                print(f"OK   {nbytes/1e6:7.1f} MB  {fname}", flush=True)
            except Exception as e:
                failures.append(fn)
                print(f"FAIL {fn}: {e}", flush=True)
    print(f"\n=== DONE ===  ok={len(todo)-len(failures)} failed={len(failures)}")
    if failures:
        print("failed:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
