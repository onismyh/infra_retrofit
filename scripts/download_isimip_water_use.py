#!/usr/bin/env python3
"""Download the ISIMIP3b sectoral water-USE files needed to subtract domestic and irrigation.

Why this exists: `data/water/` holds only `qtot` (runoff). The basin budget currently deducts
other users through a single lumped knob, `WATER_EXTRACTABLE_FRACTION x (1 - existing_
withdrawal_share)` = 0.20 x 0.15, and `optimization/data_prep.py` states outright that those
two factors are ALIASED — the solver sees only their product, so the study cannot say whether
the environmental-flow standard or the allocation rule is what binds. Subtracting domestic and
irrigation explicitly de-aliases that.

WHICH VARIABLES, AND WHY THESE ONES. The basin constraint acts on CONSUMPTION, so what must be
deducted is other users' consumption, not their withdrawal (irrigation returns a large share of
what it diverts). The two hydrology models do not publish the same set:

    CWatM         has pdomuse, pinduse, pliveuse, ptotuse   but NO pirruse
    WaterGAP2-2e  has pdomuse, pirruse                      but NO adomww

So irrigation consumption is taken directly from WaterGAP (`pirruse`) and derived for CWatM as
    pirruse = ptotuse - pdomuse - pinduse - pliveuse
which is why CWatM needs four files per scenario and WaterGAP only two.

Manufacturing/industrial consumption is deliberately NOT deducted: industry becomes an explicit
decision agent in this model, so deducting it here as well would charge the same water twice.

Caveat to carry into the Methods: these are `2015soc-from-histsoc` runs, i.e. socioeconomic
drivers are frozen at 2015. Domestic demand is therefore effectively constant through 2100;
irrigation demand still varies because crop water requirement responds to climate.

Downloading follows `download_isimip_extra_gcms.py`: requests+OpenSSL (this box's mingw curl
fails the files.isimip.org handshake), 2 workers, resume by byte range, retry budget spent on
stalls rather than attempts.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEST = "data/water"
BASE = "https://files.isimip.org/ISIMIP3b/OutputData/water_global"
GCM = "gfdl-esm4"          # the headline GCM; extend when the other four are needed
SSPS = ("ssp126", "ssp370")
NEEDED: dict[str, tuple[str, tuple[str, ...]]] = {
    "cwatm": ("CWatM", ("pdomuse", "pinduse", "pliveuse", "ptotuse")),
    "watergap2-2e": ("WaterGAP2-2e", ("pdomuse", "pirruse")),
}

MAX_STALLS = 6
ATTEMPT_CAP = 200
MAX_WORKERS = 2
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 180


def _basename(hydro: str, ssp: str, var: str) -> str:
    return (f"{hydro}_{GCM}_w5e5_{ssp}_2015soc-from-histsoc_default_"
            f"{var}_global_monthly_2015_2100.nc")


URLS: list[tuple[str, str]] = [
    (_basename(h, ssp, var), f"{BASE}/{H}/{GCM}/future/{_basename(h, ssp, var)}")
    for h, (H, variables) in NEEDED.items()
    for ssp in SSPS
    for var in variables
]


def _local_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def fetch(url: str, fname: str) -> tuple[str, int]:
    final = os.path.join(DEST, fname)
    partial = final + ".part"
    if os.path.exists(final):
        return fname, _local_size(final)
    stalls = 0
    for _ in range(ATTEMPT_CAP):
        offset = _local_size(partial)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
                if response.status_code not in (200, 206):
                    raise OSError(f"HTTP {response.status_code}")
                mode = "ab" if offset and response.status_code == 206 else "wb"
                if mode == "wb":
                    offset = 0
                with open(partial, mode) as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            total = int(response.headers.get("content-range", "/0").split("/")[-1] or 0)
            if total and _local_size(partial) >= total:
                os.replace(partial, final)
                return fname, _local_size(final)
            if response.status_code == 200:
                os.replace(partial, final)
                return fname, _local_size(final)
        except Exception as exc:  # noqa: BLE001 - any transport error is a retry candidate
            print(f"  {fname}: {type(exc).__name__} {exc}", file=sys.stderr)
        # An attempt that moved the byte counter forward is progress and costs no budget.
        if _local_size(partial) > offset:
            stalls = 0
        else:
            stalls += 1
            if stalls >= MAX_STALLS:
                raise RuntimeError(f"{fname}: {MAX_STALLS} consecutive stalls")
            time.sleep(5)
    raise RuntimeError(f"{fname}: attempt cap reached")


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    todo = [(f, u) for f, u in URLS if not os.path.exists(os.path.join(DEST, f))]
    print(f"{len(URLS)} files needed, {len(todo)} missing")
    started = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch, url, fname): fname for fname, url in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, size = future.result()
                print(f"  done {name} ({size/1e6:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - report and continue with the rest
                print(f"  FAIL {name}: {exc}", file=sys.stderr)
    print(f"finished in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
