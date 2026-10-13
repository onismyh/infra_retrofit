#!/usr/bin/env python3
"""Download the ISIMIP3b sectoral water-USE files needed to subtract domestic and irrigation.

Why this exists: `data/water/` holds only `qtot` (runoff). When this script was written
(2026-09-06) the water budget deducted other users through a single lumped knob,
`WATER_EXTRACTABLE_FRACTION x (1 - existing_withdrawal_share)` = 0.20 x 0.15, whose two factors
are ALIASED — the solver sees only their product, so the study cannot say whether the
environmental-flow standard or the allocation rule is what binds. Subtracting domestic and
irrigation explicitly was meant to de-alias that. The knob was deleted on 2026-09-26. Since
2026-10-01 the node limit is `max(qtot x 0.20 - domestic - irrigation, existing coal use)` (environmental
flow, on consumption; `optimization/water_access._water_available_by_node`): `scripts/build_water_use.py`
turns these files into the basin sums in `inputs/water_basin_use.csv`. The allocation rule is
still the official basin total-withdrawal quota (on withdrawal); see the water-budget comment in
`optimization/scenario.py`. `ptotuse` (cwatm, gfdl-esm4, ssp126) is also the province-to-basin
split weight when building the official basin quotas (`builders/water_quota.py`).

WHICH VARIABLES, AND WHY THESE ONES. The subtraction targets the node limit, which acts on
CONSUMPTION, so what must be deducted is other users' consumption, not their withdrawal
(irrigation returns a large share of what it diverts). The two hydrology models do not publish
the same set:

    CWatM         has pdomuse, pinduse, pliveuse, ptotuse   but NO pirruse
    WaterGAP2-2e  has pdomuse, pirruse                      but NO adomww

So irrigation consumption is taken directly from WaterGAP (`pirruse`) and derived for CWatM as
    pirruse = ptotuse - pdomuse - pinduse - pliveuse
which is why CWatM needs four files per scenario and WaterGAP only two.

Manufacturing/industrial consumption is deliberately NOT deducted: the model's industrial hubs choose
their routes, so their water is a decision variable charged to the basin withdrawal quota, not to the
node limit. ISIMIP's industrial use is exogenous and covers those same plants, so deducting it here
would count them once exogenously at the node and once by decision at the basin. The cost: industry
outside the model is not deducted from the node limit either.
Livestock (CWatM `pliveuse`) is not deducted either: only domestic and irrigation, the same pair
for both hydrology models.

Caveat to carry into the Methods: these are `2015soc-from-histsoc` runs, i.e. socioeconomic
drivers are frozen at 2015. Domestic demand is therefore effectively constant through 2100;
irrigation demand still varies because crop water requirement responds to climate.

Downloading is shared with `download_isimip_qtot.py` (`isimip_fetch.py`): requests+OpenSSL (this box's
mingw curl fails the files.isimip.org handshake), 2 workers, resume by byte range, retry budget spent on
stalls rather than attempts, and a file counts as complete only when its size equals the server's length.
"""
from __future__ import annotations

import sys

from _bootstrap import REPO_ROOT
from isimip_fetch import download_all

# 落到仓库根 data/water/，与工作目录无关（builders 从 REPO_ROOT/data 读）。下的是全球文件，下完用
# `python scripts/crop_isimip_china.py data/water` 原地裁到中国框；已有同名文件的不重下，裁过的也算。
DEST = REPO_ROOT / "data" / "water"
BASE = "https://files.isimip.org/ISIMIP3b/OutputData/water_global"
# Every GCM in `inputs/water_scenarios.csv`: each member's deduction comes from its own run.
GCMS = ("gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll")
SSPS = ("ssp126", "ssp370")
NEEDED: dict[str, tuple[str, tuple[str, ...]]] = {
    "cwatm": ("CWatM", ("pdomuse", "pinduse", "pliveuse", "ptotuse")),
    "watergap2-2e": ("WaterGAP2-2e", ("pdomuse", "pirruse")),
}


def _basename(hydro: str, gcm: str, ssp: str, var: str) -> str:
    return (f"{hydro}_{gcm}_w5e5_{ssp}_2015soc-from-histsoc_default_"
            f"{var}_global_monthly_2015_2100.nc")


URLS: list[tuple[str, str]] = [
    (_basename(h, gcm, ssp, var), f"{BASE}/{H}/{gcm}/future/{_basename(h, gcm, ssp, var)}")
    for h, (H, variables) in NEEDED.items()
    for gcm in GCMS
    for ssp in SSPS
    for var in variables
]


def main() -> None:
    if download_all(URLS, DEST):
        sys.exit(1)


if __name__ == "__main__":
    main()
