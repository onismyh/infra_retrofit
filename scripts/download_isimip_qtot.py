#!/usr/bin/env python3
"""Download the 30 ISIMIP3b qtot files (20 future + 10 historical), ~10.2 GB, with byte-range resume.

  future      (cwatm + watergap2-2e) x 5 GCMs x {ssp126, ssp370} = 20, ~5.2 GB
  historical  (cwatm + watergap2-2e) x 5 GCMs                     = 10, ~5.0 GB

The gfdl-esm4 future pair -- the members of the head scenario `ST_WA_cwatm_126_dry_oq` -- is in the
list since 2026-10-13 (file renamed from `download_isimip_extra_gcms.py`, which fetched only the
four other GCMs; the gfdl-esm4 future files had no download script). Download mechanics (requests
instead of curl, resume, stall budget, two workers) are in `isimip_fetch.py`.

The historical half is a prerequisite, not a bonus: bias correction is estimated per
(hydrology model, GCM) pair against that pair's own historical run, so a member whose
historical file is absent stays uncorrected while its siblings are corrected -- a
correction-on/off contrast that would masquerade as GCM spread in the ensemble. The
gfdl-esm4 historical pair is in the list too (2026-10-12; until then it had no download script).

The CWatM historical files are kept under the names the server gives them, although those
names are wrong: ISIMIP issue #34 (data.isimip.org/issues/34) shifted CWatM's historical qtot
by one GCM -- the file labelled ipsl-cm6a-lr is GFDL's run (same qtot data cell for cell as the gfdl
file; only the header's GCM name differs), mpi-esm1-2-hr is IPSL's, mri-esm2-0 is MPI's, ukesm1-0-ll
is MRI's, and UKESM's own run does not exist; only the future files were re-issued (2023-02-16).
`builders/water_bias.HISTORICAL_FILE_GCM`
maps each member to the file that actually drove it, so the ukesm-labelled file is still needed
(it is the mri member's baseline). Verified 2026-10-12 with `scripts/attribute_isimip_forcing.py`.

Naming follows the existing convention so `builders/water.py`'s WATER_PATTERN parses
the future files. Resume-safe: on a broken connection we restart from the byte offset
already on disk (Range request); the retry budget is spent on stalls, not attempts.
"""
from __future__ import annotations

import sys

from _bootstrap import REPO_ROOT
from isimip_fetch import download_all

# 落到仓库根 data/water/，与工作目录无关（builders 从 REPO_ROOT/data 读）。下的是全球文件，下完用
# `python scripts/crop_isimip_china.py data/water` 原地裁到中国框；已有同名文件的不重下，裁过的也算。
DEST = REPO_ROOT / "data" / "water"
GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
SSPS = ["ssp126", "ssp370"]
HYD = {"cwatm": "CWatM", "watergap2-2e": "WaterGAP2-2e"}
BASE = "https://files.isimip.org/ISIMIP3b/OutputData/water_global"


# The local name is byte-identical to the remote one -- builders/water.py's WATER_PATTERN keys
# off the `w5e5` token, so it must survive into the filename on disk.
def _basename(h: str, g: str, s: str) -> str:
    return f"{h}_{g}_w5e5_{s}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"


def _historical_basename(h: str, g: str) -> str:
    return f"{h}_{g}_w5e5_historical_histsoc_default_qtot_global_monthly_1850_2014.nc"


# 20 future files (hydro x gcm x ssp), ~5.2 GB.
FUTURE_URLS = [
    (_basename(h, g, s), f"{BASE}/{HYD[h]}/{g}/future/{_basename(h, g, s)}")
    for h in HYD
    for g in GCMS
    for s in SSPS
]

# 10 historical files (hydro x gcm), ~5.0 GB. Not optional: `builders/water_bias.basin_bias_factors`
# estimates each member's basin bias against ITS OWN (hydrology model, GCM) historical run, so a
# missing historical file silently leaves that member uncorrected at factor 1.0. Shipping an
# ensemble where gfdl-esm4 is bias-corrected and the other four GCMs are raw would be worse than
# no correction at all -- WaterGAP2-2e runs 2.43x too wet in Hai, so the uncorrected members
# would separate from gfdl mostly through the correction rather than through their hydrology,
# and any GCM-spread panel would report that artefact as GCM variance. For CWatM "its own" run is
# the file `builders/water_bias.HISTORICAL_FILE_GCM` points to, not the one carrying the member's name
# (see the module docstring).
HISTORICAL_URLS = [
    (_historical_basename(h, g), f"{BASE}/{HYD[h]}/{g}/historical/{_historical_basename(h, g)}")
    for h in HYD
    for g in GCMS
]

URLS = FUTURE_URLS + HISTORICAL_URLS


def main() -> None:
    if download_all(URLS, DEST):
        sys.exit(1)


if __name__ == "__main__":
    main()
