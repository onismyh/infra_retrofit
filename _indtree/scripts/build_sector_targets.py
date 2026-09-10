"""Build the sector emission-cap and output-index tables from China TIMES V2.0 CN60.

Writes two inputs the sector-target model reads (see `optimization/scenario.py`,
`sector_target_source`):

    inputs/sector_targets_times_cn60.csv        cap on each sector's residual emissions in every
                                                planning year, as a FRACTION OF ITS OWN 2030
                                                LEVEL in the same TIMES run
    inputs/industry_output_index_times_cn60.csv exogenous production index per industrial sector
                                                and planning year, 2030 = 1

Source workbook: China TIMES V2.0 `Exported_files/CN60-noTS.xlsx`, scenario `ndc_2c-cn60`,
national single region, annual 2019-2100. VEDA export layout: row 0 units, row 1 the pivot
label, row 2 the header (`Commodity`/`Process`, then years), data from row 3 -- read positionally.

Sector groups and what they are read off:

    power      CO2ELC + CCS-ELC-BIO (captured)          FOSSIL power-sector CO2: the net
                                                        trajectory with the BECCS credit added
                                                        back, floored at zero
    steel      CO2IIS + IND-TECH-IIS-PROC               energy + process
    cement     CO2IBU + IND-TECH-IBU                    energy + process (calcination)
    chemicals  CO2ICH                                   ammonia + methanol in this model.
                                                        NB: TIMES books methanol/CTL/coking
                                                        under CO2UPS, so this is the ammonia-
                                                        like part of the sector only.

The power cap is the FOSSIL trajectory (author's call, 2026-09-10). TIMES' net CO2ELC goes to
-273 / -792 Mt in 2050 / 2060 on the strength of 372 / 746 Mt captured at biomass power
(`EM-CO2-CCS Captured`, CCS-ELC-BIO), most of it dedicated biomass plants this model does not
contain. Scaling the net series onto the coal fleet would demand that the fleet alone reach
-223 / -648 Mt, which the 2026-09-10 smoke solve showed lands in `target_shortfall_power`
(163 / 377 Mt) and dominates the objective through the slack penalty. Adding the BECCS
capture back gives the sector's fossil combustion emissions -- 5 104 / 3 489 / 100 / -46 Mt --
which is the quantity the coal fleet's residual (BECCS credit netted, exactly as here) is
capped by. The 2060 value is floored at zero: the fleet must be net-zero, and any negative
contribution is a result, not a requirement.

Output index (production, Mt product):
    steel_bf_bof  IND_IIS_STEALOXY
    steel_eaf     IND_IIS_STEALELC
    cement        sum of IND_IBU_CEMMAT2-* incl. CCS
    ammonia       sum of IND_ICH_NH3*
    methanol      NO TIMES SERIES -- takes the ammonia index as the chemicals proxy (flagged)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT

logger = logging.getLogger(__name__)

TIMES_XLSX = Path(r"D:\6. Transfer\China-TIMES2.0\China-TIMES-V2.0\Exported_files\CN60-noTS.xlsx")
PLANNING_YEARS = (2030, 2040, 2050, 2060)
BASE_YEAR = 2030
SOURCE_TAG = "times_cn60"


def _sheet_rows(sheet: str) -> dict[str, dict[int, float]]:
    """{row name: {year: value}} for one VEDA export sheet."""
    raw = pd.read_excel(TIMES_XLSX, sheet_name=sheet, header=None)
    header = raw.iloc[2].tolist()
    year_cols: dict[int, int] = {}
    for index, label in enumerate(header):
        try:
            year = int(float(label))
        except (TypeError, ValueError):
            continue
        if year in PLANNING_YEARS:
            year_cols[year] = index
    name_col = next((header.index(c) for c in ("Commodity", "Process") if c in header), 3)
    rows: dict[str, dict[int, float]] = {}
    for r in range(3, len(raw)):
        name = raw.iat[r, name_col]
        if pd.isna(name):
            continue
        rows[str(name)] = {
            year: (float(raw.iat[r, col]) if pd.notna(raw.iat[r, col]) else 0.0)
            for year, col in year_cols.items()
        }
    return rows


def _series(rows: dict[str, dict[int, float]], *names: str) -> dict[int, float]:
    out = {year: 0.0 for year in PLANNING_YEARS}
    for name in names:
        if name not in rows:
            raise KeyError(f"{name!r} not found; have {sorted(rows)[:12]}...")
        for year in PLANNING_YEARS:
            out[year] += rows[name][year]
    return out


def _prefix_sum(rows: dict[str, dict[int, float]], prefix: str) -> dict[int, float]:
    names = [name for name in rows if name.startswith(prefix)]
    if not names:
        raise KeyError(f"no rows start with {prefix!r}")
    return _series(rows, *names)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    co2_all = _sheet_rows("EM-CO2-ALL")
    ccs_captured = _sheet_rows("EM-CO2-CCS Captured")
    co2_ind = _sheet_rows("EM-CO2-Industry")
    co2_proc = _sheet_rows("EM-CO2-INDPROC")
    steel = _sheet_rows("IND-IIS-STEEL")
    cement = _sheet_rows("IND-IBU-Cement")
    nh3 = _sheet_rows("IND-ICH-NH3")

    power_net = _series(co2_all, "CO2ELC")
    power_beccs = _series(ccs_captured, "CCS-ELC-BIO")
    emissions = {
        "power": {y: max(0.0, power_net[y] + power_beccs[y]) for y in PLANNING_YEARS},
        "steel": {y: a + b for (y, a), b in zip(_series(co2_ind, "CO2IIS").items(),
                                                 _series(co2_proc, "IND-TECH-IIS-PROC").values())},
        "cement": {y: a + b for (y, a), b in zip(_series(co2_ind, "CO2IBU").items(),
                                                  _series(co2_proc, "IND-TECH-IBU").values())},
        "chemicals": _series(co2_ind, "CO2ICH"),
    }
    target_rows = []
    for group, series in emissions.items():
        base = series[BASE_YEAR]
        if base <= 0:
            raise ValueError(f"{group}: non-positive {BASE_YEAR} emissions {base}")
        for year in PLANNING_YEARS:
            target_rows.append({
                "sector_group": group,
                "planning_year": year,
                "times_mt": round(series[year], 1),
                "cap_fraction_of_2030": round(series[year] / base, 4),
                "source": (
                    f"China TIMES V2.0 ndc_2c-cn60, CN60-noTS.xlsx ({SOURCE_TAG})"
                    + ("; power = CO2ELC + CCS-ELC-BIO captured, floored at 0" if group == "power" else "")
                ),
            })
    targets = pd.DataFrame(target_rows)

    output = {
        "steel_bf_bof": _series(steel, "IND_IIS_STEALOXY"),
        "steel_eaf": _series(steel, "IND_IIS_STEALELC"),
        "cement": _prefix_sum(cement, "IND_IBU_CEM"),
        "ammonia": _prefix_sum(nh3, "IND_ICH_NH3"),
    }
    output["methanol"] = dict(output["ammonia"])  # proxy, see module docstring
    index_rows = []
    for sector, series in output.items():
        base = series[BASE_YEAR]
        for year in PLANNING_YEARS:
            index_rows.append({
                "sector": sector,
                "planning_year": year,
                "times_output_mt": round(series[year], 1),
                "output_index": round(series[year] / base, 4),
                "basis": "ammonia proxy (no TIMES methanol series)" if sector == "methanol" else "TIMES sub-technology output",
            })
    index = pd.DataFrame(index_rows)
    return targets, index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not TIMES_XLSX.exists():
        raise FileNotFoundError(TIMES_XLSX)
    targets, index = build()
    inputs = ROOT / "inputs"
    targets.to_csv(inputs / f"sector_targets_{SOURCE_TAG}.csv", index=False)
    index.to_csv(inputs / f"industry_output_index_{SOURCE_TAG}.csv", index=False)
    print(targets.pivot(index="sector_group", columns="planning_year", values="cap_fraction_of_2030"))
    print(index.pivot(index="sector", columns="planning_year", values="output_index"))


if __name__ == "__main__":
    main()
