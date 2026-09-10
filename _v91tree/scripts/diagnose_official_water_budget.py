"""Per-basin water budget on the official 用水总量控制指标 basis, against the runoff basis.

Diagnostic only -- writes nothing to `inputs/`. Answers the one question that decides whether
the switch is viable: after reserving what the 水资源公报 says other users already withdraw,
does the official cap leave the coal fleet more or less room than `qtot x 0.20 x 0.15` did?

BASIS. The caps are enforced against 用水量 (公报 编写说明 2.(7)), which counts 直流火(核)电
cooling inside 工业用水. So the fleet is priced on WITHDRAWAL -- recalibrated onto the bulletin
by `once_through_calibration`, because the raw table's once-through rows are US values that put
coal alone above national industrial withdrawal. The 定额 (quota) and consumption columns are
printed alongside only to show how far apart the three bases are; neither is comparable to these
caps, and using 定额 here would understate the fleet's claim on the cap roughly fivefold.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from _bootstrap import ROOT

from coal_retrofit.builders.water import _assign_basin_codes
from coal_retrofit.builders.water_quota import (
    basin_caps,
    basin_reserved_withdrawal,
    modelled_industry_withdrawal,
    calibrated_withdrawal_intensities,
    province_basin_shares,
)
from coal_retrofit.constants_water_quota import (
    BASIN_NAMES_ZH,
    BASIN_WITHDRAWAL_2025_1E8_M3,
    YELLOW_RIVER_CONSUMPTION_CAP_1E8_M3,
)
from coal_retrofit.optimization.scenario import OptimizationAssumptions
from coal_retrofit.paths import ProjectPaths

logger = logging.getLogger(__name__)
CAP_YEAR = 2030


def fleet_water_by_basin(paths: ProjectPaths) -> tuple[pd.DataFrame, float]:
    """Current coal fleet water use by basin, 亿 m3/yr, on every basis. Returns (table, factor)."""
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    assumptions = OptimizationAssumptions()
    hours = plants["province_mode"].astype(str).map(assumptions.province_operating_hours)
    missing = int(hours.isna().sum())
    if missing:
        logger.warning("%d hubs have no province operating hours; using capacity_factor %.2f",
                       missing, assumptions.capacity_factor)
    hours = hours.fillna(assumptions.capacity_factor * 8760.0)
    generation = plants["total_capacity_mw"].astype(float) * hours

    base, _, _, _, factor = calibrated_withdrawal_intensities(plants, generation)

    located = plants.rename(columns={"centroid_latitude": "latitude",
                                     "centroid_longitude": "longitude"})
    plants["basin_code"] = _assign_basin_codes(paths, located)
    plants["withdrawal_calibrated"] = generation * base / 1e8
    plants["withdrawal_raw"] = generation * plants["withdrawal_intensity_m3_per_mwh"].astype(float) / 1e8
    plants["quota"] = generation * plants["quota_intensity_m3_per_mwh"].astype(float) / 1e8
    plants["consumption"] = generation * plants["consumption_intensity_m3_per_mwh"].astype(float) / 1e8
    columns = ["withdrawal_calibrated", "withdrawal_raw", "quota", "consumption"]
    return plants.groupby("basin_code")[columns].sum(), factor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = ProjectPaths(ROOT)

    caps = basin_caps(paths, CAP_YEAR)
    caps_area = basin_caps(paths, CAP_YEAR, kind="area")
    reserved = basin_reserved_withdrawal()
    reserved_loose = basin_reserved_withdrawal(exclude_all_industry=True)
    carve_out = modelled_industry_withdrawal(paths)
    fleet, factor = fleet_water_by_basin(paths)

    print(f"\n直流取水标定系数 k = {factor:.3f}"
          f"  (模型原始 {fleet['withdrawal_raw'].sum():.0f} -> 标定后 "
          f"{fleet['withdrawal_calibrated'].sum():.0f} 亿m3，公报直流火核电 453.8)")
    print(f"\n官方指标口径流域水预算  指标年 {CAP_YEAR}  单位 亿 m3/yr  取水(用水量)口径")
    print(f"{'流域':<11}{'指标':>7}{'面积权重':>9}{'现状':>7}{'预留':>7}{'余量':>7}"
          f"{'工业扣回':>9}{'模型余量':>9}{'煤电':>7}{'占模型余量':>11}{'宽口径余量':>11}")
    print("-" * 96)
    totals = dict.fromkeys(("cap", "area", "actual", "reserved", "left", "loose", "fleet",
                            "carve", "model"), 0.0)
    for code in sorted(BASIN_NAMES_ZH):
        cap, actual = caps[code], BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"]
        left, loose = cap - reserved[code], caps[code] - reserved_loose[code]
        industry = carve_out.get(code, 0.0)
        # 与 write_basin_caps 逐字相同的式子——这一列才是求解器看到的右端项。
        ratio = min(1.0, cap / actual) if actual > 0 else 1.0
        model_left = cap - reserved[code] * ratio + industry * ratio
        used = float(fleet["withdrawal_calibrated"].get(code, 0.0))
        pct = f"{used / model_left * 100:.0f}%" if model_left > 0 else "超限"
        print(f"{code + ' ' + BASIN_NAMES_ZH[code]:<11}{cap:>7.0f}{caps_area[code]:>9.0f}"
              f"{actual:>7.0f}{reserved[code]:>7.0f}{left:>7.0f}{industry:>9.1f}"
              f"{model_left:>9.1f}{used:>7.1f}{pct:>11}{loose:>11.0f}")
        for key, value in (("cap", cap), ("area", caps_area[code]), ("actual", actual),
                           ("reserved", reserved[code]), ("left", left), ("loose", loose),
                           ("fleet", used), ("carve", industry), ("model", model_left)):
            totals[key] += value
    print("-" * 96)
    print(f"{'全国':<11}{totals['cap']:>7.0f}{totals['area']:>9.0f}{totals['actual']:>7.0f}"
          f"{totals['reserved']:>7.0f}{totals['left']:>7.0f}{totals['carve']:>9.1f}"
          f"{totals['model']:>9.1f}{totals['fleet']:>7.1f}"
          f"{totals['fleet'] / totals['model'] * 100:>10.0f}%{totals['loose']:>11.0f}")
    print("  『余量』= 指标 − 预留，是口径说明用的中间量。")
    print("  『模型余量』= 指标 − 预留x折减 + 已建模工业x折减，**这一列才是流域约束的右端项**")
    print("    （builders/water_quota.write_basin_caps，写入 inputs/water_basin_caps.csv）。")
    _tight = [c for c in BASIN_NAMES_ZH
              if (caps[c] - reserved[c] * min(1.0, caps[c] / BASIN_WITHDRAWAL_2025_1E8_M3[c]["total"])
                  + carve_out.get(c, 0.0) * min(1.0, caps[c] / BASIN_WITHDRAWAL_2025_1E8_M3[c]["total"]))
              < carve_out.get(c, 0.0)]
    for c in _tight:
        print(f"  ⚠ {c} {BASIN_NAMES_ZH[c]}：已建模工业的扣回量 {carve_out.get(c, 0.0):.1f} 大于"
              f"模型余量本身，也就是说煤电被约束到的那点空间，几乎全部是尚未进入求解器的"
              f"工业点源当前占用的取水。工业成为决策变量后，两者要在同一个额度里竞争。")

    print("\n煤电四口径对照（亿 m3/yr），说明为何只有标定取水可与上表相比:")
    print(f"{'流域':<11}{'标定取水':>10}{'原始取水':>10}{'定额':>8}{'耗水':>8}")
    print("-" * 49)
    for code in sorted(BASIN_NAMES_ZH):
        row = [float(fleet[c].get(code, 0.0)) for c in
               ("withdrawal_calibrated", "withdrawal_raw", "quota", "consumption")]
        print(f"{code + ' ' + BASIN_NAMES_ZH[code]:<11}{row[0]:>10.1f}{row[1]:>10.1f}"
              f"{row[2]:>8.1f}{row[3]:>8.1f}")
    print("-" * 49)
    print(f"{'全国':<11}{fleet['withdrawal_calibrated'].sum():>10.1f}"
          f"{fleet['withdrawal_raw'].sum():>10.1f}{fleet['quota'].sum():>8.1f}"
          f"{fleet['consumption'].sum():>8.1f}")

    print(f"\n黄河交叉校验: 指标分摊 {caps['D']:.0f} (取水)  vs  八七分水现行 "
          f"{YELLOW_RIVER_CONSUMPTION_CAP_1E8_M3:.1f} (耗水) —— 口径不同，仅量级参考")
    _, unassigned = province_basin_shares(paths)
    print(f"落在流域多边形外的省内权重最大份额 {max(unassigned) * 100:.1f}%（上海，已按比例"
          f"重分配；该省 100% 属长江区，无实质影响）")


if __name__ == "__main__":
    sys.exit(main())
