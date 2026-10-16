"""流域取水余量的预留与扣出（`builders/water_quota.write_basin_caps`，2026-10-16 改）。

公报的工业用水里含闭式循环与空冷煤电的取水（只有直流火(核)电单列）。模型的煤电取水是决策变量，所以这份水要和
已建模工业一样从预留里扣出、交还给模型；2026-10-16 前只扣工业，煤电的这份水在预留里与约束里各计一次。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import coal_retrofit.builders.water_grid as water_grid
import coal_retrofit.builders.water_quota as water_quota
from coal_retrofit.constants import PLANNING_YEARS, WATER_INTENSITY_BY_TECH_M3_PER_MWH
from coal_retrofit.constants_province import PROVINCE_OPERATING_HOURS
from coal_retrofit.constants_water_quota import BASIN_NAMES_ZH, BASIN_WITHDRAWAL_2025_1E8_M3, BULLETIN_YEAR
from coal_retrofit.paths import ProjectPaths


def _paths(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths(tmp_path)
    paths.ensure_inputs_dir()
    return paths


def test_coal_carve_out_counts_closed_loop_and_air_units_built_by_the_bulletin_year(tmp_path, monkeypatch) -> None:
    """只数公报年及以前投产、非直流冷却的机组，发电量按省利用小时，强度查 Wang (2023) 表（不带 CCS），按 hub 归流域。"""
    paths = _paths(tmp_path)
    pd.DataFrame({
        "plant_id": ["P1", "P2"], "centroid_latitude": [37.0, 44.0], "centroid_longitude": [112.0, 87.0],
    }).to_csv(paths.inputs_dir / "plants.csv", index=False)
    pd.DataFrame({
        "unit_id": ["u1", "u2", "u3", "u4"],
        "plant_id": ["P1", "P1", "P1", "P2"],
        "capacity_mw": [600.0, 600.0, 1000.0, 300.0],
        "commission_year": [2010, 2010, BULLETIN_YEAR + 2, BULLETIN_YEAR],
        "combustion": ["subcritical", "supercritical", "ultra-supercritical", "supercritical"],
        "cooling_technology": ["recirculating", "once-through", "Air", "Air"],
        "province": ["Shanxi", "Shanxi", "Shanxi", "Xinjiang"],
    }).to_csv(paths.inputs_dir / "plants_unit_hub.csv", index=False)
    monkeypatch.setattr(water_grid, "assign_basin_codes", lambda _paths, frame: ["C" if pid == "P1" else "K"
                                                                                 for pid in frame["plant_id"]])

    carve_out = water_quota.modelled_coal_closed_loop_withdrawal(paths)

    def withdrawal(capacity_mw: float, province: str, combustion: str, cooling: str) -> float:
        intensity = WATER_INTENSITY_BY_TECH_M3_PER_MWH[(combustion, cooling)]["withdrawal"]
        return capacity_mw * PROVINCE_OPERATING_HOURS[province] * intensity / 1e8

    # u2 是直流冷却（随公报的直流火(核)电子列整列扣出），u3 在公报年之后投产，都不计。
    assert carve_out == pytest.approx({
        "C": withdrawal(600.0, "Shanxi", "subcritical", "recirculating"),
        "K": withdrawal(300.0, "Xinjiang", "supercritical", "air"),
    })


def _stub_carve_outs(monkeypatch, industry: dict[str, float], coal: dict[str, float]) -> dict[str, float]:
    """各流域上限固定为公报实际取水的 1.2 倍（K 流域 0.9 倍，即超指标、预留按比例缩），两类扣出取给定值。"""
    scale = {code: 0.9 if code == "K" else 1.2 for code in BASIN_NAMES_ZH}
    caps = {code: BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"] * scale[code] for code in BASIN_NAMES_ZH}
    monkeypatch.setattr(water_quota, "basin_caps", lambda _paths, _year, _kind="demand": dict(caps))
    monkeypatch.setattr(water_quota, "modelled_industry_withdrawal", lambda _paths: dict(industry))
    monkeypatch.setattr(water_quota, "modelled_coal_closed_loop_withdrawal", lambda _paths: dict(coal))
    return caps


def test_residual_hands_back_both_modelled_industry_and_coal(tmp_path, monkeypatch) -> None:
    """余量 = 上限 − (预留 − 已建模工业 − 煤电闭式循环) × 达标比例；未缩放的余量把两类扣出原样加回。"""
    industry, coal = {"C": 5.0, "K": 1.0}, {"C": 10.0, "K": 3.0}
    caps = _stub_carve_outs(monkeypatch, industry, coal)
    frame = water_quota.write_basin_caps(_paths(tmp_path)).set_index(["planning_year", "basin_code"])
    reserved = water_quota.basin_reserved_withdrawal()
    year = PLANNING_YEARS[0]
    for code in ("C", "K", "J"):
        row = frame.loc[(year, code)]
        actual = BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"]
        ratio = min(1.0, caps[code] / actual)
        handed_back = industry.get(code, 0.0) + coal.get(code, 0.0)
        assert row["compliance_ratio"] == pytest.approx(ratio, abs=1e-4)
        assert row["modelled_coal_closed_loop_1e8_m3"] == pytest.approx(coal.get(code, 0.0))
        assert row["residual_1e8_m3"] == pytest.approx(caps[code] - (reserved[code] - handed_back) * ratio, abs=1e-3)
        assert row["residual_uncapped_1e8_m3"] == pytest.approx(caps[code] - reserved[code] + handed_back, abs=1e-3)
    # K 流域超指标（比例 0.9）：交还的煤电 3 亿 m3 也按比例缩，余量比只扣工业时多 3 × 0.9。
    k = frame.loc[(year, "K")]
    only_industry = caps["K"] - (reserved["K"] - industry["K"]) * 0.9
    assert k["residual_1e8_m3"] - only_industry == pytest.approx(3.0 * 0.9, abs=1e-3)


def test_carve_outs_beyond_the_bulletin_industry_are_rejected(tmp_path, monkeypatch) -> None:
    """两类扣出合计超过公报的工业减直流火(核)电，说明模型的强度或冷却方式与公报对不上：报错，不写表。"""
    closed_loop_industry = (BASIN_WITHDRAWAL_2025_1E8_M3["J"]["industry"]
                            - BASIN_WITHDRAWAL_2025_1E8_M3["J"]["thermal_once_through"])
    _stub_carve_outs(monkeypatch, {"J": 1.0}, {"J": closed_loop_industry})
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="basin J"):
        water_quota.write_basin_caps(paths)
    assert not (paths.inputs_dir / "water_basin_caps.csv").exists()
