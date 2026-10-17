"""流域取水余量的预留与两项扣出（`builders/water_quota.write_basin_caps`、`builders/water_quota_carveout`），不读 ISIMIP。

公报的工业用水里含闭式循环与空冷煤电的取水（只有直流火(核)电单列）。模型的煤电取水是决策变量，所以这份水要和
已建模工业一样从预留里扣出、交还给模型；2026-10-16 前只扣工业，煤电的这份水在预留里与约束里各计一次（PR #68 改）。
2026-10-17 起两项扣出都只计公报年在运的机组与点源，煤电的利用小时按 hub 所在省取（与求解侧同法）。
余量 = cap − (reserved − industry − coal) × ratio，ratio = min(1, cap / actual)。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import coal_retrofit.builders.water_grid as water_grid
import coal_retrofit.builders.water_quota as water_quota
from coal_retrofit.builders.water_quota_carveout import modelled_coal_closed_loop_withdrawal, modelled_industry_withdrawal
from coal_retrofit.constants import PLANNING_YEARS, WATER_INTENSITY_BY_TECH_M3_PER_MWH
from coal_retrofit.constants_water_quota import BASIN_NAMES_ZH, BASIN_WITHDRAWAL_2025_1E8_M3, BULLETIN_YEAR
from coal_retrofit.optimization.assumptions import OptimizationAssumptions
from coal_retrofit.optimization.data_prep_plants import _prepare_plants
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.paths import ProjectPaths
from toy_inputs import _write_toy_inputs

REPO = Path(__file__).resolve().parents[1]
COAL_COLUMN = "modelled_coal_closed_loop_1e8_m3"
WITHDRAWAL = {key: entry["withdrawal"] for key, entry in WATER_INTENSITY_BY_TECH_M3_PER_MWH.items()}


def _fake_assign_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    """按坐标分流域：东经 100° 以西归 K、以东归 C（toy 的点在 87–88°E 与 112–113°E，纬度 36–44°N）。"""
    lon, lat = nodes["longitude"].to_numpy(float), nodes["latitude"].to_numpy(float)
    assert ((lat > 30.0) & (lat < 50.0)).all(), "latitude/longitude swapped"
    return np.where(lon < 100.0, "K", "C")


def _write_plants(paths: ProjectPaths) -> None:
    """四个 hub、八台机组。P1（新疆，hub 省与其中一台机组的省不同）：在运循环冷却、直流（不计）、公报年当年投产的 CFB；
    P2（Neimenggu，别名 → Inner Mongolia）：空冷与公报年后一年投产的机组（不计）；P3（Shanxi）：带 /CCS 标签的超临界循环冷却；
    P4（表外省 Atlantis，缺省小时）：全部机组在公报年之后（扣出为 0）。"""
    paths.inputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "plant_id": ["P1", "P2", "P3", "P4"],
        "province_mode": ["Xinjiang", "Neimenggu", "Shanxi", "Atlantis"],
        "total_capacity_mw": [1900.0, 900.0, 800.0, 600.0],
        "centroid_longitude": [87.0, 88.0, 112.0, 113.0],
        "centroid_latitude": [44.0, 43.0, 37.0, 36.0],
    }).to_csv(paths.inputs_dir / "plants.csv", index=False)
    pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(1, 9)],
        "plant_id": ["P1", "P1", "P1", "P2", "P2", "P3", "P4", "P4"],
        "capacity_mw": [1000.0, 600.0, 300.0, 600.0, 300.0, 800.0, 300.0, 300.0],
        "commission_year": [2010, 2012, BULLETIN_YEAR, 2015, BULLETIN_YEAR + 1, 2018, BULLETIN_YEAR + 1, BULLETIN_YEAR + 3],
        "combustion": ["supercritical", "subcritical", "CFB", "subcritical", "ultra-supercritical",
                       "supercritical/CCS", "subcritical", "subcritical"],
        "cooling_technology": ["recirculating", "once-through", "recirculating", "Air", "recirculating",
                               "recirculating", "recirculating", "recirculating"],
        # u1 的机组省（Gansu）与 hub 省（Xinjiang）不同：小时按 hub 省取。
        "province": ["Gansu", "Xinjiang", "Xinjiang", "Inner Mongolia", "Inner Mongolia", "Shanxi", "Atlantis", "Atlantis"],
    }).to_csv(paths.inputs_dir / "plants_unit_hub.csv", index=False)


def test_coal_carve_out_counts_closed_loop_and_air_units_built_by_the_bulletin_year(tmp_path, monkeypatch, caplog) -> None:
    """只数公报年及以前投产、非直流冷却的机组；强度查 Wang (2023) 表（机型标签经 combustion_class）；发电量按 hub 所在省的
    现状利用小时（别名与表外省与求解侧同法）；按 hub 质心归流域。"""
    paths = ProjectPaths(tmp_path)
    _write_plants(paths)
    monkeypatch.setattr(water_grid, "assign_basin_codes", _fake_assign_basin_codes)
    assumptions = OptimizationAssumptions()
    h_xj, h_nm, h_sx = (assumptions.province_hours(p) for p in ("Xinjiang", "Inner Mongolia", "Shanxi"))
    assert h_xj != assumptions.province_hours("Gansu")
    with caplog.at_level(logging.WARNING):
        result = modelled_coal_closed_loop_withdrawal(paths)
    expected_k = (
        1000.0 * h_xj * WITHDRAWAL[("supercritical", "recirculating")]     # u1：机组省 Gansu，小时按 hub 省 Xinjiang
        + 300.0 * h_xj * WITHDRAWAL[("subcritical", "recirculating")]      # u3：CFB → subcritical，公报年当年投产按全年计
        + 600.0 * h_nm * WITHDRAWAL[("subcritical", "air")]                # u4：Neimenggu 经别名取 Inner Mongolia 的小时
    ) / 1e8
    expected_c = 800.0 * h_sx * WITHDRAWAL[("supercritical", "recirculating")] / 1e8  # u6：supercritical/CCS 只看本体机型
    assert result == pytest.approx({"K": expected_k, "C": expected_c}, rel=1e-6)
    assert any("Atlantis" in record.getMessage() for record in caplog.records)  # 表外省按缺省小时，告警一次


def test_coal_carve_out_matches_prepare_plants_generation(tmp_path, monkeypatch) -> None:
    """全部机组在运时，扣出 = `_prepare_plants` 的发电量 × plants.csv 的淡水取水强度（取水 − 直流项），含直流项非零的 hub。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    rows = [plants.iloc[0].copy() for _ in range(3)]
    for row, pid, province, capacity, lon in zip(
        rows, ("P1", "P2", "P3"), ("Neimenggu", "Atlantis", "Shanxi"), (1000.0, 600.0, 800.0), (87.0, 88.0, 112.0)
    ):
        row["plant_id"], row["province_mode"], row["total_capacity_mw"], row["centroid_longitude"] = pid, province, capacity, lon
    plants = pd.DataFrame(rows).reset_index(drop=True)
    # plants.csv 的两列按 `plants_water.finalize_water_intensities` 的写法：取水 = 全部机组加权（all），直流项 = all − fresh。
    p2_all = 0.5 * WITHDRAWAL[("subcritical", "once-through")] + 0.5 * WITHDRAWAL[("subcritical", "air")]
    p2_fresh = 0.5 * WITHDRAWAL[("subcritical", "air")]
    plants["withdrawal_intensity_m3_per_mwh"] = [
        WITHDRAWAL[("ultra-supercritical", "recirculating")], round(p2_all, 4), WITHDRAWAL[("subcritical", "recirculating")],
    ]
    plants["once_through_withdrawal_intensity_m3_per_mwh"] = [0.0, round(p2_all - p2_fresh, 4), 0.0]
    plants.to_csv(paths.inputs_dir / "plants.csv", index=False)
    pd.DataFrame({
        "plant_id": ["P1", "P2", "P2", "P3"], "capacity_mw": [1000.0, 300.0, 300.0, 800.0],
        "commission_year": [2000, 2005, 2008, 2010],
        "combustion": ["ultra-supercritical", "subcritical", "subcritical", "subcritical"],
        "cooling_technology": ["recirculating", "once-through", "Air", "recirculating"],
    }).to_csv(paths.inputs_dir / "plants_unit_hub.csv", index=False)
    monkeypatch.setattr(water_grid, "assign_basin_codes", _fake_assign_basin_codes)
    prepared = _prepare_plants(paths, OptimizationScenario(experiment_id="toy", description="toy"), OptimizationAssumptions())
    result = modelled_coal_closed_loop_withdrawal(paths)
    assert prepared["province_name"].tolist() == ["Inner Mongolia", "Atlantis", "Shanxi"]
    closed_loop = prepared["withdrawal_intensity_m3_per_mwh"] - prepared["once_through_withdrawal_intensity_m3_per_mwh"]
    basins = np.where(prepared["centroid_longitude"].to_numpy(float) < 100.0, "K", "C")
    expected = (prepared["annual_generation_mwh"] * closed_loop).groupby(basins).sum() / 1e8
    assert result == pytest.approx(expected.to_dict(), rel=1e-6)
    assert result["K"] > 0 and result["C"] > 0


@pytest.mark.parametrize(
    "table, column",
    [("plants.csv", c) for c in ("plant_id", "province_mode", "total_capacity_mw", "centroid_longitude", "centroid_latitude")]
    + [("plants_unit_hub.csv", c) for c in ("plant_id", "capacity_mw", "commission_year", "combustion", "cooling_technology")],
)
def test_coal_carve_out_requires_every_needed_column(tmp_path, table, column) -> None:
    paths = ProjectPaths(tmp_path)
    _write_plants(paths)
    pd.read_csv(paths.inputs_dir / table).drop(columns=[column]).to_csv(paths.inputs_dir / table, index=False)
    with pytest.raises(KeyError, match=rf"{table} lacks \['{column}'\].*build_water_basin_caps.py"):
        modelled_coal_closed_loop_withdrawal(paths)


@pytest.mark.parametrize("table", ["plants.csv", "plants_unit_hub.csv"])
def test_coal_carve_out_names_the_missing_table_and_the_rebuild_command(tmp_path, table) -> None:
    paths = ProjectPaths(tmp_path)
    _write_plants(paths)
    (paths.inputs_dir / table).unlink()
    with pytest.raises(FileNotFoundError, match=rf"{table} not found.*build_plant_inputs.py --hubs, then .*build_water_basin_caps.py"):
        modelled_coal_closed_loop_withdrawal(paths)


def _break_units(units: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "nan_year":
        return units.assign(commission_year=units["commission_year"].astype(float).where(units.index != 0))
    if how == "nan_capacity":
        return units.assign(capacity_mw=units["capacity_mw"].where(units.index != 0))
    if how == "extra_hub":
        return pd.concat([units, units.iloc[[0]].assign(plant_id="P9")], ignore_index=True)
    if how == "capacity":
        return units.assign(capacity_mw=units["capacity_mw"] * 1.01)
    raise ValueError(how)


@pytest.mark.parametrize("how", ["nan_year", "nan_capacity", "extra_hub", "capacity"])
def test_coal_carve_out_rejects_a_unit_table_from_another_clustering(tmp_path, how) -> None:
    """空的投产年或装机、多出的 hub、装机对不上：与求解侧 `_read_unit_hub_map` 同一核对，否则扣出静默偏低、表照样写出。"""
    paths = ProjectPaths(tmp_path)
    _write_plants(paths)
    units = pd.read_csv(paths.inputs_dir / "plants_unit_hub.csv")
    _break_units(units, how).to_csv(paths.inputs_dir / "plants_unit_hub.csv", index=False)
    with pytest.raises(ValueError, match=r"does not match plants.csv.*build_plant_inputs.py --hubs"):
        modelled_coal_closed_loop_withdrawal(paths)


def test_industry_carve_out_leaves_out_sources_under_construction(tmp_path, monkeypatch) -> None:
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    pd.DataFrame({
        "source_id": ["S1", "S2", "S3"], "status": ["operating", "construction", "operating pre-retirement"],
        "water_m3_per_year": [1.0e8, 5.0e8, 2.0e8], "longitude": [87.0, 88.0, 112.0], "latitude": [44.0, 43.0, 37.0],
    }).to_csv(paths.inputs_dir / "industry_sources.csv", index=False)
    monkeypatch.setattr(water_grid, "assign_basin_codes", _fake_assign_basin_codes)
    assert modelled_industry_withdrawal(paths) == pytest.approx({"K": 1.0, "C": 2.0})
    assert modelled_industry_withdrawal(ProjectPaths(tmp_path / "empty")) == {}
    pd.read_csv(paths.inputs_dir / "industry_sources.csv").drop(columns=["status"]).to_csv(
        paths.inputs_dir / "industry_sources.csv", index=False
    )
    with pytest.raises(KeyError, match=r"lacks \['status'\].*build_industry_inputs.py, then .*build_water_basin_caps.py"):
        modelled_industry_withdrawal(paths)


def _reserved(code: str) -> float:
    row = BASIN_WITHDRAWAL_2025_1E8_M3[code]
    return row["domestic"] + row["agriculture"] + row["ecology"] + row["industry"] - row["thermal_once_through"]


def _stub(monkeypatch, caps: dict[str, float], industry: dict[str, float], coal: dict[str, float]) -> None:
    monkeypatch.setattr(water_quota, "basin_caps", lambda _paths, _year, _kind="demand": dict(caps))
    monkeypatch.setattr(water_quota, "modelled_industry_withdrawal", lambda _paths: dict(industry))
    monkeypatch.setattr(water_quota, "modelled_coal_closed_loop_withdrawal", lambda _paths: dict(coal))


def _caps_above_actual() -> dict[str, float]:
    return {code: BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"] + 100.0 for code in BASIN_NAMES_ZH}


def test_residual_hands_back_both_modelled_industry_and_coal_at_the_compliance_ratio(tmp_path, monkeypatch) -> None:
    """余量 = cap − (reserved − industry − coal) × ratio。K 用真实分摊指标 641.152（现状 729.0，reserved = actual）；
    F 的指标压到 1500（现状 2049.0、reserved 1654.5 ≠ actual、直流子列 394.4），钉住 ratio 的分母与 reserved 项；其余流域 ratio = 1。"""
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    caps = _caps_above_actual()
    caps.update({"K": 641.152, "F": 1500.0})
    _stub(monkeypatch, caps, {"K": 2.389, "F": 17.6, "C": 13.7}, {"K": 9.352, "F": 11.2, "C": 9.2})
    frame = water_quota.write_basin_caps(paths)
    assert len(frame) == len(PLANNING_YEARS) * len(BASIN_NAMES_ZH)
    first = frame[frame["planning_year"] == PLANNING_YEARS[0]].set_index("basin_code")

    ratio_k = 641.152 / 729.0
    k = first.loc["K"]
    assert k["reserved_1e8_m3"] == pytest.approx(729.0)  # 公报：24.8 + 645.4 + 39.4 + 19.4 − 0
    assert k["compliance_ratio"] == pytest.approx(ratio_k, abs=5e-5)
    assert k[COAL_COLUMN] == pytest.approx(9.352)
    assert k["residual_uncapped_1e8_m3"] == pytest.approx(641.152 - 729.0 + 2.389 + 9.352, abs=1e-3)
    assert k["residual_1e8_m3"] == pytest.approx((2.389 + 9.352) * ratio_k, abs=1e-3)
    assert k["residual_1e8_m3"] == pytest.approx(10.326, abs=1e-3)
    assert k["residual_m3_per_year"] == pytest.approx(round((2.389 + 9.352) * ratio_k * 1e8), abs=1)

    ratio_f = 1500.0 / 2049.0
    f = first.loc["F"]
    assert f["reserved_1e8_m3"] == pytest.approx(1654.5)  # 352.0 + 1041.3 + 88.3 + 567.3 − 394.4
    assert f["compliance_ratio"] == pytest.approx(ratio_f, abs=5e-5)
    assert f["residual_1e8_m3"] == pytest.approx(1500.0 - (1654.5 - 17.6 - 11.2) * ratio_f, abs=1e-3)
    assert f["residual_uncapped_1e8_m3"] == pytest.approx(1500.0 - 1654.5 + 17.6 + 11.2, abs=1e-3)

    c = first.loc["C"]
    assert c["compliance_ratio"] == 1.0
    assert c["residual_1e8_m3"] == pytest.approx(caps["C"] - _reserved("C") + 13.7 + 9.2, abs=1e-3)
    a = first.loc["A"]  # 没有建模用户的流域余量就是 cap − reserved
    assert a[COAL_COLUMN] == 0.0 and a["residual_1e8_m3"] == pytest.approx(caps["A"] - _reserved("A"), abs=1e-3)

    # 2030 年后指标持平：四个规划年相同；m3 列是未取 3 位小数前的余量取整。
    assert frame.groupby("basin_code")["residual_m3_per_year"].nunique().max() == 1
    assert (frame["residual_m3_per_year"] / 1e8 - frame["residual_1e8_m3"]).abs().max() < 1e-3
    destination = paths.inputs_dir / "water_basin_caps.csv"
    assert destination.read_bytes().startswith(b"\xef\xbb\xbf")  # utf-8-sig：Excel 直接打开中文流域名不乱码
    written = pd.read_csv(destination)
    assert list(written.columns) == list(frame.columns) and COAL_COLUMN in written.columns
    assert written["basin_name"].tolist() == frame["basin_name"].tolist()


def test_carve_outs_beyond_the_bulletin_industry_are_rejected(tmp_path, monkeypatch) -> None:
    """两类扣出合计超过公报的工业减直流火(核)电（F：567.3 − 394.4 = 172.9），说明模型的强度或冷却方式与公报对不上：
    报错并点名两项，不写表；恰好相等不报错。"""
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    _stub(monkeypatch, _caps_above_actual(), {"F": 100.0}, {"F": 72.9 + 0.01})
    with pytest.raises(ValueError, match=r"basin F \(长江\): modelled industry 100.00 \+ coal closed-loop 72.91 = 172.91 .* 172.90"):
        water_quota.write_basin_caps(paths)
    assert not (paths.inputs_dir / "water_basin_caps.csv").exists()
    _stub(monkeypatch, _caps_above_actual(), {"F": 100.0}, {"F": 72.9})
    assert water_quota.write_basin_caps(paths).set_index("basin_code").loc["F", COAL_COLUMN].iloc[0] == pytest.approx(72.9)


def test_write_basin_caps_rejects_carve_out_codes_outside_the_nine_basins(tmp_path, monkeypatch) -> None:
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    _stub(monkeypatch, _caps_above_actual(), {}, {"K": 1.0, "Z": 2.0})
    with pytest.raises(ValueError, match=r"\['Z'\]"):
        water_quota.write_basin_caps(paths)


def test_write_basin_caps_treats_a_basin_without_withdrawal_as_fully_compliant(tmp_path, monkeypatch) -> None:
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    caps = _caps_above_actual()
    monkeypatch.setitem(
        BASIN_WITHDRAWAL_2025_1E8_M3, "J",
        {"domestic": 0.0, "industry": 0.0, "thermal_once_through": 0.0, "agriculture": 0.0, "ecology": 0.0, "total": 0.0},
    )
    _stub(monkeypatch, caps, {}, {})
    j = water_quota.write_basin_caps(paths).set_index("basin_code").loc["J"].iloc[0]
    assert j["compliance_ratio"] == 1.0 and j["residual_1e8_m3"] == pytest.approx(caps["J"])


def test_committed_table_satisfies_the_residual_identity_and_the_bulletin_bound() -> None:
    """只读库内表、不碰 data/：36 行；残差恒等式；两项扣出之和不超过公报工业减直流；m3 列与 1e8 列一致；四个规划年相同；
    K 的余量高于 `warn_basin_floor` 核出的下限 2.71 亿 m3（README §0 2026-10-17，输入变了要一起改）。"""
    path = REPO / "_indtree" / "inputs" / "water_basin_caps.csv"
    if not path.exists():
        pytest.skip("需要 _indtree/inputs/water_basin_caps.csv")
    table = pd.read_csv(path)
    assert len(table) == len(PLANNING_YEARS) * len(BASIN_NAMES_ZH)
    ratio = np.minimum(1.0, table["cap_1e8_m3"] / table["actual_2025_1e8_m3"])
    residual = table["cap_1e8_m3"] - (table["reserved_1e8_m3"] - table["modelled_industry_1e8_m3"] - table[COAL_COLUMN]) * ratio
    assert (residual - table["residual_1e8_m3"]).abs().max() < 2e-3
    assert (table["residual_m3_per_year"] / 1e8 - table["residual_1e8_m3"]).abs().max() < 1e-3
    for code, rows in table.groupby("basin_code"):
        bulletin = BASIN_WITHDRAWAL_2025_1E8_M3[str(code)]
        assert rows["reserved_1e8_m3"].iloc[0] == pytest.approx(_reserved(str(code)), abs=1e-3)
        modelled = rows["modelled_industry_1e8_m3"].iloc[0] + rows[COAL_COLUMN].iloc[0]
        assert modelled <= bulletin["industry"] - bulletin["thermal_once_through"] + 1e-6, code
        assert rows["residual_m3_per_year"].nunique() == 1, code
    assert table.loc[table["basin_code"] == "K", "residual_m3_per_year"].iloc[0] >= 2.71e8


def test_committed_table_matches_the_carve_outs_recomputed_from_the_inputs(tmp_path) -> None:
    """库内表的两列扣出与当前 `plants.csv`、`plants_unit_hub.csv`、`industry_sources.csv` 及缺省利用小时表一致：
    改了这些输入或小时表而没重跑 `scripts/build_water_basin_caps.py`，这里会先于求解发现。"""
    inputs = REPO / "_indtree" / "inputs"
    basins = REPO / "data" / "ChinaBasins" / "basin_l1.gpkg"
    if not (inputs / "water_basin_caps.csv").exists() or not basins.exists():
        pytest.skip("需要 _indtree/inputs 与 data/ChinaBasins")
    root = tmp_path / "tree"
    root.mkdir()
    try:
        os.symlink(inputs, root / "inputs", target_is_directory=True)
        os.symlink(REPO / "data", root / "data", target_is_directory=True)
    except OSError as error:  # Windows 无符号链接权限
        pytest.skip(f"建不了符号链接：{error}")
    paths = ProjectPaths(root)
    table = pd.read_csv(inputs / "water_basin_caps.csv")
    first = table[table["planning_year"] == table["planning_year"].min()].set_index("basin_code")
    coal = modelled_coal_closed_loop_withdrawal(paths)
    industry = modelled_industry_withdrawal(paths)
    for code in first.index:
        assert first.loc[code, COAL_COLUMN] == pytest.approx(coal.get(code, 0.0), abs=6e-4), code
        assert first.loc[code, "modelled_industry_1e8_m3"] == pytest.approx(industry.get(code, 0.0), abs=6e-4), code
