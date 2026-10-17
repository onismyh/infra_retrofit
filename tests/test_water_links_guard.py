"""水链路、流域取水下限与流域余量表版本的三道前置检查（2026-10-16）。

1. 厂侧水平衡在所有情景里都建（`model_resources.add_resource_balances`）：半径内没有水节点的 hub 耗水被迫为 0、
   只剩退役可行，且不报错。`data_prep._prepare_water` 现在直接拒绝这样的输入。
2. 流域余量低于"工业最少取水 + 煤电不退役的最低取水"时，硬约束只能靠退役该流域的煤电满足；
   `water_access.warn_basin_floor` 建模前逐流域告警（PR #68 之前 `_indtree` 的西北诸河 K 2030 年就是这种情形；现行输入许可空冷时不触发、
   关空冷时 K 2030 年触发；下面的用例用合成数）。煤电的最低取水在许可空冷改造时按全部改空冷的取水强度计，关空冷改造时（`allow_air_cooling_retrofit`
   关，求解侧空冷份额上界为 0）按原冷却的取水强度计：`basin_withdrawal_floor` 的两种口径、告警文案，以及经
   `year_matrices._build_year_matrices` 的接线（`allow_air_cooling` 传没传）都在这里核对。
3. `water_basin_caps.csv` 缺 `modelled_coal_closed_loop_1e8_m3` 列（2026-10-16 前的旧表，余量里煤电闭式循环取水计了两次）时，
   `data_prep._prepare_basin_caps` 报错，不静默读旧表。
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pytest

import coal_retrofit.builders.water_grid as water_grid
import coal_retrofit.builders.water_quota as water_quota
from coal_retrofit.constants import WATER_MATCH_BUFFER_KM
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState
from coal_retrofit.optimization.assumptions import OptimizationAssumptions
from coal_retrofit.optimization.data_prep import _prepare_basin_caps, _prepare_water, prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.optimization.water_access import basin_withdrawal_floor, warn_basin_floor
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from coal_retrofit.paths import ProjectPaths
from toy_inputs import YEARS, _write_targets, _write_toy_inputs

WATER_ACCESS_LOGGER = "coal_retrofit.optimization.water_access"
WATER_SCENARIO_ID = "toy|gcm|ssp126"
COAL_COLUMN = "modelled_coal_closed_loop_1e8_m3"
UNABATED = PATHWAY_INDEX["unabated"]
# 替换掉取水定额表的 toy 取水强度（m3/MWh）：原冷却 2.0，空冷 0.2。
TOY_BASE_WITHDRAWAL, TOY_AIR_WITHDRAWAL = 2.0, 0.2


def test_prepare_water_rejects_a_plant_with_no_water_node_in_range(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    # toy 的水节点在厂旁；加一座 2 000 km 以外（半径 200 km 之外）没有任何节点的厂。
    far = plants.iloc[[0]].copy()
    far["plant_id"] = "P_far"
    far["centroid_longitude"], far["centroid_latitude"] = 90.0, 50.0
    with pytest.raises(ValueError, match=rf"1 个煤电 hub 在 {WATER_MATCH_BUFFER_KM:g} km 内没有水节点（P_far）"):
        _prepare_water(paths, pd.concat([plants, far], ignore_index=True), OptimizationAssumptions())


def test_prepare_water_links_every_plant_in_range(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    _, links, _ = _prepare_water(paths, plants, OptimizationAssumptions())
    assert set(links["plant_id"]) == {"P1"}
    assert int(links["distance_rank"].min()) == 1


def _floor_inputs() -> dict[str, np.ndarray]:
    """三座厂、两个流域（厂 0、1 在流域 0，厂 2 在流域 1）、六条路径（列序同 PATHWAYS）。

    路径列在每座厂上都非零且互不相同，原冷却取水强度与空冷取水强度也各列不同：煤电下限只由 unabated 列决定，
    对全部路径列求和或取错强度矩阵都会改变结果。工业：第 1 个 hub 氢路线可用且取水更少，第 2 个 hub 只有不改造。
    """
    generation = np.arange(1, 19, dtype=float).reshape(3, 6) * 1.0e5
    generation[:, UNABATED] = [1.0e6, 2.0e6, 3.0e6]
    withdrawal = np.arange(1, 19, dtype=float).reshape(3, 6) + 10.0
    withdrawal[:, UNABATED] = [2.0, 3.0, 4.0]
    air = np.arange(1, 19, dtype=float).reshape(3, 6) * 0.01
    air[:, UNABATED] = [0.2, 0.3, 0.5]
    return {
        "generation_by_pathway": generation,
        "withdrawal": withdrawal,
        "air_withdrawal": air,
        "basin_membership": np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "industry_water_m3": np.array([[1.0e6, 1.2e6, 0.6e6], [3.0e6, 3.5e6, 1.0e6]]),
        "industry_route_available": np.array([[True, True, True], [True, False, False]]),
        "industry_basin_membership": np.array([[1.0, 1.0], [0.0, 0.0]]),
    }


def test_basin_floor_sums_coal_all_air_and_industry_least_water_route() -> None:
    # 缺省许可空冷改造：煤电下限 = unabated 列的发电量 x 空冷取水强度，按流域加总（厂 0、1 同属流域 0）。
    coal_floor, industry_floor = basin_withdrawal_floor(**_floor_inputs())
    assert coal_floor.tolist() == pytest.approx([1.0e6 * 0.2 + 2.0e6 * 0.3, 3.0e6 * 0.5])
    assert industry_floor.tolist() == pytest.approx([0.6e6 + 3.0e6, 0.0])


def test_basin_floor_uses_the_current_cooling_intensity_without_air_cooling_retrofit() -> None:
    """关空冷改造时煤电不退役能做到的最低取水是原冷却的取水强度（不是空冷的）；此前一律按空冷强度，关空冷时漏报。工业下限不变。"""
    inputs = _floor_inputs()
    allowed_coal, allowed_industry = basin_withdrawal_floor(**inputs, allow_air_cooling=True)
    coal_floor, industry_floor = basin_withdrawal_floor(**inputs, allow_air_cooling=False)
    assert coal_floor.tolist() == pytest.approx([1.0e6 * 2.0 + 2.0e6 * 3.0, 3.0e6 * 4.0])
    assert (coal_floor > allowed_coal).all()
    assert industry_floor.tolist() == pytest.approx(allowed_industry.tolist())


def test_warn_basin_floor_flags_only_basins_below_the_floor(caplog) -> None:
    codes = ["C", "K"]
    residual = np.array([5.0e8, 2.1e8])
    coal_floor = np.array([1.0e8, 0.98e8])
    industry_floor = np.array([1.5e8, 1.72e8])
    with caplog.at_level(logging.WARNING, logger="coal_retrofit.optimization.water_access"):
        flagged = warn_basin_floor(2030, codes, residual, coal_floor, industry_floor)
    assert flagged == ["K"]
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and "basin K in 2030" in messages[0] and "2.70" in messages[0]
    assert "below industry's floor alone" not in messages[0]
    with caplog.at_level(logging.WARNING, logger="coal_retrofit.optimization.water_access"):
        assert warn_basin_floor(2040, ["K"], np.array([1.0e8]), coal_floor[1:], industry_floor[1:]) == ["K"]
    assert "below industry's floor alone" in caplog.records[-1].getMessage()


def test_warn_basin_floor_names_the_cooling_basis_of_the_coal_floor(caplog) -> None:
    """文案里煤电下限的口径随 `allow_air_cooling` 变（全部改空冷 / 维持原冷却），告警的流域不变。"""
    args = (2030, ["K"], np.array([1.0e8]), np.array([0.9e8]), np.array([0.5e8]))
    with caplog.at_level(logging.WARNING, logger=WATER_ACCESS_LOGGER):
        assert warn_basin_floor(*args) == ["K"]
        assert warn_basin_floor(*args, allow_air_cooling=False) == ["K"]
    allowed, disabled = (record.getMessage() for record in caplog.records)
    assert "coal all air-cooled without retirement 0.90" in allowed
    assert "air-cooling retrofit disabled" not in allowed
    assert "coal at its current cooling (air-cooling retrofit disabled) without retirement 0.90" in disabled
    assert "all air-cooled" not in disabled


def _toy_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    return np.array(["B1"] * len(nodes), dtype=object)


def _toy_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series):
    """(基线, 带捕集, 空冷基线, 空冷带捕集) 取水强度 m3/MWh 与直流冷却标定系数；toy 没有 plants.csv 的取水定额表。"""
    ones = pd.Series(1.0, index=plants.index)
    return ones * TOY_BASE_WITHDRAWAL, ones * 3.2, ones * TOY_AIR_WITHDRAWAL, ones * 1.1, 1.0


def _write_basin_caps(paths: ProjectPaths, residual_m3: float, columns: tuple[str, ...] = (COAL_COLUMN,)) -> None:
    pd.DataFrame(
        [{"basin_code": "B1", "planning_year": year, "residual_m3_per_year": residual_m3, **{c: 0.0 for c in columns}}
         for year in YEARS]
    ).to_csv(paths.inputs_dir / "water_basin_caps.csv", index=False)


def _toy_water_paths(root, residual_m3: float) -> ProjectPaths:
    """toy 输入 + 水节点可用量、生活与灌溉耗水（取 0）、流域余量三张小表；唯一的流域码 B1。"""
    paths = _write_toy_inputs(root, retirement_year=9999)
    _write_targets(paths, {year: 0.5 for year in YEARS})
    pd.DataFrame(
        [{"water_node_id": "W1", "planning_year": year, "scenario_family": "baseline", "scenario_id": WATER_SCENARIO_ID,
          "basin_code": "B1", "available_water_m3_per_year": 4.0e7, "dry_season_water_m3_per_year": 2.0e7, "bias_factor": 1.0}
         for year in YEARS]
    ).to_csv(paths.inputs_dir / "water_availability.csv", index=False)
    pd.DataFrame(
        [{"scenario_id": WATER_SCENARIO_ID, "planning_year": year, "basin_code": "B1",
          "domestic_m3_per_year": 0.0, "irrigation_m3_per_year": 0.0,
          "dry_season_domestic_m3_per_year": 0.0, "dry_season_irrigation_m3_per_year": 0.0}
         for year in YEARS]
    ).to_csv(paths.inputs_dir / "water_basin_use.csv", index=False)
    _write_basin_caps(paths, residual_m3)
    return paths


def _year_data(paths: ProjectPaths, year: int, **assumption_overrides) -> YearData:
    scenario = OptimizationScenario(
        experiment_id="TEST-FLOOR", description="toy", planning_years=YEARS, sector_target_source="toy",
        water_mode="grid_supply", water_scenario_id=WATER_SCENARIO_ID, water_season="dry",
    )
    assumptions = OptimizationAssumptions(**assumption_overrides)
    assert assumptions.apply_basin_cap  # 流域上限缺省开：下限告警只在开着时建
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _build_year_matrices(prepared, scenario, assumptions, year, state)


def _floor_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == WATER_ACCESS_LOGGER]


def _floor_total_m3(data: YearData, allow_air_cooling: bool) -> float:
    assert data.withdrawal_intensity is not None and data.air_withdrawal_intensity is not None
    coal, industry = basin_withdrawal_floor(
        data.generation_by_pathway, data.withdrawal_intensity, data.air_withdrawal_intensity, data.water_basin_membership,
        data.industry.water_m3, data.industry.route_available, data.industry_basin_membership,
        allow_air_cooling=allow_air_cooling,
    )
    assert coal.shape == industry.shape == (1,)  # toy 只有流域 B1
    return float(coal[0] + industry[0])


def test_year_matrices_warn_the_basin_floor_on_the_cooling_basis_the_run_allows(tmp_path, monkeypatch, caplog) -> None:
    """toy：煤电不改造发电量 G，原冷却强度 2.0、空冷 0.2，工业最少取水 1.0e6 m3。许可空冷改造时下限 = 0.2 G + 1.0e6，
    关空冷改造时 = 2.0 G + 1.0e6（G = 4.63e6 MWh，约 1.93e6 对 10.26e6 m3）。余量取三档：低于两者 / 夹在中间 / 高于两者。
    接线漏传 `allow_air_cooling`（取缺省许可）时，夹在中间的余量在关空冷时不告警，文案也不会换口径。"""
    monkeypatch.setattr(water_grid, "assign_basin_codes", _toy_basin_codes)
    monkeypatch.setattr(water_quota, "calibrated_withdrawal_intensities", _toy_withdrawal)
    paths = _toy_water_paths(tmp_path, residual_m3=1.5e6)
    data = _year_data(paths, 2050)
    generation = float(data.generation_by_pathway[0, UNABATED])
    industry_least = float(data.industry.water_m3[0, 0])  # 第 0 列不改造；氢路线列不可用，水泥 hub 取水最少的是它
    floor_air = _floor_total_m3(data, allow_air_cooling=True)
    floor_current = _floor_total_m3(data, allow_air_cooling=False)
    assert floor_air == pytest.approx(TOY_AIR_WITHDRAWAL * generation + industry_least)
    assert floor_current == pytest.approx(TOY_BASE_WITHDRAWAL * generation + industry_least)
    low, middle, high = 1.5e6, 5.0e6, 1.2e7
    assert low < floor_air < middle < floor_current < high

    def warnings(residual_m3: float, year: int = 2050, **assumption_overrides) -> list[str]:
        _write_basin_caps(paths, residual_m3)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=WATER_ACCESS_LOGGER):
            _year_data(paths, year, **assumption_overrides)
        return _floor_warnings(caplog)

    # 余量够：两种口径都不告警。
    assert warnings(high) == []
    assert warnings(high, allow_air_cooling_retrofit=False) == []
    # 低于许可空冷时的下限：告警点名流域码与年份，口径是"全部改空冷"。
    for year in YEARS:
        (message,) = warnings(low, year)
        assert f"basin B1 in {year}" in message and "coal all air-cooled without retirement" in message
        assert "air-cooling retrofit disabled" not in message
        assert f"the floor {floor_air / 1e8:.2f} " in message
    # 同一余量关空冷改造：文案换成原冷却口径，下限更高。
    (disabled,) = warnings(low, allow_air_cooling_retrofit=False)
    assert "basin B1 in 2050" in disabled and "air-cooling retrofit disabled" in disabled and "all air-cooled" not in disabled
    assert f"the floor {floor_current / 1e8:.2f} " in disabled
    assert float(re.search(r"the floor ([0-9.]+) ", disabled).group(1)) > floor_air / 1e8
    # 夹在两个下限之间：许可空冷时够，关空冷时不够（关空冷时漏报的就是这种情形）。
    assert warnings(middle) == []
    (message,) = warnings(middle, allow_air_cooling_retrofit=False)
    assert "basin B1 in 2050" in message and "air-cooling retrofit disabled" in message


def test_prepare_basin_caps_rejects_a_table_without_the_coal_carve_out_column(tmp_path) -> None:
    """2026-10-16 前的表没有 `modelled_coal_closed_loop_1e8_m3`（余量里煤电闭式循环取水计了两次）：有水约束时拒绝读，信息点名该列；
    有这一列时正常读入；无水约束时不读这张表。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    water = OptimizationScenario(experiment_id="TEST-CAPS", description="toy", water_mode="grid_supply")
    no_water = OptimizationScenario(experiment_id="TEST-CAPS", description="toy")
    assert no_water.water_mode == "no_water"
    _write_basin_caps(paths, 6.0e6, columns=())
    with pytest.raises(ValueError, match=rf"water_basin_caps.csv lacks required columns \['{COAL_COLUMN}'\]"):
        _prepare_basin_caps(paths, water)
    assert _prepare_basin_caps(paths, no_water).empty
    _write_basin_caps(paths, 6.0e6)
    caps = _prepare_basin_caps(paths, water)
    assert len(caps) == len(YEARS) and set(caps["basin_code"]) == {"B1"} and COAL_COLUMN in caps.columns
