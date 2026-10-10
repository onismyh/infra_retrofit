"""煤电逐 hub 毛热耗（2026-10-02）：分档查表、按装机加权到 hub、全取旧的全国值时与旧公式相同；另有 CCS 额外燃料比。
不需要 Gurobi。

旧公式（2026-10-02 前）：全国一个热耗 8.5714 GJ/MWh、排放强度 0.82 t/MWh、效率 0.42，整个 hub 到期后热耗乘
0.42/0.45。8.5714 是 3.6/0.42 取四位，现在的效率由热耗反推（3.6/8.5714 = 0.4200014），重建热耗取
min(热耗, 3.6/0.45 = 8.0)（旧 8.5714 x 0.42/0.45 = 7.99997）：含效率或重建热耗的项与旧公式差 3.3e-6（相对），其余相等；
只有一处是有意的改动：重建后生物质与空冷惩罚的燃料按重建机组自己的效率 0.45 折算，旧公式按 hub 的 0.42。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.builders.plants import (
    _combustion_class,
    _supply_coal_rate_g_per_kwh,
    build_plant_dataframe,
    is_cfb,
    mark_named_cfb,
    unit_heat_rate_gj_per_mwh,
    unit_rebuild_heat_rate_cap_gj_per_mwh,
)
from coal_retrofit.constants import COAL_STATION_SERVICE_RATE, STANDARD_COAL_GJ_PER_KG
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from toy_inputs import TOY_HEAT_RATE_GJ_PER_MWH, _write_toy_inputs

GROSS = (1.0 - COAL_STATION_SERVICE_RATE) * STANDARD_COAL_GJ_PER_KG  # g/kWh 供电 -> GJ/MWh 毛


@pytest.mark.parametrize(
    ("combustion", "capacity_mw", "cooling", "expected"),
    [
        ("ultra-supercritical", 1000.0, "recirculating", 285.0),
        ("ultra-supercritical", 900.0, "recirculating", 285.0),  # 1000MW 级下限 900 MW
        ("ultra-supercritical", 660.0, "recirculating", 293.0),
        ("supercritical", 450.0, "recirculating", 300.0),  # 600MW 级下限 450 MW
        ("supercritical", 350.0, "recirculating", 308.0),
        ("subcritical", 600.0, "recirculating", 314.0),
        ("subcritical", 300.0, "recirculating", 323.0),
        ("cfb", 300.0, "recirculating", 290.0),
        ("igcc", 250.0, "recirculating", 270.0),
        ("supercritical", 600.0, "air", 315.0),  # 空冷 +15
        ("Ultra-Supercritical/CCS", 1000.0, "recirculating", 285.0),  # /CCS 按本体机型
        ("ultra-supercritical/CFB", 660.0, "recirculating", 293.0),  # /CFB 同样按本体机型
    ],
)
def test_supply_coal_rate_follows_the_class_table(combustion, capacity_mw, cooling, expected) -> None:
    assert _supply_coal_rate_g_per_kwh(combustion, capacity_mw, cooling) == expected


def test_named_cfb_units_get_a_cfb_mark_and_keep_their_steam_class() -> None:
    """厂名写明 CFB 的机组（`GEM_NAMED_CFB_UNIT_IDS`）加 `/CFB`（2026-10-07 起）：`is_cfb` 认它，毛热耗与用水强度的机型不变；
    不在名单上的、已是 CFB 的不动，重复调用不再加，入参不变。`is_cfb` 对缺失值、空串为否。"""
    units = pd.DataFrame({
        "unit_id": ["G100000115436", "G100000107349", "G999", "G100000115465"],
        "combustion": ["ultra-supercritical", "subcritical", "subcritical", "CFB"],
        "capacity_mw": [700.0, 300.0, 300.0, 660.0],
        "cooling_technology": ["recirculating"] * 4,
    })
    original = units.copy()
    marked = mark_named_cfb(units)
    pd.testing.assert_frame_equal(units, original)
    assert marked.tolist() == ["ultra-supercritical/CFB", "subcritical/CFB", "subcritical", "CFB"]
    assert mark_named_cfb(units.assign(combustion=marked)).tolist() == marked.tolist()
    assert is_cfb(marked).tolist() == [True, True, False, True]
    labels = pd.Series(["CFB/CCS", " cfb", "supercritical/CCS", "supercritical/CFB/CCS", "", None, float("nan")])
    assert is_cfb(labels).tolist() == [True, True, False, True, False, False, False]
    assert is_cfb(pd.Series([None, float("nan"), pd.NA], dtype=object)).tolist() == [False, False, False]
    assert is_cfb(pd.Series([float("nan")])).tolist() == [False]
    pd.testing.assert_series_equal(
        unit_heat_rate_gj_per_mwh(units.assign(combustion=marked)), unit_heat_rate_gj_per_mwh(units)
    )
    assert [_combustion_class(label) for label in marked] == [_combustion_class(label) for label in units["combustion"]]


def test_unknown_combustion_label_raises() -> None:
    with pytest.raises(ValueError, match="no supply coal rate"):
        _supply_coal_rate_g_per_kwh("unknown", 600.0, "recirculating")


def test_unit_heat_rate_is_the_supply_rate_grossed_down() -> None:
    """毛热耗 = 供电煤耗 x (1 - 厂用电率) x 标准煤热值；冷却方式按 GEM 标签归类（dry -> 空冷）。"""
    units = pd.DataFrame({
        "combustion": ["supercritical", "supercritical"], "capacity_mw": [600.0, 600.0],
        "cooling_technology": ["recirculating", "dry"],
    })
    np.testing.assert_allclose(unit_heat_rate_gj_per_mwh(units), [300.0 * GROSS, 315.0 * GROSS], rtol=1e-12)
    assert unit_heat_rate_gj_per_mwh(units).iloc[0] == pytest.approx(8.352666, abs=1e-6)  # 300 x 0.95 x 0.0293076


def test_rebuild_cap_keeps_the_air_cooling_adder() -> None:
    """原址重建后毛热耗的上限：湿冷 3.6 / 0.45 = 8.0，空冷加同一个 +15 g/kWh（15 x 0.95 x 0.0293076 = 0.418）。重建取
    min(原机组, 上限)：湿冷亚临界 300MW 级（8.99）重建为 8.0，空冷的（338 g/kWh，9.41）为 8.418，空冷超超临界 1000MW 级
    （300 g/kWh，8.353）本来就低于上限、不变。"""
    units = pd.DataFrame({
        "combustion": ["subcritical", "subcritical", "ultra-supercritical"], "capacity_mw": [300.0, 300.0, 1000.0],
        "cooling_technology": ["recirculating", "Air", "Air"],
    })
    cap = unit_rebuild_heat_rate_cap_gj_per_mwh(units, 0.45)
    air_cap = 3.6 / 0.45 + 15.0 * GROSS
    np.testing.assert_allclose(cap, [3.6 / 0.45, air_cap, air_cap], rtol=1e-12)
    rebuilt = np.minimum(unit_heat_rate_gj_per_mwh(units), cap)
    np.testing.assert_allclose(rebuilt, [3.6 / 0.45, air_cap, 300.0 * GROSS], rtol=1e-12)


def test_hub_heat_rate_is_weighted_by_capacity_not_by_the_dominant_type() -> None:
    """两台亚临界 300 MW 与一台超超临界 1 000 MW：众数机型是亚临界，hub 热耗按装机加权，偏向超超临界。"""
    units = pd.DataFrame({
        "plant_site": ["A", "A", "A"], "capacity_mw": [300.0, 300.0, 1000.0],
        "combustion": ["subcritical", "subcritical", "ultra-supercritical"],
        "cooling_technology": ["recirculating"] * 3, "commission_year": [1995, 1995, 2015],
        "province": ["Shanxi"] * 3, "latitude": [37.0] * 3, "longitude": [112.0] * 3,
    })
    hub = build_plant_dataframe(units, "toy", "toy").iloc[0]
    assert hub["dominant_combustion"] == "subcritical"
    expected = (2 * 300.0 * 323.0 + 1000.0 * 285.0) / 1600.0 * GROSS
    assert hub["heat_rate_gj_per_mwh"] == pytest.approx(round(expected, 4), abs=1e-12)


SCENARIO = OptimizationScenario(
    experiment_id="TEST-HR", description="toy", planning_years=(2050, 2060), sector_target_source="toy",
)


def _year_data(root, retirement_year: int) -> YearData:
    """toy 输入上 2050 年的系数矩阵。"""
    paths = _write_toy_inputs(root, retirement_year=retirement_year)
    prepared = prepare_inputs(paths, SCENARIO, OptimizationAssumptions())
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _build_year_matrices(prepared, SCENARIO, OptimizationAssumptions(), 2050, state)


@pytest.mark.parametrize("retirement_year", [9999, 2045])
def test_flat_heat_rate_reproduces_the_old_formulas(tmp_path, retirement_year: int) -> None:
    """toy 电厂 hub 热耗取 8.5714 时，各项系数与 2026-10-02 前的公式相同；2045 年到期的那组 2050 年只能重建或退役，
    运行的都是重建机组，生物质与空冷惩罚的燃料按它自己的效率 0.45 折算（旧公式按 hub 的 0.42），空冷背压的排放仍按 hub。"""
    assumptions = OptimizationAssumptions()
    data = _year_data(tmp_path, retirement_year)
    year, old_hr, old_eta = 2050, TOY_HEAT_RATE_GJ_PER_MWH, 0.42
    assert old_hr == 8.5714
    eff_ratio = old_eta / 0.45 if year >= retirement_year else 1.0
    own_eta = 0.45 if year >= retirement_year else old_eta
    hr_eff = old_hr * eff_ratio
    ef_gj = 0.82 / old_hr
    price = assumptions.province_coal_cost("Shanxi")
    capture = SCENARIO.capture_rate
    gen = data.generation
    gen_by_path = data.generation_by_pathway
    basis = gen_by_path.copy()
    basis[:, PATHWAY_INDEX["retire"]] = gen
    ccs_cols = np.zeros(gen_by_path.shape[1])
    ccs_cols[[PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]]] = 1.0
    ratio = assumptions.ccs_energy_penalty_ratio(year)
    air_ratio = assumptions.air_retrofit_efficiency_penalty_pp / old_eta
    air_gross = gen_by_path * 0.82 / 1e6 * air_ratio
    air_gross[:, PATHWAY_INDEX["retire"]] = 0.0
    air_cost = basis * assumptions.air_retrofit_efficiency_penalty_pp / own_eta * hr_eff * price
    air_cost[:, PATHWAY_INDEX["retire"]] = 0.0
    life_left = max(0, retirement_year - year)
    # 基线净运行成本的旧公式；2026-10-10 起进目标的是它减去未改造列（参照，增量口径）。
    net = gen_by_path * (hr_eff * price + assumptions.baseline_om_cost_cny_per_mwh - SCENARIO.electricity_price_for_year(year))
    # 名称: (新, 旧, rtol)。只含热耗与排放因子的项相等；含效率或重建热耗的项差 3.3e-6。基线净运行成本是燃料 + 运维 − 售电，
    # 燃料项的差按 燃料 / |净值| 放大。
    loose = 1e-5 if year >= retirement_year else 1e-12
    margin = hr_eff * price + assumptions.baseline_om_cost_cny_per_mwh - SCENARIO.electricity_price_for_year(year)
    loose_net = loose * hr_eff * price / abs(margin)
    pairs = {
        "emissions_mt": (data.emissions_mt, gen * 0.82 / 1e6, 1e-12),
        "heat_rate_eff": (data.heat_rate_eff, np.full(1, hr_eff), loose),
        "emissions_operating_mt": (data.emissions_operating_mt, gen * 0.82 / 1e6 * eff_ratio, loose),
        "energy_penalty_matrix": (data.energy_penalty_matrix, basis * ratio * hr_eff * price * ccs_cols, loose),
        "ccs_penalty_emissions_matrix": (
            data.ccs_penalty_emissions_matrix, basis * ratio * hr_eff * ef_gj * (1 - capture) / 1e6 * ccs_cols, loose,
        ),
        "biomass_penalty_coeff_per_level": (
            data.biomass_penalty_coeff_per_level,
            np.full(1, assumptions.biomass_efficiency_penalty_per_ratio / own_eta * hr_eff * price), 1e-5,
        ),
        "air_penalty_emissions_matrix": (data.air_penalty_emissions_matrix, air_gross * (1 - ccs_cols * capture), 1e-5),
        "air_penalty_cost_matrix": (data.air_penalty_cost_matrix, air_cost, 1e-5),
        "baseline_net_matrix": (data.baseline_net_matrix, net - net[:, [PATHWAY_INDEX["unabated"]]], loose_net),
        "baseline_reference_cny": (data.baseline_reference_cny, net[:, PATHWAY_INDEX["unabated"]], loose_net),
        "stranded_per_plant": (
            data.stranded_per_plant,
            np.full(1, 1000.0 * assumptions.stranded_asset_base_cny_per_kw * 1000.0 * min(1.0, life_left / 20)),
            1e-12,
        ),
    }
    for name, (new, old, rtol) in pairs.items():
        np.testing.assert_allclose(new, old, rtol=rtol, atol=0.0, err_msg=name)


def test_plants_without_a_positive_heat_rate_are_rejected(tmp_path) -> None:
    """plants.csv 缺毛热耗列或有非正值时读入报错（2026-10-02 前的 plants.csv 没有这一列）。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    for bad in (plants.drop(columns="heat_rate_gj_per_mwh"), plants.assign(heat_rate_gj_per_mwh=0.0)):
        bad.to_csv(paths.inputs_dir / "plants.csv", index=False)
        with pytest.raises(ValueError, match="lacks positive heat_rate_gj_per_mwh"):
            prepare_inputs(paths, SCENARIO, OptimizationAssumptions())


def test_hub_heat_rate_drives_emissions_fuel_and_penalties(tmp_path) -> None:
    """hub 热耗取 9.0：排放 = 发电量 x 9.0 x 排放因子，效率 = 3.6 / 9.0 = 0.4，生物质与空冷惩罚按它折燃料。"""
    assumptions = OptimizationAssumptions()
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    plants["heat_rate_gj_per_mwh"] = 9.0
    plants.to_csv(paths.inputs_dir / "plants.csv", index=False)
    prepared = prepare_inputs(paths, SCENARIO, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    data = _build_year_matrices(prepared, SCENARIO, assumptions, 2050, state)
    price = assumptions.province_coal_cost("Shanxi")
    unabated = PATHWAY_INDEX["unabated"]
    assert data.emissions_mt[0] == pytest.approx(
        data.generation[0] * 9.0 * assumptions.coal_emission_factor_t_per_gj / 1e6
    )
    assert data.heat_rate_eff[0] == 9.0
    assert data.biomass_penalty_coeff_per_level[0] == pytest.approx(
        assumptions.biomass_efficiency_penalty_per_ratio / 0.4 * 9.0 * price
    )
    air_ratio = assumptions.air_retrofit_efficiency_penalty_pp / 0.4
    assert data.air_penalty_cost_matrix[0, unabated] == pytest.approx(data.generation[0] * air_ratio * 9.0 * price)
    assert data.air_penalty_emissions_matrix[0, unabated] == pytest.approx(
        data.generation[0] * air_ratio * 9.0 * assumptions.coal_emission_factor_t_per_gj / 1e6
    )


def test_ccs_extra_fuel_is_p_over_one_minus_p() -> None:
    """CCS 额外燃料比 = p / (1 - p)，p 是 An et al. 2025 SI Table 7 的煤电能耗惩罚（出力损失）：净出力不变要多烧的煤。"""
    p = np.array([0.222, 0.156, 0.133, 0.111])
    np.testing.assert_allclose(OptimizationAssumptions().ccs_energy_penalty_ratio_by_year, p / (1 - p), atol=5e-5)
