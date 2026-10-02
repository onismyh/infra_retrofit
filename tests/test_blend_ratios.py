"""连续 hub 下的掺烧比例换算、明细表的空冷运行份额、逐路径减排拆分与逐厂碳成本。

连续 hub 下一个 hub 可以把不同份额改造到不同档位，`blend_level = Σ l·select` 只是档位下标的加权和：
一半第 1 档、一半第 3 档记作 2，按档位读成 0.25，实际是 0.30。结果表改按约束里的 Σβ_l·z_l 除以
路径份额换算（`results_plant._blend_ratios`）；逐路径的减排量与捕集量按约束逐项拆分（`results_plant._pathway_split`），
逐厂相加等于求解器的值；成本表的碳成本与目标函数同式，用求解器的逐厂减排量。求解 toy 的几条
（只开 BECCS 的生物质用量、独热档位对照、比例列与未截断的商、逐路径拆分与求解器对拍、成本表碳成本与目标函数对拍、
掺氨用量）需要 Gurobi，其余不依赖。
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.optimization._shared import (
    PATHWAY_INDEX,
    PreparedInputs,
    _discount_factor,
    _year_objective_weight,
)
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results import _build_cost_breakdown, _build_sanity_checks
from coal_retrofit.optimization.results_plant import (
    _blend_ratios,
    _build_pathway_table,
    _build_plant_cost_table,
    _build_plant_detail_table,
    _build_province_table,
)
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_types import YearData

SCENARIO = OptimizationScenario(experiment_id="T", description="toy")
LEVELS_B = SCENARIO.biomass_blend_levels  # (0.10, 0.25, 0.50, 0.75, 1.00)
LEVELS_A = SCENARIO.ammonia_blend_levels  # (0.10, 0.20, 0.30, 0.40, 0.50)
BIO, BECCS, AMM = PATHWAY_INDEX["biomass"], PATHWAY_INDEX["beccs"], PATHWAY_INDEX["ammonia"]


def _three_hubs() -> dict[str, np.ndarray]:
    """三个 hub 的份额与 Σβ·z，档位混合各不相同。

    hub 0：全部生物质，改造容量一半第 1 档（0.10）、一半第 3 档（0.50）。档位下标 2.0，比例 0.30。
    hub 1：生物质 0.4 用第 4 档（0.75），BECCS 0.6 用第 2 档（0.25）。档位下标 2.8，两条路径比例不同。
    hub 2：掺氨 0.5 用第 5 档（0.50），其余未改造。档位下标 2.5，比例 0.50。
    """
    share = np.zeros((3, len(PATHWAYS)))
    share[0, BIO] = 1.0
    share[1, BIO], share[1, BECCS] = 0.4, 0.6
    share[2, AMM], share[2, PATHWAY_INDEX["unabated"]] = 0.5, 0.5
    return {
        "share": share,
        "bio_xs": np.array([0.5 * 0.10 + 0.5 * 0.50, 0.4 * 0.75, 0.0]),
        "beccs_xs": np.array([0.0, 0.6 * 0.25, 0.0]),
        "amm_xs": np.array([0.0, 0.0, 0.5 * 0.50]),
        "level_b": np.array([0.5 * 1 + 0.5 * 3, 0.4 * 4 + 0.6 * 2, 0.0]),
        "level_a": np.array([0.0, 0.0, 0.5 * 5]),
    }


def _prepared(n: int) -> PreparedInputs:
    plants = pd.DataFrame({
        "plant_id": [f"P{i}" for i in range(n)], "province_name": ["Shanxi"] * n,
        "total_capacity_mw": [1000.0] * n, "centroid_longitude": [112.0] * n,
        "centroid_latitude": [37.0] * n, "retirement_year": [2060] * n,
    })
    network = SimpleNamespace(
        edges=pd.DataFrame(columns=["from_node_id", "to_node_id", "length_km"]),
        storage_node_ids={}, plant_node_ids={},
    )
    return cast(PreparedInputs, SimpleNamespace(plants=plants, network=network))


def test_effective_ratio_is_sum_beta_z_over_the_pathway_share() -> None:
    hubs = _three_hubs()
    levels = {"biomass_levels": LEVELS_B, "ammonia_levels": LEVELS_A}
    ratios = _blend_ratios(hubs["share"], hubs["bio_xs"], hubs["beccs_xs"], hubs["amm_xs"], **levels)
    np.testing.assert_allclose(ratios["biomass"], [0.30, 0.75, 0.0], rtol=1e-12)
    np.testing.assert_allclose(ratios["beccs"], [0.0, 0.25, 0.0], rtol=1e-12)
    np.testing.assert_allclose(ratios["ammonia"], [0.0, 0.0, 0.50], rtol=1e-12)
    # 份额为零（或在可行性容差之内）的路径记 0，不做 0/0。
    tiny = np.zeros((1, len(PATHWAYS)))
    tiny[0, BIO] = 5e-7
    assert _blend_ratios(tiny, np.array([5e-7]), np.zeros(1), np.zeros(1), **levels)["biomass"][0] == 0.0
    # 份额只比容差略大时商是噪声，可能越出档位范围：截到 [0, 最高档]，BECCS 用生物质的档位表。
    near = np.zeros((1, len(PATHWAYS)))
    near[0, BIO] = near[0, BECCS] = near[0, AMM] = 2e-6
    clipped = _blend_ratios(near, np.array([5e-6]), np.array([-1e-7]), np.array([5e-6]), **levels)
    assert (clipped["biomass"][0], clipped["beccs"][0], clipped["ammonia"][0]) == (max(LEVELS_B), 0.0, max(LEVELS_A))


def test_detail_table_uses_the_effective_ratios() -> None:
    hubs = _three_hubs()
    n = len(hubs["share"])
    zeros = np.zeros(n)
    year_data = cast(YearData, SimpleNamespace(generation=np.full(n, 5.0e6), emissions_mt=np.full(n, 10.0)))

    detail = _build_plant_detail_table(
        _prepared(n), SCENARIO, 2040, hubs["share"], zeros, zeros, zeros, zeros,
        hubs["level_b"], hubs["level_a"], np.zeros((n, len(PATHWAYS))),
        year_data=year_data, plant_reduction_mt=np.array([3.0, 8.0, 2.5]),
        biomass_blend_x_share=hubs["bio_xs"], beccs_blend_x_share=hubs["beccs_xs"],
        ammonia_blend_x_share=hubs["amm_xs"], air_installed=zeros,
    )
    np.testing.assert_allclose(detail["biomass_blend_ratio"], [0.30, 0.75, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["beccs_blend_ratio"], [0.0, 0.25, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["ammonia_blend_ratio"], [0.0, 0.0, 0.50], rtol=1e-12)
    # 档位下标照原样保留（连续 hub 下是加权下标）。
    np.testing.assert_allclose(detail["biomass_blend_level"], hubs["level_b"], rtol=1e-12)


def test_detail_air_operating_share_leaves_out_the_retire_pathway() -> None:
    """hub 2 掺氨 0.5 全转空冷，另有 0.3 的空冷份额落在退役路径上（已装存量以内，对用水与成本都没有作用，模型可任取）：
    运行份额只计前者，已装份额照抄空冷存量。"""
    hubs = _three_hubs()
    n = len(hubs["share"])
    zeros = np.zeros(n)
    share = hubs["share"].copy()
    share[2, PATHWAY_INDEX["unabated"]], share[2, PATHWAY_INDEX["retire"]] = 0.2, 0.3
    air_share = np.zeros((n, len(PATHWAYS)))
    air_share[2, AMM], air_share[2, PATHWAY_INDEX["retire"]] = 0.5, 0.3
    year_data = cast(YearData, SimpleNamespace(generation=np.full(n, 5.0e6), emissions_mt=np.full(n, 10.0)))
    detail = _build_plant_detail_table(
        _prepared(n), SCENARIO, 2040, share, zeros, zeros, zeros, zeros,
        hubs["level_b"], hubs["level_a"], air_share,
        year_data=year_data, plant_reduction_mt=zeros,
        biomass_blend_x_share=hubs["bio_xs"], beccs_blend_x_share=hubs["beccs_xs"],
        ammonia_blend_x_share=hubs["amm_xs"], air_installed=np.array([0.0, 0.0, 0.9]),
    )
    np.testing.assert_allclose(detail["air_operating_share"], [0.0, 0.0, 0.5], rtol=1e-12)
    np.testing.assert_allclose(detail["air_installed_share"], [0.0, 0.0, 0.9], rtol=1e-12)


def test_pathway_split_charges_each_term_to_its_own_pathway() -> None:
    """两个 hub，基线排放各 10 Mt，改造路径的发电量带 CF 提升（按 11.5 Mt 计），η = 0.9。

    hub 0：CCS、掺氨 10% 各一半。CCS 残余 11.5 × 0.1 × 0.5 = 0.575，减排 4.425；掺氨残余 11.5 × 0.9 × 0.5 = 5.175，
    比它的基线份额 5 还多，减排 −0.175，即这条路径净增排。此前按经典比例（0.9、0.1）拆分再缩放到厂合计 4.25，
    掺氨分到 +0.425，符号反了。
    hub 1：生物质 0.4（Σβz = 0.3）、BECCS 0.6（Σβz = 0.15），惩罚项都记在 BECCS 或生物质上：
    生物质残余 11.5 × (0.4 − 0.3) + 掺烧惩罚 0.03 = 1.18；BECCS 残余 11.5 × (0.1 × 0.6 − 0.15) + 掺烧惩罚 0.0075
    + 空冷背压 0.1 × 0.2 + CCS 能耗惩罚 0.05 × 0.6 = −0.9775；BECCS 捕集 11.5 × 0.9 × 0.6 + 0.0675 + 0.9 × 0.2
    + 0.45 × 0.6 = 6.7275。
    """
    n, bio, beccs, ccs = 2, BIO, BECCS, PATHWAY_INDEX["ccs"]
    share = np.zeros((n, len(PATHWAYS)))
    share[0, ccs] = share[0, AMM] = 0.5
    share[1, bio], share[1, beccs] = 0.4, 0.6
    air_share = np.zeros((n, len(PATHWAYS)))
    air_share[1, beccs] = 0.2
    penalty = {name: np.zeros((n, len(PATHWAYS))) for name in ("ccs_em", "ccs_cap", "air_em", "air_cap")}
    penalty["ccs_em"][1, beccs], penalty["ccs_cap"][1, beccs] = 0.05, 0.45
    penalty["air_em"][1, beccs], penalty["air_cap"][1, beccs] = 0.1, 0.9
    year_data = cast(YearData, SimpleNamespace(
        generation=np.full(n, 5.0e6), emissions_mt=np.full(n, 10.0), emissions_operating_mt=np.full(n, 10.0),
        emissions_retrofit_mt=np.full(n, 11.5), generation_by_pathway=np.full((n, len(PATHWAYS)), 5.0e6),
        ccs_penalty_emissions_matrix=penalty["ccs_em"], ccs_penalty_captured_matrix=penalty["ccs_cap"],
        air_penalty_emissions_matrix=penalty["air_em"], air_penalty_captured_matrix=penalty["air_cap"],
        biomass_penalty_emissions_coeff_per_level=np.full(n, 2e-8),
        beccs_penalty_emissions_coeff_per_level=np.full(n, 1e-8),
        beccs_penalty_captured_coeff_per_level=np.full(n, 9e-8),
    ))
    table = _build_pathway_table(
        _prepared(n), SCENARIO, 2050, share, np.array([0.0, 0.3]), np.array([0.0, 0.15]), np.array([0.05, 0.0]),
        year_data=year_data, air_share=air_share,
        rebuilt_share=np.zeros_like(share), rebuilt_blend_x_share=np.zeros_like(share),
    )
    hub0 = table[table["plant_id"] == "P0"].set_index("pathway")
    assert hub0.loc["ccs", "abatement_mt"] == pytest.approx(4.425, rel=1e-9)
    assert hub0.loc["ammonia", "abatement_mt"] == pytest.approx(-0.175, rel=1e-9)
    assert hub0.loc["ccs", "captured_mt"] == pytest.approx(11.5 * 0.9 * 0.5, rel=1e-9)
    hub1 = table[table["plant_id"] == "P1"].set_index("pathway")
    assert hub1.loc["biomass", "abatement_mt"] == pytest.approx(4.0 - 1.18, rel=1e-9)
    assert hub1.loc["beccs", "abatement_mt"] == pytest.approx(6.0 + 0.9775, rel=1e-9)
    assert hub1.loc["beccs", "captured_mt"] == pytest.approx(6.7275, rel=1e-9)
    # 份额为零的路径没有减排也没有捕集；捕集只在 CCS、BECCS 上。
    for hub in (hub0, hub1):
        idle = hub[hub["share"] == 0.0]
        assert (idle["abatement_mt"] == 0.0).all() and (idle["captured_mt"] == 0.0).all()
        assert (hub.drop(index=["ccs", "beccs"])["captured_mt"] == 0.0).all()


def test_plant_cost_carbon_cost_is_the_objective_expression() -> None:
    """碳成本 = 碳价 × 1e6 × (基线排放 − 求解器逐厂减排量)，与 `model_costs._operating_costs` 同式；
    与份额、掺烧档位无关（`year_data` 里故意没有 `emissions_retrofit_mt`：旧的近似式要用它）。"""
    n = 2
    zeros_path = np.zeros((n, len(PATHWAYS)))
    share = np.zeros((n, len(PATHWAYS)))
    share[0, PATHWAY_INDEX["ccs"]] = 1.0
    share[1, PATHWAY_INDEX["unabated"]] = 1.0

    def table(carbon_price: float) -> pd.DataFrame:
        year_data = cast(YearData, SimpleNamespace(
            emissions_mt=np.array([10.0, 4.0]), carbon_price=carbon_price,
            retrofit_stock_capex=np.zeros((n, 1)), baseline_net_matrix=zeros_path,
            biomass_flow_scale=1.0, coal_savings_per_gj=np.zeros(n), fixed_cost_matrix=zeros_path,
            energy_penalty_matrix=zeros_path, air_penalty_cost_matrix=zeros_path, allow_air_cooling_retrofit=False,
        ))
        return _build_plant_cost_table(
            _prepared(n), 2040, year_data, share, np.zeros(n),
            # 第 2 个 hub 的减排为负：惩罚燃料使排放高于基线，碳成本随之高于基线排放的碳价。
            plant_reduction_mt=np.array([7.5, -0.2]),
            retrofit_new=np.zeros((n, 1)), ccs_om_by_plant=np.zeros(n), stranded_by_plant=np.zeros(n),
            capex_pathway_indices=(PATHWAY_INDEX["ccs"],), rebuilt_share=zeros_path,
            air_share=zeros_path, rebuilt_air_share=zeros_path, bio_penalty_by_plant=np.zeros(n),
        )

    priced = table(100.0)
    np.testing.assert_allclose(priced["carbon_cost_cny"], [100.0 * 1e6 * 2.5, 100.0 * 1e6 * 4.2], rtol=1e-12)
    np.testing.assert_allclose(priced["total_plant_cost_cny"], priced["carbon_cost_cny"], rtol=1e-12)
    assert (table(0.0)["carbon_cost_cny"] == 0.0).all()


def test_solved_blend_x_share_is_the_quantity_the_constraints_use(tmp_path) -> None:
    """求解后提取的 Σβ·z 与约束里用的是同一个量。只开放 BECCS、2050 年电力目标为零排放（同
    `test_capex_stock_and_lifetimes` 的 BECCS 用例）：生物质与氨两列为 0；生物质用量约束
    `biomass_use_gj = 热耗 × (G_bio·Σβz_bio + G_beccs·Σβz_beccs)` 成立；有效比例落在档位之内。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    from test_capex_stock_and_lifetimes import _solve_toy
    from toy_inputs import _write_targets, _write_toy_inputs

    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 0.0, 2060: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-BLEND", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire", "ccs", "biomass", "ammonia"),
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
    y50 = _solve_toy(paths, scenario, assumptions)["year_solutions"][2050]

    bio_xs, beccs_xs = float(y50["biomass_blend_x_share"][0]), float(y50["beccs_blend_x_share"][0])
    assert bio_xs == pytest.approx(0.0, abs=1e-9)
    assert float(y50["ammonia_blend_x_share"][0]) == pytest.approx(0.0, abs=1e-9)
    assert beccs_xs > 0.0
    gen = y50["year_data"].generation_by_pathway[0]
    expected_gj = float(y50["year_data"].heat_rate_eff[0]) * (gen[BIO] * bio_xs + gen[BECCS] * beccs_xs)
    assert float(y50["biomass_use_gj"][0]) == pytest.approx(expected_gj, rel=1e-6)
    # 用未截断的商核对：落在档位之内要靠约束本身，不靠 `_blend_ratios` 的截断。
    raw = beccs_xs / float(y50["share"][0, BECCS])
    assert min(LEVELS_B) - 1e-6 <= raw <= max(LEVELS_B) + 1e-6
    ratios = _blend_ratios(
        y50["share"], y50["biomass_blend_x_share"], y50["beccs_blend_x_share"], y50["ammonia_blend_x_share"],
        biomass_levels=LEVELS_B, ammonia_levels=LEVELS_A,
    )
    assert float(ratios["beccs"][0]) == pytest.approx(raw, abs=1e-6)


# 只开放一类掺烧，其余改造路径关掉：BECCS 用生物质的档位表，掺氨用氨的档位表。
_ONLY = {
    "beccs": ("retire", "ccs", "biomass", "ammonia"),
    "ammonia": ("retire", "ccs", "biomass", "beccs"),
}


def _solve_blend_toy(
    root, pathway_disable, power_caps, *, continuous=True, carbon=(0.0, 0.0), coal=None, mip_gap=None, units=None
):
    """求解 2050、2060 两年的 toy，生物质与氨的供给挪到电厂旁边且充足。

    `coal` 改 toy 电厂所在省（山西）的煤价，元/GJ；`mip_gap` 缺省用情景的缺省值；`units` 是 ((装机, 投产年), ...)，
    换掉 toy 的机组表（装机合计仍 1 000 MW，机型同 toy 机组，hub 毛热耗随之取机组的），缺省一台机组、两年都不到期。
    返回 (scenario, assumptions, prepared, solution)；`prepared` 供结果表函数用。
    """
    from test_capex_stock_and_lifetimes import _solve_toy
    from toy_inputs import TOY_UNIT_TYPE, _write_targets, _write_toy_inputs, _write_toy_units

    paths = _write_toy_inputs(root, retirement_year=9999)
    if units is not None:
        _write_toy_units(paths, pd.DataFrame(
            {
                "plant_id": "P1", "capacity_mw": [c for c, _ in units], "commission_year": [y for _, y in units],
                **{column: values * len(units) for column, values in TOY_UNIT_TYPE.items()},
            }
        ))
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    # toy 的氨只有 2050 年一行、离电厂远：两年各放一行，挪到电厂旁边，量足价低。
    amm = pd.read_csv(paths.inputs_dir / "ammonia_supply_curve.csv")
    amm = pd.concat([amm.assign(year=2050), amm.assign(year=2060)], ignore_index=True)
    amm["longitude"], amm["latitude"], amm["province_name"] = 112.05, 37.0, "Shanxi"
    amm["nh3_supply_kg_per_year"], amm["nh3_cost_lb_usd_per_kg"] = 1.0e10, 0.05
    amm.to_csv(paths.inputs_dir / "ammonia_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, **power_caps})
    scenario = OptimizationScenario(
        experiment_id="TEST-BLEND", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=carbon,
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=pathway_disable,
        solver_time_limit=300,
    )
    if mip_gap is not None:
        scenario = replace(scenario, mip_gap=mip_gap)
    assumptions = OptimizationAssumptions(
        storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0), hub_decisions_continuous=continuous,
    )
    if coal is not None:
        assumptions = replace(
            assumptions, province_coal_cost_cny_per_gj={**assumptions.province_coal_cost_cny_per_gj, "Shanxi": coal},
        )
    solution = _solve_toy(paths, scenario, assumptions)
    return scenario, assumptions, prepare_inputs(paths, scenario, assumptions), solution


@pytest.fixture(scope="module")
def ammonia_continuous(tmp_path_factory):
    """连续 hub、只开放掺氨，电力上限 2050 年 0.75、2060 年 0.6：两年都只改造一部分份额。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    return _solve_blend_toy(tmp_path_factory.mktemp("ammonia"), _ONLY["ammonia"], {2050: 0.75, 2060: 0.6})


@pytest.fixture(scope="module")
def mixed_continuous(tmp_path_factory):
    """连续 hub、除退役外全部路径开放、有碳价，电力上限 2050 年 0.45、2060 年 0.2。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    return _solve_blend_toy(
        tmp_path_factory.mktemp("mixed"), ("retire",), {2050: 0.45, 2060: 0.2}, carbon=(300.0, 600.0),
    )


@pytest.fixture(scope="module")
def mixed_partly_expired(tmp_path_factory):
    """同 `mixed_continuous`，电厂换成两台机组：600 MW 2005 年投产（2045 年到期）、400 MW 2025 年投产（2065 年到期），
    两年都部分到期（f = 0.6）。退役不开放，到期装机只能原址重建，重建与未重建部分按各自的毛热耗计。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    return _solve_blend_toy(
        tmp_path_factory.mktemp("partly_expired"), ("retire",), {2050: 0.45, 2060: 0.2}, carbon=(300.0, 600.0),
        units=((600.0, 2005), (400.0, 2025)),
    )


@pytest.mark.parametrize(
    ("pathway", "power_caps"), [("beccs", {2050: 0.0, 2060: 0.0}), ("ammonia", {2050: 0.75, 2060: 0.6})],
)
def test_one_hot_levels_give_the_ratio_of_the_selected_level(tmp_path, pathway, power_caps) -> None:
    """独热档位（`hub_decisions_continuous=False`）下档位下标是整数，所选档位的比例与 Σβ·z ÷ 份额相同。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    _, _, _, solution = _solve_blend_toy(tmp_path, _ONLY[pathway], power_caps, continuous=False)
    level_key, levels = ("blend_level_b", LEVELS_B) if pathway == "beccs" else ("blend_level_a", LEVELS_A)
    for ys in solution["year_solutions"].values():
        assert float(ys["share"][0, PATHWAY_INDEX[pathway]]) > 0.0
        ratios = _blend_ratios(
            ys["share"], ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            biomass_levels=LEVELS_B, ammonia_levels=LEVELS_A,
        )
        level = float(ys[level_key][0])
        assert level == pytest.approx(round(level), abs=1e-4) and 1 <= round(level) <= len(levels)
        by_level = levels[round(level) - 1]
        assert float(ratios[pathway][0]) == pytest.approx(by_level, abs=1e-6)
        # 截断会盖住偏大的 Σβ·z（toy 的掺氨解正好在最高档）：未截断的商也要等于按档位读出的比例。
        raw = float(ys[f"{pathway}_blend_x_share"][0]) / float(ys["share"][0, PATHWAY_INDEX[pathway]])
        assert raw == pytest.approx(by_level, abs=1e-6)


def test_detail_ratio_is_the_unclipped_quotient(ammonia_continuous) -> None:
    """连续 hub 下档位下标不是整数，按档位换算不出比例；明细表的比例列等于未截断的 Σβ·z ÷ 份额。"""
    scenario, _, prepared, solution = ammonia_continuous
    for year, ys in solution["year_solutions"].items():
        level = float(ys["blend_level_a"][0])
        assert abs(level - round(level)) > 1e-3, "前提：档位取份额，下标不是整数"
        detail = _build_plant_detail_table(
            prepared, scenario, year, ys["share"], ys["captured_mt_by_plant"], ys["biomass_use_gj"],
            ys["ammonia_use_kg"], ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"], ys["air_share"],
            year_data=ys["year_data"], plant_reduction_mt=ys["plant_reduction_mt"],
            biomass_blend_x_share=ys["biomass_blend_x_share"], beccs_blend_x_share=ys["beccs_blend_x_share"],
            ammonia_blend_x_share=ys["ammonia_blend_x_share"], air_installed=ys["air_installed"],
        )
        # 解正好在氨的最高档：比例列要等于未截断的商，不能靠截断碰巧对上。
        raw = float(ys["ammonia_blend_x_share"][0]) / float(ys["share"][0, AMM])
        assert float(detail["ammonia_blend_ratio"].iloc[0]) == pytest.approx(raw, abs=1e-6)


@pytest.mark.parametrize("solved", ["ammonia_continuous", "mixed_continuous", "mixed_partly_expired"])
def test_pathway_split_adds_up_to_the_solver(request, solved) -> None:
    """逐路径的减排量与捕集量逐厂相加，等于求解器的逐厂减排量与捕集量。拆分由份额、Σβ·z、空冷份额、重建部分与
    `year_data` 的系数重算，求解器那边是约束表达式与捕集变量的取值，两边各算各的。合理性检查的 `pathway_split_closure`
    行随之通过。部分到期的 toy 里重建部分落在 CCS、掺氨（2050 年）与 BECCS（2060 年）上，这几条路径按两部分之差的项
    被测到；生物质与空冷没用上。两边用同一组差值系数，这里核对的是逐项加总，系数本身由 `test_unit_expiry` 独立核对。"""
    scenario, _, prepared, solution = request.getfixturevalue(solved)
    captured_any = False
    rebuilt_retrofit = 0.0
    for year, ys in solution["year_solutions"].items():
        pathways = _build_pathway_table(
            prepared, scenario, year, ys["share"],
            ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            year_data=ys["year_data"], air_share=ys["air_share"],
            rebuilt_share=ys["rebuilt_share"], rebuilt_blend_x_share=ys["rebuilt_blend_x_share"],
        )
        rebuilt_retrofit += float(ys["rebuilt_share"][:, [PATHWAY_INDEX["ccs"], BIO, BECCS, AMM]].sum())
        # 重建与未重建两部分各自恰好分摊到各档位上（`rzsum_*`、rz <= z）：Σβ·rz ÷ 重建份额、
        # (Σβ·z - Σβ·rz) ÷ (份额 - 重建份额) 都落在档位之内。
        for col, key, levels in (
            (BIO, "biomass", LEVELS_B), (BECCS, "beccs", LEVELS_B), (AMM, "ammonia", LEVELS_A),
        ):
            rebuilt, rebuilt_xs = ys["rebuilt_share"][:, col], ys["rebuilt_blend_x_share"][:, col]
            unexpired, unexpired_xs = ys["share"][:, col] - rebuilt, ys[f"{key}_blend_x_share"] - rebuilt_xs
            for part, part_xs in ((rebuilt, rebuilt_xs), (unexpired, unexpired_xs)):
                on = part > 1e-6
                raw = part_xs[on] / part[on]
                assert np.all((raw >= min(levels) - 1e-6) & (raw <= max(levels) + 1e-6)), (year, key, raw)
        by_plant = pathways.groupby("plant_id", sort=False)[["abatement_mt", "captured_mt"]].sum()
        np.testing.assert_allclose(by_plant["abatement_mt"], ys["plant_reduction_mt"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(by_plant["captured_mt"], ys["captured_mt_by_plant"], rtol=1e-6, atol=1e-9)
        captured_any |= bool(np.any(ys["captured_mt_by_plant"] > 1e-6))
        checks = _build_sanity_checks(
            year, ys["slacks"], pathways, _build_province_table(pathways),
            plant_reduction_mt=ys["plant_reduction_mt"], captured_mt=ys["captured_mt_by_plant"],
        ).set_index("check_name")
        assert checks.loc["pathway_split_closure", "status"] == "pass"
    # 前提：混合 toy 至少有一年在捕集，否则捕集量的拆分没被测到；部分到期的 toy 有重建部分在改造路径上。
    assert captured_any or solved == "ammonia_continuous"
    assert (rebuilt_retrofit > 1e-6) == (solved == "mixed_partly_expired")


@pytest.mark.parametrize("solved", ["mixed_continuous", "mixed_partly_expired"])
def test_plant_cost_carbon_cost_adds_up_to_the_objective_term(request, solved) -> None:
    """有碳价时，成本表逐厂碳成本之和 + 工业残余排放的碳成本 = 目标函数当年的碳成本项（除去折现与年金系数）；
    逐厂基线净运行成本之和 = 目标函数的同名项（部分到期的 toy 含重建部分的差）；能耗惩罚三列（CCS 额外燃料、空冷背压、
    生物质效率）之和 = 目标函数的 energy_penalty_cost，与 `cost_breakdown.csv` 的不折现列相同。除退役外全部路径开放、连续 hub。"""
    scenario, assumptions, prepared, solution = request.getfixturevalue(solved)
    biomass_penalty = 0.0
    for year, ys in solution["year_solutions"].items():
        year_data = ys["year_data"]
        price = float(year_data.carbon_price)
        table = _build_plant_cost_table(
            prepared, year, year_data, ys["share"], ys["biomass_use_gj"],
            plant_reduction_mt=ys["plant_reduction_mt"],
            retrofit_new=ys["retrofit_new"], ccs_om_by_plant=ys["ccs_om_by_plant"],
            stranded_by_plant=ys["stranded_by_plant"], capex_pathway_indices=solution["capex_pathway_indices"],
            rebuilt_share=ys["rebuilt_share"], air_share=ys["air_share"],
            rebuilt_air_share=ys["rebuilt_air_share"], bio_penalty_by_plant=ys["bio_penalty_by_plant"],
        )
        plant_carbon = float(table["carbon_cost_cny"].sum())
        industry = year_data.industry
        industry_residual = float(
            (industry.baseline_emissions_mt - (industry.reduction_mt * ys["industry_share"]).sum(axis=1)).sum()
        )
        interval = scenario.interval_years(scenario.planning_years, scenario.planning_years.index(year), assumptions)
        weight = _discount_factor(year, scenario.discount_base_year, scenario.discount_rate) * _year_objective_weight(
            interval, scenario.discount_rate
        )
        assert price > 0.0 and plant_carbon != 0.0
        assert plant_carbon + price * 1e6 * industry_residual == pytest.approx(
            ys["cost_breakdown_cny"]["carbon_cost"] / weight, rel=1e-9
        )
        assert float(table["baseline_net_cost_cny"].sum()) == pytest.approx(
            ys["cost_breakdown_cny"]["baseline_net_cost"] / weight, rel=1e-9
        )
        penalties = table[["energy_penalty_cny", "air_penalty_cny", "biomass_penalty_cny"]].sum()
        biomass_penalty += float(penalties["biomass_penalty_cny"])
        assert float(penalties.sum()) == pytest.approx(
            ys["cost_breakdown_cny"]["energy_penalty_cost"] / weight, rel=1e-9
        )
        breakdown = _build_cost_breakdown(year, ys["cost_breakdown_cny"], ys["cost_weights"]).set_index("category")
        assert breakdown.loc["energy_penalty_cost", "kind"] == "annual"
        assert breakdown.loc["energy_penalty_cost", "cost_undiscounted_cny"] == pytest.approx(
            float(penalties.sum()), rel=1e-9
        )
    assert biomass_penalty > 0.0, "前提：2060 年走 BECCS，有生物质效率惩罚"


def test_ammonia_blend_x_share_is_the_quantity_the_constraints_use(ammonia_continuous) -> None:
    """掺氨的 Σβ·z 与约束里用的是同一个量：`ammonia_use_kg = G_amm × 热耗 ÷ 氨低热值 × Σβz_amm`；
    生物质与 BECCS 两列为 0；未截断的比例落在氨的档位之内。"""
    _, assumptions, _, solution = ammonia_continuous
    for ys in solution["year_solutions"].values():
        amm_xs = float(ys["ammonia_blend_x_share"][0])
        assert amm_xs > 0.0
        assert float(ys["biomass_blend_x_share"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float(ys["beccs_blend_x_share"][0]) == pytest.approx(0.0, abs=1e-9)
        year_data = ys["year_data"]
        expected_kg = (
            float(year_data.generation_by_pathway[0, AMM]) * float(year_data.heat_rate_eff[0])
            / assumptions.nh3_lhv_gj_per_kg * amm_xs
        )
        assert float(ys["ammonia_use_kg"][0]) == pytest.approx(expected_kg, rel=1e-6)
        raw = amm_xs / float(ys["share"][0, AMM])
        assert min(LEVELS_A) - 1e-6 <= raw <= max(LEVELS_A) + 1e-6
