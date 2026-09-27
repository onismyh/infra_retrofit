"""连续 hub 下的掺烧比例换算与逐厂碳成本。

连续 hub 下一个 hub 可以把不同份额改造到不同档位，`blend_level = Σ l·select` 只是档位下标的加权和：
一半第 1 档、一半第 3 档记作 2，按档位读成 0.25，实际是 0.30。结果表改按约束里的 Σβ_l·z_l 除以
路径份额换算（`results_plant._blend_ratios`）；`blend_level_to_ratio` 只认整数档位；`plot_style` 的
残余排放只认比例列，没有就报错；成本表的碳成本与目标函数同式，用求解器的逐厂减排量。求解 toy 的几条
（只开 BECCS 的生物质用量、独热档位对照、`plot_style` 残余排放与求解器对拍、成本表碳成本与目标函数对拍、
掺氨用量）需要 Gurobi，其余不依赖。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
from coal_retrofit.optimization.emissions import blend_level_to_ratio
from coal_retrofit.optimization.results_plant import (
    _blend_ratios,
    _build_pathway_table,
    _build_plant_cost_table,
    _build_plant_detail_table,
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


def test_blend_level_to_ratio_reads_integer_levels_only() -> None:
    assert blend_level_to_ratio(None, LEVELS_B) == 0.0
    assert blend_level_to_ratio(0.0, LEVELS_B) == 0.0
    assert blend_level_to_ratio(2.0, LEVELS_B) == 0.25
    assert blend_level_to_ratio(5.0 + 5e-5, LEVELS_B) == 1.0  # 二元档位的整数容差之内
    # hub 0 的下标恰为整数，按档位只能读成第 2 档：这正是它不能用于连续 hub 的原因。
    assert blend_level_to_ratio(_three_hubs()["level_b"][0], LEVELS_B) == 0.25
    for bad in (2.5, 2.8, 6.0, -1.0):
        with pytest.raises(ValueError, match="blend_ratio"):
            blend_level_to_ratio(bad, LEVELS_B)


def test_detail_and_pathway_tables_use_the_effective_ratios() -> None:
    hubs = _three_hubs()
    n = len(hubs["share"])
    prepared = _prepared(n)
    emissions = np.full(n, 10.0)
    year_data = cast(YearData, SimpleNamespace(generation=np.full(n, 5.0e6), emissions_mt=emissions))
    reduction = np.array([3.0, 8.0, 2.5])
    zeros = np.zeros(n)

    detail = _build_plant_detail_table(
        prepared, SCENARIO, 2040, hubs["share"], zeros, zeros, zeros, zeros,
        hubs["level_b"], hubs["level_a"], np.zeros((n, len(PATHWAYS))),
        year_data=year_data, plant_reduction_mt=reduction,
        biomass_blend_x_share=hubs["bio_xs"], beccs_blend_x_share=hubs["beccs_xs"],
        ammonia_blend_x_share=hubs["amm_xs"],
    )
    np.testing.assert_allclose(detail["biomass_blend_ratio"], [0.30, 0.75, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["beccs_blend_ratio"], [0.0, 0.25, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["ammonia_blend_ratio"], [0.0, 0.0, 0.50], rtol=1e-12)
    # 档位下标照原样保留（连续 hub 下是加权下标）。
    np.testing.assert_allclose(detail["biomass_blend_level"], hubs["level_b"], rtol=1e-12)

    pathways = _build_pathway_table(
        prepared, SCENARIO, 2040, hubs["share"], zeros,
        hubs["bio_xs"], hubs["beccs_xs"], hubs["amm_xs"],
        year_data=year_data, plant_reduction_mt=reduction,
    )
    hub1 = pathways[pathways["plant_id"] == "P1"].set_index("pathway")["abatement_mt"]
    # 厂合计锚定到求解器的减排量；生物质与 BECCS 按各自的比例拆分：
    # 生物质 0.75 × 0.4 = 0.30，BECCS (0.90 + 0.25) × 0.6 = 0.69。此前把档位下标 2.8 当比例拆，生物质分到 1.12 / 3.34。
    assert hub1.sum() == pytest.approx(8.0, rel=1e-12)
    assert hub1["biomass"] == pytest.approx(8.0 * 0.30 / 0.99, rel=1e-12)
    assert hub1["beccs"] == pytest.approx(8.0 * 0.69 / 0.99, rel=1e-12)


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
            energy_penalty_matrix=zeros_path, ccs_om_matrix=zeros_path, stranded_per_plant=np.zeros(n),
        ))
        return _build_plant_cost_table(
            _prepared(n), 2040, year_data, share, np.zeros(n),
            # 第 2 个 hub 的减排为负：惩罚燃料使排放高于基线，碳成本随之高于基线排放的碳价。
            plant_reduction_mt=np.array([7.5, -0.2]),
            retrofit_installed=np.zeros((n, 1)), capex_pathway_indices=(PATHWAY_INDEX["ccs"],),
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


def _solve_blend_toy(root, pathway_disable, power_caps, *, continuous=True, carbon=(0.0, 0.0), coal=None, mip_gap=None):
    """求解 2050、2060 两年的 toy，生物质与氨的供给挪到电厂旁边且充足。

    `coal` 改 toy 电厂所在省（山西）的煤价，元/GJ；`mip_gap` 缺省用情景的缺省值。
    返回 (scenario, assumptions, prepared, solution)；`prepared` 供结果表函数用。
    """
    from test_capex_stock_and_lifetimes import _solve_toy
    from toy_inputs import _write_targets, _write_toy_inputs

    paths = _write_toy_inputs(root, retirement_year=9999)
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


@pytest.mark.parametrize(
    ("pathway", "power_caps"), [("beccs", {2050: 0.0, 2060: 0.0}), ("ammonia", {2050: 0.75, 2060: 0.6})],
)
def test_one_hot_levels_give_the_ratio_blend_level_to_ratio_reads(tmp_path, pathway, power_caps) -> None:
    """独热档位（`hub_decisions_continuous=False`）下档位下标是整数，按档位读出的比例与 Σβ·z ÷ 份额相同。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    _, _, _, solution = _solve_blend_toy(tmp_path, _ONLY[pathway], power_caps, continuous=False)
    level_key, levels = ("blend_level_b", LEVELS_B) if pathway == "beccs" else ("blend_level_a", LEVELS_A)
    for ys in solution["year_solutions"].values():
        assert float(ys["share"][0, PATHWAY_INDEX[pathway]]) > 0.0
        ratios = _blend_ratios(
            ys["share"], ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            biomass_levels=LEVELS_B, ammonia_levels=LEVELS_A,
        )
        by_level = blend_level_to_ratio(float(ys[level_key][0]), levels)
        assert float(ratios[pathway][0]) == pytest.approx(by_level, abs=1e-6)
        # 截断会盖住偏大的 Σβ·z（toy 的掺氨解正好在最高档）：未截断的商也要等于按档位读出的比例。
        raw = float(ys[f"{pathway}_blend_x_share"][0]) / float(ys["share"][0, PATHWAY_INDEX[pathway]])
        assert raw == pytest.approx(by_level, abs=1e-6)


def test_plot_style_residual_reads_the_ratio_columns(ammonia_continuous, monkeypatch) -> None:
    """`plot_style.residual_emissions_mt` 按 `*_blend_ratio` 列算，与求解器的残余排放一致；删去这几列就报错。
    解取连续 hub，档位下标不是整数，按档位换算不出比例。"""
    pytest.importorskip("matplotlib", reason="plot_style imports matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import plot_style

    scenario, _, prepared, solution = ammonia_continuous
    for year, ys in solution["year_solutions"].items():
        level = float(ys["blend_level_a"][0])
        assert abs(level - round(level)) > 1e-3, "前提：档位取份额，下标不是整数"
        detail = _build_plant_detail_table(
            prepared, scenario, year, ys["share"], ys["captured_mt_by_plant"], ys["biomass_use_gj"],
            ys["ammonia_use_kg"], ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"], ys["air_share"],
            year_data=ys["year_data"], plant_reduction_mt=ys["plant_reduction_mt"],
            biomass_blend_x_share=ys["biomass_blend_x_share"], beccs_blend_x_share=ys["beccs_blend_x_share"],
            ammonia_blend_x_share=ys["ammonia_blend_x_share"],
        )
        # 解正好在氨的最高档：比例列要等于未截断的商，不能靠截断碰巧对上。
        raw = float(ys["ammonia_blend_x_share"][0]) / float(ys["share"][0, AMM])
        assert float(detail["ammonia_blend_ratio"].iloc[0]) == pytest.approx(raw, abs=1e-6)
        model_residual = float(np.sum(ys["year_data"].emissions_mt - ys["plant_reduction_mt"]))
        assert plot_style.residual_emissions_mt(detail, year) == pytest.approx(model_residual, rel=1e-9)
        without_ratios = detail.drop(columns=["biomass_blend_ratio", "beccs_blend_ratio", "ammonia_blend_ratio"])
        with pytest.raises(ValueError, match="blend_ratio"):
            plot_style.residual_emissions_mt(without_ratios, year)


def test_plot_style_residual_refuses_tables_without_ratio_columns(monkeypatch) -> None:
    """没有 `*_blend_ratio` 列就报错，哪怕档位下标是整数：整数下标可能是独热档位，也可能是连续 hub 下
    几档的混合（`_three_hubs` 的 hub 0：一半第 1 档、一半第 3 档记作 2，实际比例 0.30），表里分不出来。
    此前的回退分支把它按第 2 档读成 0.25，静默出数。
    补上比例列就按比例算：档位下标不参与，BECCS 读自己的列。"""
    pytest.importorskip("matplotlib", reason="plot_style imports matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import plot_style

    hub0 = pd.DataFrame([{
        "retirement_year": 2060, "baseline_emissions_mt": 10.0, "annual_generation_mwh": 5.0e6,
        "share_unabated": 0.0, "share_ccs": 0.0, "share_biomass": 1.0, "share_beccs": 0.0,
        "share_ammonia": 0.0, "share_retire": 0.0,
        "biomass_blend_level": float(_three_hubs()["level_b"][0]), "ammonia_blend_level": 0.0,
    }])
    with pytest.raises(ValueError, match="blend_ratio"):
        plot_style.residual_emissions_mt(hub0, 2040)
    with_ratios = hub0.assign(biomass_blend_ratio=0.30, beccs_blend_ratio=0.0, ammonia_blend_ratio=0.0)
    value = plot_style.residual_emissions_mt(with_ratios, 2040)
    # 档位下标不参与：换下标结果不变；按实际比例 0.30 算，比按第 2 档读成 0.25 排得少。
    assert plot_style.residual_emissions_mt(with_ratios.assign(biomass_blend_level=4.0), 2040) == value
    assert value < plot_style.residual_emissions_mt(with_ratios.assign(biomass_blend_ratio=0.25), 2040)
    # BECCS 读自己的比例列（比例高的排得少），不读生物质的（此前的回退分支令 BECCS 与生物质同比例）。
    beccs = with_ratios.assign(share_biomass=0.0, share_beccs=1.0,
                               biomass_blend_ratio=0.75, beccs_blend_ratio=0.30)
    residual = plot_style.residual_emissions_mt(beccs, 2040)
    assert residual < plot_style.residual_emissions_mt(beccs.assign(beccs_blend_ratio=0.10), 2040)
    assert residual == plot_style.residual_emissions_mt(beccs.assign(biomass_blend_ratio=0.0), 2040)


def test_plant_cost_carbon_cost_adds_up_to_the_objective_term(tmp_path) -> None:
    """有碳价时，成本表逐厂碳成本之和 + 工业残余排放的碳成本 = 目标函数当年的碳成本项（除去折现与年金系数）。
    除退役外全部路径开放、连续 hub。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    scenario, assumptions, prepared, solution = _solve_blend_toy(
        tmp_path, ("retire",), {2050: 0.45, 2060: 0.2}, carbon=(300.0, 600.0),
    )
    for year, ys in solution["year_solutions"].items():
        year_data = ys["year_data"]
        price = float(year_data.carbon_price)
        table = _build_plant_cost_table(
            prepared, year, year_data, ys["share"], ys["biomass_use_gj"],
            plant_reduction_mt=ys["plant_reduction_mt"],
            retrofit_installed=ys["retrofit_installed"],
            capex_pathway_indices=solution["capex_pathway_indices"],
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
