"""掺生物质比例的炉型上限（2026-10-02 起，`constraints._add_blend_level_constraints`）与 hub 的 CFB 装机份额
（`data_prep._with_expiry`）。

高于煤粉炉上限（0.15）的档位只有 CFB 能用：落在这些档位上的份额（生物质与 BECCS 合计）不超过 hub 的 CFB 装机份额；
高于 CFB 上限（0.30）的档位谁都不能用。约束的用例只建一个 hub 的掺烧档位约束（需要 Gurobi），校验的用例在建模之前
报错（不需要 Gurobi），份额的用例读 toy 机组表。
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from coal_retrofit.optimization._shared import PATHWAY_INDEX
from coal_retrofit.optimization.constraints import _add_blend_level_constraints
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from coal_retrofit.run_controls import initial_state
from test_blend_ratios import SCENARIO
from test_unit_expiry import _unit_toy

GUROBI = "gurobipy is required for solver integration tests"


def _max_blend_ratio(
    cfb_share: float, *, share: float = 0.5, continuous: bool = True, scenario=SCENARIO, **assumption_overrides
) -> float:
    """一个 hub，生物质与 BECCS 份额各 `share`：两条路径的 Σβ·z 之和除以份额合计（掺烧份额的平均掺烧比例）最大能到多少。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    year_data = cast(YearData, SimpleNamespace(
        emissions_retrofit_mt=np.ones(1), heat_rate_eff=np.full(1, 9.0),
        generation_by_pathway=np.ones((1, len(PATHWAYS))),
        biomass_penalty_coeff_per_level=0.0, biomass_penalty_emissions_coeff_per_level=0.0,
        beccs_penalty_emissions_coeff_per_level=0.0, beccs_penalty_captured_coeff_per_level=0.0,
        cfb_share=np.full(1, cfb_share),
    ))
    model = gp.Model()
    model.Params.OutputFlag = 0
    fixed = np.zeros((1, len(PATHWAYS)))
    fixed[0, PATHWAY_INDEX["biomass"]] = fixed[0, PATHWAY_INDEX["beccs"]] = share
    shares = model.addMVar(fixed.shape, lb=fixed, ub=fixed)
    assumptions = OptimizationAssumptions(hub_decisions_continuous=continuous, **assumption_overrides)
    blocks = _add_blend_level_constraints(model, shares, 1, scenario, assumptions, year_data, "")
    model.setObjective(blocks[-3][0] + blocks[-2][0], gp.GRB.MAXIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    return float(model.ObjVal) / (2.0 * share)


@pytest.mark.parametrize(("continuous", "mixed"), [(True, 0.4 * 0.20 + 0.6 * 0.15), (False, 0.15)])
def test_levels_above_the_pulverized_max_run_only_on_the_cfb_share(continuous: bool, mixed: float) -> None:
    """CFB 装机份额 0.4，生物质与 BECCS 份额合计 1：连续 hub 下 0.4 的容量用 0.20 档、其余 0.6 用 0.15 档，平均 0.17；
    独热档位下两条路径的份额都落在所选的一档上，0.20 档要占满整个 hub，超过 CFB 份额，只能到 0.15。上限按生物质与
    BECCS 合计，不是各 0.4。全是 CFB 时到最高档 0.20，没有 CFB 时到 0.15。"""
    assert _max_blend_ratio(0.4, continuous=continuous) == pytest.approx(mixed, abs=1e-9)
    assert _max_blend_ratio(1.0, continuous=continuous) == pytest.approx(0.20, abs=1e-9)
    assert _max_blend_ratio(0.0, continuous=continuous) == pytest.approx(0.15, abs=1e-9)


def test_one_hot_hub_takes_the_cfb_level_when_the_blend_share_fits_in_the_cfb_share() -> None:
    """独热档位下上限加在落档的份额上，不加在所选档位上：CFB 装机份额 0.4 的 hub，生物质与 BECCS 份额合计 0.4 时能选
    0.20 档，合计 0.5 时只到 0.15。"""
    assert _max_blend_ratio(0.4, share=0.2, continuous=False) == pytest.approx(0.20, abs=1e-9)
    assert _max_blend_ratio(0.4, share=0.25, continuous=False) == pytest.approx(0.15, abs=1e-9)


def test_levels_above_the_cfb_max_are_closed() -> None:
    """加一个 0.35 档（高于 CFB 上限 0.30）：全是 CFB 的 hub 也只到 0.20。两个上限都设 1.0（不限）时，没有 CFB 的 hub
    也到 0.35。"""
    scenario = replace(SCENARIO, biomass_blend_levels=(0.10, 0.15, 0.20, 0.35))
    assert _max_blend_ratio(1.0, scenario=scenario) == pytest.approx(0.20, abs=1e-9)
    unlimited = {"biomass_blend_max_pulverized": 1.0, "biomass_blend_max_cfb": 1.0}
    assert _max_blend_ratio(0.0, scenario=scenario, **unlimited) == pytest.approx(0.35, abs=1e-9)


def test_bad_levels_and_caps_are_rejected() -> None:
    """档位不在 (0, 1] 内严格升序（乱序、重复、0、大于 1、NaN），或煤粉炉上限不为正、高于 CFB 上限时建约束报错，不静默地
    多关档位。校验在建模之前，模型、份额与年度数据都传 None，不需要 Gurobi。"""
    def build(scenario: OptimizationScenario = SCENARIO, **assumption_overrides: float) -> None:
        _add_blend_level_constraints(None, None, 1, scenario, OptimizationAssumptions(**assumption_overrides), None, "")

    for levels in ((0.20, 0.10, 0.15), (0.0, 0.10), (0.10, 1.5), (0.10, float("nan"))):
        with pytest.raises(ValueError, match="biomass_blend_levels"):
            build(replace(SCENARIO, biomass_blend_levels=levels))
    with pytest.raises(ValueError, match="ammonia_blend_levels"):
        build(replace(SCENARIO, ammonia_blend_levels=(0.10, 0.10)))
    for pulverized, cfb in ((0.30, 0.15), (0.0, 0.30)):
        with pytest.raises(ValueError, match="biomass_blend_max_pulverized"):
            build(biomass_blend_max_pulverized=pulverized, biomass_blend_max_cfb=cfb)


@pytest.mark.parametrize("continuous", [True, False])
def test_cfb_share_is_the_cfb_capacity_share_of_the_hub(tmp_path, continuous: bool) -> None:
    """hub 的 CFB 装机份额按机组表的 `combustion` 计，到年度数据 `YearData.cfb_share`：600 MW CFB（带空格、`/CCS` 后缀、
    小写的也算）+ 400 MW 超临界为 0.6。不随规划年变：2050 年 CFB 机组已到期（连续 hub）、整个 hub 已到期（整数 hub），
    份额都不变。"""
    for label in ("CFB", " cfb/CCS"):
        paths = _unit_toy(tmp_path / label.strip().replace("/", "_"), combustion=(label, "supercritical"))
        scenario = OptimizationScenario(
            experiment_id="T", description="toy", planning_years=(2040, 2050), sector_target_source="toy"
        )
        assumptions = OptimizationAssumptions(hub_decisions_continuous=continuous)
        prepared = prepare_inputs(paths, scenario, assumptions)
        assert prepared.plants.loc[0, "expired_share_2050"] > 0.0
        state = initial_state(prepared)
        for year in scenario.planning_years:
            year_data = _build_year_matrices(prepared, scenario, assumptions, year, state)
            assert year_data.cfb_share == pytest.approx([0.6], abs=1e-12)
