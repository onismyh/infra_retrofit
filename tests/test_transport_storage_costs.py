"""管道固定运维与封存按汇型定价（2026-10-02 起）。

管道运维 = 在役各代管道的 capex x `pipe_fixed_om_fraction`（`model_costs._transport_storage_costs`），在役与流量上限
同一判据，到寿命退出，原址重铺的那一代另计；按流量计的一项缺省为 0。每个汇的封存单价 = 基准价 x 海上倍率 − EOR 容量
份额 x 抵扣（`data_prep._prepare_storages`）；结果工作簿按各汇的扣前单价记封存费与 EOR 抵扣
（`results_workbook_network.cost_lines`）。封存单价与工作簿的用例不需要 Gurobi；管道运维的用例求解 toy，比例为负或 NaN
时报错的用例只建表达式，两者都要 Gurobi。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.optimization.data_prep import _prepare_storages, prepare_inputs
from coal_retrofit.optimization.model_costs import _transport_storage_costs
from coal_retrofit.optimization.results_costs import COST_COLUMNS, SYSTEM_COLUMNS
from coal_retrofit.optimization.results_network import _build_sinks_table
from coal_retrofit.optimization.results_regions import OFFSHORE
from coal_retrofit.optimization.results_workbook_network import cost_lines
from coal_retrofit.optimization.results_workbook_sources import sinks_frame
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.run_controls import initial_state
from toy_inputs import _write_targets, _write_toy_inputs

GUROBI = "gurobipy is required for solver integration tests"

# 四类汇各一个：陆上 DSA、海上 DSA、陆上 EOR、海上 EOR。
SINKS = pd.DataFrame({
    "storage_hub_id": ["S1", "S2", "S3", "S4"],
    "storage_type": ["dsa", "dsa", "eor", "eor"],
    "offshore": [False, True, False, True],
    "storage_all_mt": 100.0,
    "storage_dsa_mt": [100.0, 100.0, 0.0, 0.0],
    "storage_eor_mt": [0.0, 0.0, 100.0, 100.0],
    "injectivity_dsa_avg_mtpa": [1.0, 1.0, 0.0, 0.0],
    "injectivity_eor_avg_mtpa": [0.0, 0.0, 1.0, 1.0],
    "latitude": 37.0,
    "longitude": 112.5,
})


def _storages(tmp_path, sinks: pd.DataFrame, **assumption_overrides: float) -> pd.DataFrame:
    """把 *sinks* 写成汇表，按 `_prepare_storages` 读入。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    sinks.to_csv(paths.inputs_dir / "storage_hubs.csv", index=False)
    scenario = OptimizationScenario(experiment_id="T", description="toy")
    return _prepare_storages(paths, scenario, OptimizationAssumptions(**assumption_overrides))


def _prices(tmp_path, sinks: pd.DataFrame, **assumption_overrides: float) -> pd.DataFrame:
    """各汇扣 EOR 抵扣前、后的每吨封存成本（CNY/t），按汇号索引。"""
    storages = _storages(tmp_path, sinks, **assumption_overrides).set_index("storage_hub_id")
    return storages[["storage_cost_before_credit_cny_per_t", "storage_cost_cny_per_t"]]


def test_storage_is_priced_by_sink_type(tmp_path) -> None:
    """陆上 DSA 35、海上 DSA 35 x 2.2、陆上 EOR 35 − 12、海上 EOR 35 x 2.2 − 12（海上倍率是注入的成本，抵扣另算）。
    基准价设回 32、海上倍率设 1 即旧口径（DSA 32、EOR 20，不分陆海）；汇表没有 `offshore` 列时都按陆上计。"""
    prices = _prices(tmp_path, SINKS)
    assert prices["storage_cost_before_credit_cny_per_t"].tolist() == pytest.approx([35.0, 77.0, 35.0, 77.0])
    assert prices["storage_cost_cny_per_t"].tolist() == pytest.approx([35.0, 77.0, 23.0, 65.0])
    old = _prices(tmp_path, SINKS, storage_cost_cny_per_t=32.0, offshore_storage_multiplier=1.0)
    assert old["storage_cost_cny_per_t"].tolist() == pytest.approx([32.0, 32.0, 20.0, 20.0])
    onshore = _prices(tmp_path, SINKS.drop(columns="offshore"))
    assert onshore["storage_cost_cny_per_t"].tolist() == pytest.approx([35.0, 35.0, 23.0, 23.0])


def test_workbook_backs_the_eor_credit_out_of_each_sinks_own_price(tmp_path) -> None:
    """封存费 = 封存量 x 各汇的扣前单价，EOR 抵扣 = 封存量 x（扣前 − 扣后），两种单价经 `sinks.csv`（`_build_sinks_table`）
    与 `sinks_frame` 取自 `_prepare_storages`：只有 EOR 汇有抵扣，海上 DSA 汇的抵扣为 0（按全国一个基准价 35 反推会记出
    −42 元/t）；封存费 − 抵扣 = Σ 封存量 x 扣后单价，即目标函数的 storage_cost。"""
    storages = _storages(tmp_path, SINKS)
    ids = storages["storage_hub_id"].tolist()
    prepared = SimpleNamespace(storages=storages, network=SimpleNamespace(storage_node_ids={i: f"n_{i}" for i in ids}))
    node_province = {f"n_{i}": OFFSHORE if offshore else "Shanxi" for i, offshore in zip(ids, storages["offshore"])}
    years = [2050, 2060]
    injected = {2050: [1.0, 2.0, 3.0, 4.0], 2060: [0.0, 5.0, 0.0, 6.0]}
    capacity = storages["available_capacity_mt"].astype(float).to_numpy()

    def sinks_table(year: int) -> pd.DataFrame:
        ys = {
            "storage_use_mtpa": np.array(injected[year]),
            "year_data": SimpleNamespace(storage_injectivity_mtpa=storages["injectivity_mtpa"].astype(float).to_numpy()),
            "slacks": {"injectivity_slack_mtpa": np.zeros(len(ids)), "storage_slack_mt": np.zeros(len(ids))},
        }
        return _build_sinks_table(
            prepared, year, ys, SimpleNamespace(remaining_storage_mt=capacity), 10, node_province,  # type: ignore[arg-type]
        )

    tables = {
        "sinks": pd.concat([sinks_table(year) for year in years], ignore_index=True),
        "costs": pd.DataFrame(columns=COST_COLUMNS),
        "system": pd.DataFrame(columns=SYSTEM_COLUMNS),
    }
    sinks = sinks_frame(tables, prepared, node_province)  # type: ignore[arg-type]
    lines = cost_lines(tables, sinks, years)
    [(_, _, _, stored)] = lines["Cost_Storage"]
    [(_, _, _, credit)] = lines["Revenue_EOR"]
    assert stored == pytest.approx({2050: (35.0 + 154.0 + 105.0 + 308.0) * 1e6, 2060: (385.0 + 462.0) * 1e6})
    assert credit == pytest.approx({2050: (3.0 + 4.0) * 12.0 * 1e6, 2060: 6.0 * 12.0 * 1e6})
    for year in years:
        rows = sinks[sinks["year"] == year]
        assert stored[year] - credit[year] == pytest.approx(float((rows["use_mt"] * rows["cost_cny_per_t"]).sum()) * 1e6)


def test_pipe_om_is_charged_on_the_capex_of_pipes_in_service(tmp_path) -> None:
    """1 座电厂 - 1 条边 - 1 个汇，只开放 CCS，每年捕集约一半排放（> 2 Mtpa，每条边只容一根 5 Mtpa 管）。2030 年铺的管
    2050 年仍在役，2050 年不铺新管也付运维；2060 年满 30 年寿命退出、在原址重铺，只付重铺那一代的。按流量计的一项缺省
    为 0，运输运维（各年不折现）就是 4% x 在役管道的 capex。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.5, 2040: 0.5, 2050: 0.5, 2060: 0.5})
    years = (2030, 2050, 2060)
    scenario = OptimizationScenario(
        experiment_id="TEST-PIPE-OM", description="toy", planning_years=years,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 490.0, 550.0),
        pathway_disable=("retire", "biomass", "beccs", "ammonia"),
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(
        standard_pipe_capacity_mtpa=5.0, max_parallel_pipes=1,
    )
    assert assumptions.pipe_fixed_om_fraction == 0.04 and assumptions.route_opex_cny_per_t_km == 0.0
    prepared = prepare_inputs(paths, scenario, assumptions)
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, initial_state(prepared))
    assert solution["status"] == "optimal"
    ys = solution["year_solutions"]

    def undiscounted(year: int, name: str) -> float:
        return float(ys[year]["cost_breakdown_cny"][name]) / float(ys[year]["cost_weights"][name][1])

    pipes = {year: np.rint(np.asarray(ys[year]["pipe_count"])) for year in years}
    assert pipes[2030].sum() == 1 and pipes[2050].sum() == 0 and (pipes[2060] == pipes[2030]).all()
    capex = {year: undiscounted(year, "pipe_capex") for year in years}
    assert capex[2030] > 0.0 and capex[2060] == pytest.approx(capex[2030], rel=1e-4)
    # 各年在役的建设年：2030 → {2030}；2050 → {2030, 2050}；2060 → {2050, 2060}（2030 年那一代满 30 年退出）。
    in_service = {2030: capex[2030], 2050: capex[2030] + capex[2050], 2060: capex[2050] + capex[2060]}
    for year in years:
        assert undiscounted(year, "transport_opex") == pytest.approx(0.04 * in_service[year], rel=1e-6), year
        assert float(ys[year]["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6), year


@pytest.mark.parametrize("fraction", [-0.01, float("nan")])
def test_negative_or_nan_pipe_om_fraction_is_rejected(fraction: float) -> None:
    """管道运维比例为负或 NaN 时报错，不静默当成 0（0 条边、0 个汇，只建表达式，不建模型）。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    payload = SimpleNamespace(year=2030, year_data=SimpleNamespace(edge_route_opex_coeff=np.zeros(0)))
    assumptions = OptimizationAssumptions(pipe_fixed_om_fraction=fraction)
    with pytest.raises(ValueError, match="pipe_fixed_om_fraction"):
        _transport_storage_costs(payload, [payload], 0, SimpleNamespace(), assumptions, 0, 0)  # type: ignore[arg-type, list-item]


def test_national_injection_cap_limits_total_storage_use(tmp_path) -> None:
    """部署进度是全国各汇合计的年注入上限（2026-10-10 起）：设成 1 Mt/a 时两年的注入合计都不超过它，余下的减排要求
    落进缺口；空元组即不设，toy 每年注入约 2.6 Mt（目标所需），没有缺口。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.5, 2040: 0.5})
    years = (2030, 2040)
    scenario = OptimizationScenario(
        experiment_id="TEST-NATIONAL-INJECTION", description="toy", planning_years=years, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("retire", "biomass", "beccs", "ammonia"), solver_time_limit=300,
    )

    def run(cap: tuple[float, ...]):
        assumptions = OptimizationAssumptions(storage_national_injection_mtpa_by_year=cap)
        prepared = prepare_inputs(paths, scenario, assumptions)
        solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, initial_state(prepared))
        assert solution["status"] == "optimal"
        return solution["year_solutions"]

    free, capped = run(()), run((1.0, 1.0, 1.0, 1.0))
    for year in years:
        assert float(np.sum(free[year]["storage_use_mtpa"])) > 1.0 + 1e-3
        assert float(free[year]["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6)
        assert float(np.sum(capped[year]["storage_use_mtpa"])) <= 1.0 + 1e-6
        assert float(capped[year]["slacks"]["target_shortfall_mt"]) > 0.0
