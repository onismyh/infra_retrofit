"""节点可用量 = max(生态余量, 存量) x water_multiplier（`water_access._water_available_by_node`），不求解。

生态余量 = 径流_n x 0.20 − 流域耗水 x 径流_n / 流域径流（耗水按节点径流份额摊）；存量 = 归到该节点的煤电不改造
同年耗水，每个 hub 只归它最近的节点（`distance_rank` 为 1）。数取得让 B1 的两个节点余量为负、B2 的为正。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.water_access import _water_available_by_node

MEMBER = "toy|gcm|ssp126"
YEAR = 2050
# 节点 -> (流域, 全年径流, 枯水期径流, 偏差因子)，m3/yr
NODES = {"W1": ("B1", 6.0e7, 2.0e7, 0.5), "W2": ("B1", 4.0e7, 1.0e7, 0.5), "W3": ("B2", 1.0e8, 5.0e7, 2.0)}
# 流域 -> (生活, 灌溉, 枯水期生活, 枯水期灌溉)，m3/yr
USE = {"B1": (5.0e6, 2.0e7, 5.0e6, 2.0e6), "B2": (1.0e6, 2.0e6, 1.0e6, 0.0)}
# P1 最近的节点是 W1（W3 是它的第二近），P2 最近的是 W2；不改造耗水 P1 4e6、P2 1e6 m3/yr。
LINKS = (("P1", "W1", 1), ("P1", "W3", 2), ("P2", "W2", 1))
BASELINE = np.array([4.0e6, 1.0e6])


def _prepared(use: dict = USE, links: tuple = LINKS) -> SimpleNamespace:
    availability = pd.DataFrame([
        {"water_node_id": node, "planning_year": YEAR, "scenario_family": "baseline", "scenario_id": MEMBER,
         "basin_code": basin, "available_water_m3_per_year": annual, "dry_season_water_m3_per_year": dry,
         "bias_factor": bias}
        for node, (basin, annual, dry, bias) in NODES.items()
    ])
    basin_use = pd.DataFrame([
        {"scenario_id": MEMBER, "planning_year": YEAR, "basin_code": basin, "domestic_m3_per_year": u[0],
         "irrigation_m3_per_year": u[1], "dry_season_domestic_m3_per_year": u[2],
         "dry_season_irrigation_m3_per_year": u[3]}
        for basin, u in use.items()
    ])
    return SimpleNamespace(
        water_availability=availability,
        water_basin_use=basin_use,
        water_links=pd.DataFrame(links, columns=["plant_id", "water_node_id", "distance_rank"]),
        plants=pd.DataFrame({"plant_id": ["P1", "P2"]}),
    )


def _available(prepared=None, season: str = "annual", multiplier: float = 1.0, bias: bool = True,
               baseline: np.ndarray = BASELINE, order=("W1", "W2", "W3")) -> np.ndarray:
    scenario = OptimizationScenario(
        experiment_id="T", description="toy", water_mode="grid_supply", water_scenario_id=MEMBER,
        water_season=season, water_multiplier=multiplier,
    )
    assumptions = OptimizationAssumptions(apply_bias_correction=bias)
    nodes = pd.DataFrame({"water_node_id": list(order)})
    return _water_available_by_node(prepared or _prepared(), scenario, assumptions, YEAR, nodes, baseline)


def test_residual_spreads_basin_use_by_runoff_share_and_keeps_existing_use() -> None:
    # B1：径流 1e8，耗水 2.5e7；W1 = 1.2e7 − 2.5e7 x 0.6 = −3e6，W2 = 8e6 − 2.5e7 x 0.4 = −2e6，都取存量。
    # B2：W3 = 2e7 − 3e6 = 1.7e7。
    assert _available() == pytest.approx([4.0e6, 1.0e6, 1.7e7])
    # P1 的第二近节点是 W3，存量不算在 W3 上：P1 不改造耗水 2e7 高于 W3 的余量，W3 仍取余量。
    assert _available(baseline=np.array([2.0e7, 1.0e6])) == pytest.approx([2.0e7, 1.0e6, 1.7e7])
    # 没有煤电的节点余量为负时取 0；输出按 `nodes` 的顺序排。
    assert _available(baseline=np.zeros(2), order=("W3", "W1", "W2")) == pytest.approx([1.7e7, 0.0, 0.0])
    # 余量为正但低于存量时取存量：P1 改归 W3、不改造耗水 2e7 > 1.7e7。
    moved = _prepared(links=(("P1", "W3", 1), ("P2", "W2", 1)))
    assert _available(moved, baseline=np.array([2.0e7, 1.0e6])) == pytest.approx([0.0, 1.0e6, 2.0e7])


def test_dry_season_pairs_runoff_and_use_columns() -> None:
    # B1 枯水期：径流 3e7，耗水 7e6；W1 = 4e6 − 7e6 x 2/3 < 0 取存量 4e6，W2 = 2e6 − 7e6 x 1/3 < 0 取存量 1e6。
    # B2：W3 = 1e7 − 1e6 = 9e6。若误用全年耗水，W3 会是 1e7 − 3e6 = 7e6。
    assert _available(season="dry") == pytest.approx([4.0e6, 1.0e6, 9.0e6])
    # 关掉偏差校正后 B1 的余量转正（径流 4e7、2e7）：W2 = 4e6 − 7e6 / 3 高于存量 1e6，露出余量本身；
    # W1 = 8e6 − 7e6 x 2/3 低于存量 4e6；W3 = 5e6 − 1e6。
    assert _available(season="dry", bias=False) == pytest.approx([4.0e6, 4.0e6 - 7.0e6 / 3.0, 4.0e6])


def test_bias_off_restores_runoff_only_and_multiplier_scales_the_result() -> None:
    # 关掉偏差校正：径流除回偏差因子（W1 1.2e8、W2 8e7、W3 5e7），耗水不变。
    # W1 = 2.4e7 − 2.5e7 x 0.6 = 9e6，W2 = 1.6e7 − 2.5e7 x 0.4 = 6e6，W3 = 1e7 − 3e6 = 7e6。
    assert _available(bias=False) == pytest.approx([9.0e6, 6.0e6, 7.0e6])
    # 乘子乘在最终的可用量上（含存量）。
    assert _available(multiplier=0.5) == pytest.approx([2.0e6, 0.5e6, 8.5e6])


def test_missing_use_rows_or_season_column_raise() -> None:
    with pytest.raises(ValueError, match="build_water_use.py"):
        _available(_prepared(use={"B1": USE["B1"]}))
    prepared = _prepared()
    prepared.water_availability = prepared.water_availability.drop(columns="dry_season_water_m3_per_year")
    # 缺枯水期列时报错，不退回全年值。
    with pytest.raises(ValueError, match="dry_season_water_m3_per_year"):
        _available(prepared, season="dry")
