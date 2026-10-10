"""结果表 `resource_use.csv` 的利用率列（`results_resources._build_supply_table`），不求解。

可用量 > 0 时为 用量 / 可用量；可用量 ≤ 0 而仍有用量（用了松弛）记 inf，与图 7 的"无余量"同一规则；没有用量，
或无水约束时可用量为空，记 0。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants import AMMONIA_FLOW_SCALE, NH3_H2_RATIO, WATER_FLOW_SCALE
from coal_retrofit.optimization.results_resources import _build_supply_table


def _utilization(water_available_m3: np.ndarray | None, basin_codes: list[str],
                 basin_available_m3: np.ndarray | None, basin_use_m3: np.ndarray) -> pd.Series:
    """生物质节点 B1 可用量与用量都为零，氨节点 N1 用一半，水节点 W1、W2 各用 3 与 2e6 m3。"""
    prepared = SimpleNamespace(
        biomass_links=pd.DataFrame({"biomass_node_id": ["B1"]}),
        biomass=pd.DataFrame({"biomass_node_id": ["B1"], "province_name": ["P"], "available_gj": [0.0]}),
    )
    year_data = SimpleNamespace(
        ammonia_links=pd.DataFrame({"ammonia_node_id": ["N1"]}),
        ammonia_nodes=pd.DataFrame({"ammonia_node_id": ["N1"], "province_name": ["P"]}),
        h2_available_kg=np.array([10.0 * NH3_H2_RATIO / AMMONIA_FLOW_SCALE]),
        industry_h2_node_membership=None,
        water_links=pd.DataFrame({"water_node_id": ["W1", "W2"]}),
        water_nodes=pd.DataFrame({"water_node_id": ["W1", "W2"], "province_name": ["P", "P"]}),
        water_available_m3=water_available_m3,
        water_basin_codes=basin_codes,
        water_basin_available_m3=basin_available_m3,
    )
    table = _build_supply_table(prepared, 2030, year_data, np.array([0.0]), np.array([5.0]),
                                np.array([3.0, 2.0e6]), basin_use_m3)
    return table.set_index(["resource_type", "region"])["utilization"]


def test_utilization_is_inf_where_nothing_is_available_but_something_is_used() -> None:
    utilization = _utilization(np.array([0.0, 4.0e6 / WATER_FLOW_SCALE]), ["C", "D"],
                               np.array([-1.0e6, -1.0e6]), np.array([5.0e6, 0.0]))
    assert utilization[("biomass", "B1")] == 0.0  # 可用量与用量都为零
    assert utilization[("green_h2", "N1")] == pytest.approx(0.5)  # 煤电氨 5 kg 折成氢，可用 10 kg 氨当量的氢
    assert np.isinf(utilization[("water", "W1")])  # 节点可用量为零而用了水（松弛）
    assert utilization[("water", "W2")] == pytest.approx(0.5)
    assert np.isinf(utilization[("water_basin_quota", "C")])  # 流域余量为负而仍有取水
    assert utilization[("water_basin_quota", "D")] == 0.0  # 流域余量为负而没有取水


def test_utilization_stays_zero_without_water_constraints() -> None:
    # water_mode="no_water"：节点可用量为 None（`water_access._water_access_data`），也没有流域行
    utilization = _utilization(None, [], None, np.array([]))
    assert utilization[("water", "W1")] == 0.0
    assert utilization[("water", "W2")] == 0.0
    assert "water_basin_quota" not in utilization.index.get_level_values("resource_type")


def test_green_h2_node_use_adds_industry_hydrogen_to_coal_ammonia_converted_to_hydrogen() -> None:
    """绿氢节点的用量 = 煤电氨 x NH3_H2_RATIO + 工业氢（与节点约束同口径），利用率按绿氢可用量算。"""
    from scipy import sparse

    prepared = SimpleNamespace(
        biomass_links=pd.DataFrame({"biomass_node_id": ["B1"]}),
        biomass=pd.DataFrame({"biomass_node_id": ["B1"], "province_name": ["P"], "available_gj": [0.0]}),
    )
    year_data = SimpleNamespace(
        ammonia_links=pd.DataFrame({"ammonia_node_id": ["N1"]}),
        ammonia_nodes=pd.DataFrame({"ammonia_node_id": ["N1"], "province_name": ["P"]}),
        h2_available_kg=np.array([4.0 / AMMONIA_FLOW_SCALE]),
        industry_h2_node_membership=sparse.csr_matrix(np.array([[1.0]])),
        water_links=pd.DataFrame({"water_node_id": ["W1"]}),
        water_nodes=pd.DataFrame({"water_node_id": ["W1"], "province_name": ["P"]}),
        water_available_m3=None, water_basin_codes=[], water_basin_available_m3=None,
    )
    table = _build_supply_table(prepared, 2030, year_data, np.array([0.0]), np.array([5.0]), np.array([0.0]),
                                np.array([]), industry_h2_flow_kg=np.array([1.1]))
    row = table.set_index(["resource_type", "region"]).loc[("green_h2", "N1")]
    assert row["used"] == pytest.approx(5.0 * NH3_H2_RATIO + 1.1)
    assert row["utilization"] == pytest.approx((5.0 * NH3_H2_RATIO + 1.1) / 4.0)
