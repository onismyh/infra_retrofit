"""水链路与流域取水下限的两道前置检查（2026-10-16）。

1. 厂侧水平衡在所有情景里都建（`model_resources.add_resource_balances`）：半径内没有水节点的 hub 耗水被迫为 0、
   只剩退役可行，且不报错。`data_prep._prepare_water` 现在直接拒绝这样的输入。
2. 流域余量低于"工业最少取水 + 煤电全部改空冷（不退役）的取水"时，硬约束只能靠退役该流域的煤电满足；
   `water_access.warn_basin_floor` 建模前逐流域告警（PR #68 之前 `_indtree` 的西北诸河 K 2030 年就是这种情形；现行输入不触发，
   下面的用例用合成数）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants import WATER_MATCH_BUFFER_KM
from coal_retrofit.optimization.assumptions import OptimizationAssumptions
from coal_retrofit.optimization.data_prep import _prepare_water
from coal_retrofit.optimization.water_access import basin_withdrawal_floor, warn_basin_floor
from toy_inputs import _write_toy_inputs


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


def test_basin_floor_sums_coal_all_air_and_industry_least_water_route() -> None:
    # 两座厂、两个流域；路径列序同 PATHWAYS：unabated 列的发电量与空冷取水强度决定煤电下限。
    generation = np.zeros((2, 6))
    generation[:, 0] = [1.0e6, 2.0e6]
    withdrawal = np.full((2, 6), 2.0)
    air = np.full((2, 6), 0.2)
    membership = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 工业：第 1 个 hub 氢路线可用且取水更少，第 2 个 hub 只有不改造。
    industry_water = np.array([[1.0e6, 1.2e6, 0.6e6], [3.0e6, 3.5e6, 1.0e6]])
    available = np.array([[True, True, True], [True, False, False]])
    industry_membership = np.array([[1.0, 1.0], [0.0, 0.0]])
    coal_floor, industry_floor = basin_withdrawal_floor(
        generation, withdrawal, air, membership, industry_water, available, industry_membership
    )
    assert coal_floor.tolist() == pytest.approx([0.2e6, 0.4e6])
    assert industry_floor.tolist() == pytest.approx([0.6e6 + 3.0e6, 0.0])


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
