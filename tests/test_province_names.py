"""2026-09-25：输入表里的省名在读入时按 `PROVINCE_NAME_ALIASES` 换成分省煤价表的写法，仍查不到的告警。

仓库根 `inputs/industry_hubs.csv` 把内蒙古写作拼音 "Neimenggu"，煤价表里是 "Inner Mongolia"；此前查不到就
静默退回缺省煤价 `coal_fuel_cost_cny_per_gj`，14 个水泥 hub 的捕集蒸汽按 38.2 而不是 17.9 元/GJ 计价。
这些测试不求解，不依赖 Gurobi。
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from coal_retrofit.constants_industry import capture_variable_cost_cny_per_t
from coal_retrofit.optimization.data_prep import _prepare_plants
from coal_retrofit.optimization.industry import CCS, industry_year_data
from coal_retrofit.optimization.industry_inputs import prepare_industry
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.paths import ProjectPaths

_SCENARIO = OptimizationScenario(experiment_id="T", description="toy")


def _cement_hubs(root, provinces: list[str]) -> ProjectPaths:
    """每个省名一个同样大小的水泥 hub，外加 `prepare_industry` 要读的一行氢供给曲线。"""
    paths = ProjectPaths(root=root)
    paths.ensure_inputs_dir()
    n = len(provinces)
    pd.DataFrame({
        "hub_id": [f"C{i}" for i in range(n)], "sector": ["cement"] * n, "province": provinces,
        "longitude": [112.0] * n, "latitude": [37.0] * n,
        "production_kt_per_year": [1000.0] * n, "co2_mt_per_year": [0.8] * n,
        "process_co2_mt_per_year": [0.5] * n, "h2_demand_kt_per_year": [0.0] * n,
        "water_m3_per_year": [1.0e6] * n,
    }).to_csv(paths.inputs_dir / "industry_hubs.csv", index=False)
    pd.DataFrame({
        "year": [2030], "h2_supply_kg_per_year": [1.0e9], "weighted_lcoh_usd_per_kg_h2": [3.0],
    }).to_csv(paths.inputs_dir / "ammonia_supply_curve.csv", index=False)
    return paths


def _coal_price_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """读入处发出的"查不到煤价"告警。"""
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING and "no coal price" in record.getMessage()
    ]


def test_neimenggu_hubs_are_read_as_inner_mongolia_and_price_steam_alike(tmp_path, caplog) -> None:
    """"Neimenggu" 读入后写作 "Inner Mongolia"，两种写法的水泥 hub 按同一个煤价（17.9 元/GJ）计捕集能耗，
    不告警。2026-09-25 前前者按缺省的 38.2 计。"""
    paths = _cement_hubs(tmp_path, ["Neimenggu", "Inner Mongolia"])
    assumptions = OptimizationAssumptions()
    with caplog.at_level(logging.WARNING):
        industry = prepare_industry(paths, assumptions)
    assert list(industry.hubs["province"]) == ["Inner Mongolia", "Inner Mongolia"]
    assert _coal_price_warnings(caplog) == []
    data = industry_year_data(industry, _SCENARIO, assumptions, 2030)
    assert data.opex_cny[0, CCS] == pytest.approx(data.opex_cny[1, CCS], rel=1e-12)


def test_unknown_province_warns_and_prices_steam_at_the_default_coal_price(tmp_path, caplog) -> None:
    """查不到的省名告警，并按缺省煤价 `coal_fuel_cost_cny_per_gj` 计捕集能耗：与同样大小、在内蒙古的
    hub 相比，年成本正好多出捕集量 × 两个煤价下的单位能耗差。"""
    paths = _cement_hubs(tmp_path, ["Inner Mongolia", "Atlantis"])
    assumptions = OptimizationAssumptions()
    with caplog.at_level(logging.WARNING):
        industry = prepare_industry(paths, assumptions)
    warnings = _coal_price_warnings(caplog)
    assert len(warnings) == 1
    assert "Atlantis" in warnings[0] and "Inner Mongolia" not in warnings[0]
    data = industry_year_data(industry, _SCENARIO, assumptions, 2030)
    elec = _SCENARIO.electricity_price_for_year(2030)
    gap_per_t = capture_variable_cost_cny_per_t(
        "cement", assumptions.coal_fuel_cost_cny_per_gj, elec
    ) - capture_variable_cost_cny_per_t("cement", assumptions.province_coal_cost("Inner Mongolia"), elec)
    assert gap_per_t > 0.0
    captured_t = data.captured_mt[0, CCS] * 1e6
    assert data.opex_cny[1, CCS] - data.opex_cny[0, CCS] == pytest.approx(captured_t * gap_per_t, rel=1e-9)


def test_plant_province_names_get_the_same_alias_and_warning(tmp_path, caplog) -> None:
    """煤电侧同样换写法、同样告警（现有 plants.csv 的省名全部查得到，这里是防护）。"""
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_inputs_dir()
    pd.DataFrame({
        "plant_id": ["P1", "P2"], "province_mode": ["Neimenggu", "Atlantis"],
        "total_capacity_mw": [1000.0, 1000.0], "dominant_cooling_technology": ["recirculating", "recirculating"],
    }).to_csv(paths.inputs_dir / "plants.csv", index=False)
    with caplog.at_level(logging.WARNING):
        plants = _prepare_plants(paths, _SCENARIO, OptimizationAssumptions())
    assert list(plants["province_name"]) == ["Inner Mongolia", "Atlantis"]
    warnings = _coal_price_warnings(caplog)
    assert len(warnings) == 1 and "Atlantis" in warnings[0]
