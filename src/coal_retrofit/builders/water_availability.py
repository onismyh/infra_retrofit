"""水节点与节点可用量：省界内的 0.5° 水文网格中心作节点并归一级流域（`build_water_nodes_dataframe`），
各成员、各规划年的径流按流域预算、乘偏差因子后按本地径流比例分到节点（`build_water_availability_dataframe`）。
粗化、供水链路与写盘流程在 `water`。
"""
from __future__ import annotations

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

from ..constants import PLANNING_YEARS, TARGET_GEO_CRS
from ..paths import ProjectPaths
from .water_bias import member_bias_factors
from .water_grid import (
    WATER_WINDOW_BASIS,
    _annual_and_seasonal,
    _basin_dry_window,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_window,
    _file_years,
    _resolve_source_path,
    assign_basin_codes,
    load_provinces,
)


def _reference_scenario_row(scenarios: pd.DataFrame) -> pd.Series:
    """节点网格所参照的成员：ssp126 成员里按 (水文模型, GCM) 排序的第一个。

    成员表须同时含 ssp126 与 ssp370（两个 `ST_` 水文情景族），缺一个就报错，不拿单一 SSP 的成员表建节点。
    """
    for ssp, role in (("ssp126", "baseline"), ("ssp370", "high-pressure")):
        if not (scenarios["ssp"] == ssp).any():
            raise ValueError(f"No {ssp} water scenarios available for {role} proxy")
    return scenarios[scenarios["ssp"] == "ssp126"].sort_values(["hydrology_model", "gcm"]).iloc[0]


def _candidate_grid_points(
    provinces: gpd.GeoDataFrame,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = provinces.total_bounds
    lon_mask = (lon_values >= (min_lon - 0.5)) & (lon_values <= (max_lon + 0.5))
    lat_mask = (lat_values >= (min_lat - 0.5)) & (lat_values <= (max_lat + 0.5))
    lon_indices = np.flatnonzero(lon_mask)
    lat_indices = np.flatnonzero(lat_mask)
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    frame = pd.DataFrame(
        {
            "lon_index": lon_grid.ravel().astype(np.int32),
            "lat_index": lat_grid.ravel().astype(np.int32),
        }
    )
    frame["longitude"] = lon_values[frame["lon_index"].to_numpy()].astype(np.float64)
    frame["latitude"] = lat_values[frame["lat_index"].to_numpy()].astype(np.float64)
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs=TARGET_GEO_CRS,
    )
    joined = gpd.sjoin(
        points,
        provinces[["province_id", "province_name", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    joined = (
        joined.sort_values(["lat_index", "lon_index", "province_id"])
        .drop_duplicates(subset=["lat_index", "lon_index"])
        .reset_index(drop=True)
    )
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def build_water_nodes_dataframe(paths: ProjectPaths, scenarios: pd.DataFrame) -> pd.DataFrame:
    provinces = load_provinces(paths)
    reference_row = _reference_scenario_row(scenarios)
    reference_path = _resolve_source_path(paths, str(reference_row["source"]))
    with netCDF4.Dataset(str(reference_path), "r") as ds:
        lon_values = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat_values = np.asarray(ds.variables["lat"][:], dtype=np.float64)

    nodes = _candidate_grid_points(provinces, lon_values, lat_values)
    nodes["basin_code"] = assign_basin_codes(paths, nodes)
    lon_step = float(np.median(np.abs(np.diff(lon_values)))) if len(lon_values) > 1 else 0.5
    lat_step = float(np.median(np.abs(np.diff(lat_values)))) if len(lat_values) > 1 else 0.5
    nodes["water_node_id"] = [f"W{index:05d}" for index in range(1, len(nodes) + 1)]
    nodes["grid_cell_width_deg"] = lon_step
    nodes["grid_cell_height_deg"] = lat_step
    nodes["competition_scope"] = "shared_water_node"
    nodes["match_rule"] = "china_hydrology_grid_cell_center"
    nodes["source"] = str(reference_row["source"])
    nodes["year_basis"] = "static_hydrology_grid"
    return nodes[
        [
            "water_node_id",
            "province_id",
            "province_name",
            "basin_code",
            "longitude",
            "latitude",
            "lon_index",
            "lat_index",
            "grid_cell_width_deg",
            "grid_cell_height_deg",
            "competition_scope",
            "match_rule",
            "source",
            "year_basis",
        ]
    ].sort_values("water_node_id").reset_index(drop=True)


def build_water_availability_dataframe(
    paths: ProjectPaths,
    scenarios: pd.DataFrame,
    water_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """到达每个节点的可再生水量，取自本地径流（ISIMIP `qtot`）。

    旧实现读取每个节点所在网格的 `dis`（经汇流演算的河道流量）。流量是累积量——下游
    网格承载其整个上游集水区的来水——所以对 293 个节点求和会把同一份水重复计算几十次：
    全国总量达到 430 766 x10^8 m3/yr，是中国实际可再生水资源量的 15x，由此得到的约束
    永远不会起作用（需求/供给 = 0.083%）。

    `qtot` 是本地产生的径流（kg m-2 s-1），因此对网格求和就是真实的水量预算。按省界
    掩膜后，它复现出 26 291 x10^8 m3/yr，对照官方多年平均 ~28 000，相差 6%。

    可用水量按水资源一级区做预算，再按各节点本地径流的比例分配到该流域的节点上，因此
    对所有节点求和，每个流域的可再生水量恰好只计一次：

        basin_runoff   = sum(qtot x cell_area x seconds_per_year) x bias_factor   [m3/yr]
        node_available = basin_runoff x node_runoff / sum(node_runoff in basin)

    `bias_factor` 列记该成员在该流域的因子，`bias_source` 列记它来自哪个历史文件或哪些成员的均值（`member_bias_factors`）。

    这里以流域而不是省为单元，理由有二：水是流域量（黄河流经九个省），而流域也是全球
    水文模型率定所用的尺度、中国公布官方总量所用的尺度。按省做预算还会使与官方统计之比
    落在 0.34x 到 4.00x 之间，其中大部分是用省界切割 0.5 deg 网格造成的假象；按流域时
    离散范围为 0.74x-2.43x，而 `basin_bias_factors` 连这一点也消除了。

    环境流量、其他用户耗水与存量取水都不在这里施加：环境流量是求解时乘的可提取比例（`WATER_EXTRACTABLE_FRACTION`），
    生活与灌溉耗水求解时从节点余量里扣（`water_basin_use.csv`，由 `build_water_use_dataframe` 生成），
    存量取水在流域取水上限里扣除（`water_basin_caps.csv`，由 `scripts/build_water_basin_caps.py` 生成），
    因此改这三项不必重建本表。

    同时生成年均列和枯水期列（最低的连续三个月，年化）：火电受限发生在低流量时段，
    而不是在年均水平上，且源数据是逐月的。

    枯水季是在流域汇总量上选取的——对流域径流总和的 12 个候选 3 个月窗口取最小值——
    而不是逐网格选取。逐网格选取再求和得到的是 sum(min) 而不是 min(sum)，会把流域
    枯水期流量在海河低估 2.25x、在西北诸河低估 2.13x。见选取处的注释块。

    这一列不代表什么。`qtot` 是未经汇流演算的产流量，所以枯水期数值是流域最枯一季的
    产水速率——而不是河道中可用水的速率。水库调节体现在汇流方案里，而本流程刻意不用
    汇流（见开头一段）。因此把它当作保证供水量（firm yield）预算使用，就等于假设零调蓄；
    这在海河、黄河和淮河是很严格的假设，因为那里的枯水期流量很大程度上是水库放水决策
    的结果。在此写明，是因为由此得到的约束是本研究最紧的约束，其严格程度既来自水文，
    同样也来自这一选择。
    """
    rows: list[dict[str, object]] = []
    lat_indices = water_nodes["lat_index"].astype(int).to_numpy()
    lon_indices = water_nodes["lon_index"].astype(int).to_numpy()
    node_ids = water_nodes["water_node_id"].astype(str).to_numpy()
    node_basins = water_nodes["basin_code"].astype(str).to_numpy()
    bias_cache: dict[tuple[str, str], tuple[dict[str, float], str]] = {}

    for _, scenario_row in scenarios.iterrows():
        scenario_path = _resolve_source_path(paths, str(scenario_row["source"]))
        with netCDF4.Dataset(str(scenario_path), "r") as ds:
            years = _file_years(ds)
            if "qtot" not in ds.variables:
                raise ValueError(
                    f"{scenario_path} has no 'qtot' variable; routed discharge ('dis') is not a "
                    "valid water budget — see this function's docstring."
                )
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            area_m2 = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
            basin_zones, basin_codes = _basin_zone_grid(paths, lat, lon)
            hydrology_model = str(scenario_row["hydrology_model"])
            gcm = str(scenario_row["gcm"])
            # 每个 (水文模型, GCM) 一套因子，各 SSP 共用；一个水文模型的各 GCM 一次算齐，
            # 因为没有真实历史运行的成员要取其余成员的均值（`member_bias_factors`）。
            bias_key = (hydrology_model, gcm)
            if bias_key not in bias_cache:
                model_gcms = sorted(
                    set(scenarios.loc[scenarios["hydrology_model"].astype(str) == hydrology_model, "gcm"].astype(str))
                )
                member_factors = member_bias_factors(paths, hydrology_model, model_gcms, basin_zones, basin_codes, area_m2)
                bias_cache.update({(hydrology_model, member_gcm): value for member_gcm, value in member_factors.items()})
            bias_factors, bias_source = bias_cache[bias_key]

            for planning_year in PLANNING_YEARS:
                window_start, window_end = WATER_WINDOW_BASIS[planning_year]
                mask = (years >= window_start) & (years <= window_end)
                if not mask.any():
                    continue
                window = _clean_window(ds, "qtot", mask)
                # 枯水期：气候态年循环中最低的连续 3 个日历月。`rolling_volume` 是 12 个候选
                # 3 个月窗口各自的每网格水量；流域的枯水季只选一次，且是在流域总量上选——见下面的循环。
                annual, rolling_volume = _annual_and_seasonal(window, area_m2)

                node_annual = annual[lat_indices, lon_indices]

                # 流域预算：先做偏差校正，再分配到该流域的各节点。
                node_annual_out = np.zeros(len(node_ids), dtype=np.float64)
                node_dry_out = np.zeros(len(node_ids), dtype=np.float64)
                for code in set(node_basins):
                    members = node_basins == code
                    zone_idx = basin_codes.index(code) + 1 if code in basin_codes else 0
                    cells = basin_zones == zone_idx
                    # 季节性沿用模型自身的；只校正水平。
                    modelled_annual = float(annual[cells].sum())
                    total_annual = modelled_annual * bias_factors.get(code, 1.0)
                    # 枯水期是流域量，不是网格量。
                    # 旧实现取 `rolling.min(axis=0)`——逐网格取最小值，每个网格各自挑选
                    # 自己的枯水季——再把这些互相独立的最小值在流域内求和。那是 sum(min)，
                    # 而约束需要的是 min(sum)：流域在其自身枯水季内的总流量。由 Jensen
                    # 不等式，只要各网格在不同月份见底，两者就不相等，且方向始终相同；差距
                    # 最大的恰恰是本研究判为超限的那些流域（2021-2030 窗口，
                    # cwatm|gfdl-esm4|ssp126，比值 min(sum)/sum(min)）：
                    #     C 海河 2.25x   K 西北诸河 2.13x   D 黄河 1.47x   E 淮河 1.19x
                    #     H 珠江 1.03x   F 长江 1.06x   G 东南诸河 1.11x
                    # 逐网格的相位信息无论如何都保留不下来：往下四行，流域就坍缩成一个
                    # `dry_share`，再按年径流权重重新分配到节点，所以单个网格在哪个月
                    # 见底，在它被使用之后的下一条语句就被丢弃了。
                    # 耗水表（`build_water_use_dataframe`）取同一个窗口。
                    _, dry_volume = _basin_dry_window(rolling_volume, cells)
                    dry_share = dry_volume / modelled_annual if modelled_annual > 0 else 0.0
                    total_dry = total_annual * dry_share
                    weights = node_annual[members]
                    weight_sum = float(weights.sum())
                    if weight_sum > 0:
                        share = weights / weight_sum
                    else:  # 该流域的节点上没有模拟径流：均分
                        share = np.full(int(members.sum()), 1.0 / max(1, int(members.sum())))
                    node_annual_out[members] = total_annual * share
                    node_dry_out[members] = total_dry * share

                for idx in range(len(node_ids)):
                    rows.append(
                        {
                            "water_node_id": str(node_ids[idx]),
                            "planning_year": int(planning_year),
                            "scenario_family": str(scenario_row["scenario_family"]),
                            "scenario_id": str(scenario_row["scenario_id"]),
                            "hydrology_model": str(scenario_row["hydrology_model"]),
                            "gcm": str(scenario_row["gcm"]),
                            "ssp": str(scenario_row["ssp"]),
                            "available_water_m3_per_year": max(0.0, float(node_annual_out[idx])),
                            "dry_season_water_m3_per_year": max(0.0, float(node_dry_out[idx])),
                            "local_runoff_m3_per_year": max(0.0, float(node_annual[idx])),
                            "basin_code": str(node_basins[idx]),
                            "bias_factor": round(float(bias_factors.get(str(node_basins[idx]), 1.0)), 4),
                            "bias_source": bias_source,
                            "proxy_type": "basin_budgeted_bias_corrected_local_runoff",
                            "unit": "m3/yr",
                            "source": str(scenario_row["source"]),
                            "year_basis": f"{window_start}-{window_end}",
                        }
                    )

    return pd.DataFrame(rows).sort_values(
        ["water_node_id", "planning_year", "scenario_family"]
    ).reset_index(drop=True)
