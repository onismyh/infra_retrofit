"""水输入表的构建流程（`write_water_inputs`）：成员表、粗化到 2° 节点。厂址到节点的供水链路求解时现算（`optimization.data_prep._prepare_water`）。
0.5° 网格节点与按流域预算并偏差校正的节点可用量在 `water_availability`，网格与流域的共用工具在 `water_grid`，
偏差校正在 `water_bias`。
"""
from __future__ import annotations

import re

import pandas as pd

from ..artifacts import write_csv
from ..constants import WATER_COARSE_GRID_DEGREES
from ..paths import ProjectPaths
from ..spatial import bin_index
from .water_availability import build_water_availability_dataframe, build_water_nodes_dataframe

WATER_PATTERN = re.compile(
    r"(?P<hydrology>[^_]+)_(?P<gcm>.+?)_w5e5_(?P<ssp>ssp\d+)_"
    r"(?P<socioeconomics>.+?)_default_(?P<variable>[^_]+)_global_monthly_"
    r"(?P<start_year>\d{4})_(?P<end_year>\d{4})\.nc"
)


def _scenario_family(ssp: str) -> str:
    if ssp == "ssp126":
        return "baseline"
    if ssp == "ssp370":
        return "high_pressure"
    return "other"


def parse_water_file(paths: ProjectPaths, filename: str) -> dict[str, object]:
    match = WATER_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Could not parse water scenario filename: {filename}")
    scenario_id = "|".join([match.group("hydrology"), match.group("gcm"), match.group("ssp")])
    path = paths.data_dir / "water" / filename
    return {
        "scenario_id": scenario_id,
        "hydrology_model": match.group("hydrology"),
        "gcm": match.group("gcm"),
        "ssp": match.group("ssp"),
        "scenario_family": _scenario_family(match.group("ssp")),
        "socioeconomics": match.group("socioeconomics"),
        "variable": match.group("variable"),
        "frequency": "monthly",
        "start_year": int(match.group("start_year")),
        "end_year": int(match.group("end_year")),
        "source": paths.rel(path),
        "year_basis": f"{match.group('start_year')}-{match.group('end_year')}",
    }


def build_water_scenarios_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    """只收情景成员，一个成员一个 qtot 文件。`historical` 运行与它们同在一个目录，但不是成员：
    它们是 `basin_bias_factors` 估计各模型偏差时所对照的基准。同目录的耗水文件（`WATER_USE_VARIABLES`）
    也匹配 `WATER_PATTERN`，按变量名排除。"""
    files = sorted(
        path.name for path in (paths.data_dir / "water").glob("*.nc")
        if (match := WATER_PATTERN.fullmatch(path.name)) and match.group("variable") == "qtot"
    )
    if not files:
        raise FileNotFoundError(f"No scenario .nc files matching WATER_PATTERN in {paths.data_dir / 'water'}")
    scenarios = pd.DataFrame(parse_water_file(paths, filename) for filename in files)
    return scenarios.sort_values(["hydrology_model", "gcm", "ssp"]).reset_index(drop=True)


def coarsen_water_inputs(
    fine_nodes: pd.DataFrame,
    fine_availability: pd.DataFrame,
    cell_degrees: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把水文网格聚合到更粗的网格上，供优化使用。

    网格按 (流域, 经度分箱, 纬度分箱) 分组，而不是只按分箱分组。只按分箱分组时，一个
    ~200 km 的网格会横跨多个预算单元，并被整体划给其最大节点所在的单元，从而把水量
    搬过单元边界：在早先的分省预算下，这使天津 +347%、北京 +107%、宁夏 +50%，而陕西
    减少 40%、山东减少 33%、河北减少 25%。由于可用水量预算是按流域构建的，这会悄悄
    改写模型本来要检验的那条约束。按流域拆分每个网格，可使粗化前后的流域总量保持一致。
    """
    nodes = fine_nodes.copy()
    nodes["_ci"] = bin_index(nodes["longitude"].astype(float), cell_degrees)
    nodes["_cj"] = bin_index(nodes["latitude"].astype(float), cell_degrees)
    nodes["_bid"] = nodes["basin_code"].astype(str) if "basin_code" in nodes.columns else "X"
    nodes["coarse_id"] = [
        f"WC_{ci:04d}_{cj:04d}_{bid}"
        for ci, cj, bid in zip(nodes["_ci"], nodes["_cj"], nodes["_bid"])
    ]

    # 粗节点坐标按各细节点的模拟径流（`local_runoff_m3_per_year`，所有成员与规划年的均值）加权：径流不经偏差校正，
    # 偏差因子改了坐标也不动，`water_nodes.csv` 与供水距离不随之变（作者决定 2026-10-12；此前按校正后的可用量加权）。
    weight = (
        fine_availability.groupby("water_node_id")["local_runoff_m3_per_year"].mean().rename("_w")
    )
    nodes = nodes.merge(weight, left_on="water_node_id", right_index=True, how="left")
    nodes["_w"] = nodes["_w"].fillna(0.0)

    # 粗节点的 `basin_code` 是分组用的流域，可用量就在这个流域的预算里分；节点余量扣生活与灌溉耗水时也按这一列
    # （可用量表的同名列）归流域。加权坐标可能落在相邻流域的多边形里（2026-10-13 的 373 个节点中有 5 个），不按坐标重新归属。
    rows: list[dict[str, object]] = []
    for coarse_id, group in nodes.groupby("coarse_id"):
        total = float(group["_w"].sum())
        if total > 0:
            lon = float((group["_w"] * group["longitude"]).sum() / total)
            lat = float((group["_w"] * group["latitude"]).sum() / total)
        else:
            lon = float(group["longitude"].mean())
            lat = float(group["latitude"].mean())
        rows.append(
            {
                "water_node_id": str(coarse_id),
                "province_name": str(group["province_name"].iloc[0]),
                "basin_code": str(group["_bid"].iloc[0]),
                "longitude": lon,
                "latitude": lat,
                "source": "data/water/*_qtot_*.nc",
                "year_basis": "static_hydrology_grid",
            }
        )
    coarse_nodes = pd.DataFrame(rows).sort_values("water_node_id").reset_index(drop=True)

    mapping = dict(zip(nodes["water_node_id"], nodes["coarse_id"]))
    availability = fine_availability.copy()
    availability["water_node_id"] = availability["water_node_id"].map(mapping)
    # scenario_id 必须留在分组键里：否则同一情景族（family）的所有气候成员会被加总
    # 成一个节点总量。
    keys = ["water_node_id", "planning_year", "scenario_family", "scenario_id",
            "hydrology_model", "gcm", "ssp", "basin_code"]
    keys = [key for key in keys if key in availability.columns]
    coarse_availability = availability.groupby(keys, as_index=False).agg(
        available_water_m3_per_year=("available_water_m3_per_year", "sum"),
        dry_season_water_m3_per_year=("dry_season_water_m3_per_year", "sum"),
        local_runoff_m3_per_year=("local_runoff_m3_per_year", "sum"),
        bias_factor=("bias_factor", "first"),
        bias_source=("bias_source", "first"),
    )
    coarse_availability["proxy_type"] = "basin_budgeted_bias_corrected_local_runoff"
    coarse_availability["unit"] = "m3/yr"
    coarse_availability["source"] = "data/water/*_qtot_*.nc"
    coarse_availability["year_basis"] = "decadal window"
    return coarse_nodes, coarse_availability


def write_water_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = build_water_scenarios_dataframe(paths)
    scenarios = scenarios[scenarios["ssp"].astype(str).str.startswith("ssp")].reset_index(drop=True)
    water_nodes = build_water_nodes_dataframe(paths, scenarios)
    water_availability = build_water_availability_dataframe(paths, scenarios, water_nodes)
    # 粗化以前是在构建之后手工施加的，所以已提交的输入无法从这个入口重新生成。
    # 它应当放在这里。
    if WATER_COARSE_GRID_DEGREES > 0:
        water_nodes, water_availability = coarsen_water_inputs(
            water_nodes, water_availability, WATER_COARSE_GRID_DEGREES
        )

    active_node_ids = set(
        water_availability.loc[
            water_availability["available_water_m3_per_year"].astype(float) > 0.0,
            "water_node_id",
        ].astype(str)
    )
    if active_node_ids:
        water_nodes = water_nodes[water_nodes["water_node_id"].astype(str).isin(active_node_ids)].reset_index(drop=True)
        water_availability = water_availability[
            water_availability["water_node_id"].astype(str).isin(active_node_ids)
        ].reset_index(drop=True)

    write_csv(scenarios, paths.inputs_dir / "water_scenarios.csv")
    write_csv(water_nodes, paths.inputs_dir / "water_nodes.csv")
    write_csv(water_availability, paths.inputs_dir / "water_availability.csv")
    return scenarios, water_nodes, water_availability
