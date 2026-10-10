"""水文边界条件的入口：`<树>/inputs/water_scenarios.csv` 登记的气候成员（水文模型 × GCM × SSP），选成员、核数据。

水文是求解的边界条件：节点可用水量取所选成员、所选季节的径流，扣同一成员同一季节的生活与灌溉耗水
（`optimization.water_access._water_available_by_node`）。成员与季节是情景字段 `water_scenario_id`、`water_season`；
命令行 `run <情景> --water 水文模型/GCM/SSP [--season annual|dry]` 把它们（无水约束的基底另加
`water_mode = "grid_supply"`）并到登记情景上，结果名按成员与季节自动拼（`water_overrides`）。

`check_member` 核一个成员的数据：每个规划年每个水节点恰好一行、有流域码，径流非空、非负；节点所在的每个流域都有
该年的生活与灌溉耗水，只有一行，非空、非负。给了季节就只核模型这个季节读的列，偏差因子只在关掉偏差校正时核（要为正）。
模型对缺行或径流为空的节点静默补零（`water_access` 按节点 reindex 后填 0），所以求解之前就查（`runner.run`），
`water-members` 两个季节一起逐个成员查。耗水表的空值与重复，模型读表时自己也整表查（`data_prep._prepare_basin_use`，
报错而不补零）。只读三张输入表，不建模。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .optimization.scenario import OptimizationScenario
from .scenarios import ScenarioRegistryError, _suggest, shown_path

MEMBERS_FILE = "water_scenarios.csv"
# 模型按季节读的列（`water_access._water_available_by_node`）：径流一列，生活与灌溉耗水两列。
_SEASON_COLUMNS: dict[str, tuple[str, tuple[str, str]]] = {
    "annual": ("available_water_m3_per_year", ("domestic_m3_per_year", "irrigation_m3_per_year")),
    "dry": ("dry_season_water_m3_per_year", ("dry_season_domestic_m3_per_year", "dry_season_irrigation_m3_per_year")),
}


def load_members(tree: Path) -> pd.DataFrame:
    """成员表 `<树>/inputs/water_scenarios.csv`（全部读成字符串），按表内顺序。"""
    path = Path(tree) / "inputs" / MEMBERS_FILE
    if not path.exists():
        raise ScenarioRegistryError(f"{shown_path(path)} 不存在：成员表由 builders.water 与可用水量表一起写出")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def water_arg(member: str) -> str:
    """成员 id `水文模型|GCM|SSP` 写成命令行 `--water` 的形式 `水文模型/GCM/SSP`。"""
    return member.replace("|", "/")


def member_id(text: str, members: pd.DataFrame) -> str:
    """`--water` 的 `水文模型/GCM/SSP`（也认 `|`）→ 成员 id；成员表里没有就报错。"""
    wanted = text.replace("/", "|")
    ids = list(members["scenario_id"])
    if wanted not in ids:
        close = _suggest(wanted, ids)
        raise ScenarioRegistryError(
            f"--water {text!r}：{MEMBERS_FILE} 里没有成员 {water_arg(wanted)!r}"
            f"{close.replace('|', '/')}（python -m coal_retrofit water-members 列出全部）"
        )
    return wanted


def water_overrides(
    spec_scenario: Mapping[str, Any], member: str | None, season: str | None,
) -> tuple[dict[str, Any], str]:
    """`--water` 成员与 `--season` → (scenario 节的字段覆盖, 结果名后缀 `_水文模型_GCM_SSP_季节`)。

    后缀里成员 id 原样，季节是生效的 `water_season`（不给 `--season` 时取基底的）。基底无水约束
    （`water_mode = "no_water"`）时另设 `water_mode = "grid_supply"`（与 `ST_WA_cwatm_126_dry_oq` 同）。
    `--season` 只配 `--water`：只换季节的结果名分不出成员。
    """
    if member is None:
        if season is not None:
            raise ScenarioRegistryError("--season 只配 --water：结果名按成员与季节拼，单给季节分不出是哪个成员")
        return {}, ""
    overrides: dict[str, Any] = {}
    if spec_scenario.get("water_mode", OptimizationScenario.water_mode) == "no_water":
        overrides["water_mode"] = "grid_supply"
    overrides["water_scenario_id"] = member
    if season is not None:
        overrides["water_season"] = season
    effective = overrides.get("water_season", spec_scenario.get("water_season", OptimizationScenario.water_season))
    return overrides, f"_{member.replace('|', '_')}_{effective}"


def check_member(tree: Path, member: str, years: Sequence[int], season: str | None = None,
                 check_bias: bool = True) -> list[str]:
    """一个成员在 *years* 各年的数据缺漏，逐条中文描述；空列表即齐全。

    *season* 为 None 时两个季节的列都核；*check_bias* 为假（开着偏差校正，模型不读偏差因子）时不核偏差因子。
    """
    return _problems(member, years, (season,) if season else tuple(_SEASON_COLUMNS), check_bias, *_tables(Path(tree)))


def check_members(tree: Path, years: Sequence[int]) -> dict[str, list[str]]:
    """成员表里每个成员的 `check_member`（两个季节、含偏差因子），按表内顺序。"""
    members = list(load_members(tree)["scenario_id"])
    tables = _tables(Path(tree))
    return {member: _problems(member, years, tuple(_SEASON_COLUMNS), True, *tables) for member in members}


def _tables(tree: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]:
    """水节点 id、可用水量表、耗水表（没有这个文件时为 None）。"""
    inputs = tree / "inputs"
    nodes = pd.read_csv(inputs / "water_nodes.csv", encoding="utf-8-sig")["water_node_id"].astype(str)
    availability = pd.read_csv(inputs / "water_availability.csv", encoding="utf-8-sig")
    use_path = inputs / "water_basin_use.csv"
    use = pd.read_csv(use_path, encoding="utf-8-sig") if use_path.exists() else None
    return nodes, availability, use


def _problems(member: str, years: Sequence[int], seasons: Sequence[str], check_bias: bool, nodes: pd.Series,
              availability: pd.DataFrame, use: pd.DataFrame | None) -> list[str]:
    runoff_columns = [_SEASON_COLUMNS[season][0] for season in seasons]
    use_columns = [column for season in seasons for column in _SEASON_COLUMNS[season][1]]
    bias = ["bias_factor"] if check_bias else []
    missing = sorted({"water_node_id", "scenario_id", "planning_year", "basin_code", *runoff_columns, *bias}
                     - set(availability.columns))
    if missing:
        return [f"water_availability.csv 缺列 {missing}"]
    if use is None:
        return ["没有 water_basin_use.csv（scripts/build_water_use.py 生成）"]
    missing = sorted({"scenario_id", "planning_year", "basin_code", *use_columns} - set(use.columns))
    if missing:
        return [f"water_basin_use.csv 缺列 {missing}"]
    problems: list[str] = []
    node_set = set(nodes)
    rows = availability[availability["scenario_id"].astype(str) == member]
    uses = use[use["scenario_id"].astype(str) == member]
    for year in years:
        here = rows[rows["planning_year"].astype(int) == int(year)]
        if here.empty:
            problems.append(f"{year}：可用水量表没有行")
            continue
        ids = here["water_node_id"].astype(str)
        _note(problems, year, "个水节点没有可用水量", sorted(node_set - set(ids)))
        _note(problems, year, "个水节点不在 water_nodes.csv", sorted(set(ids) - node_set))
        _note(problems, year, "个水节点重复", sorted(set(ids[ids.duplicated()])))
        for column in [*runoff_columns, *bias]:
            values = pd.to_numeric(here[column], errors="coerce").to_numpy(dtype=np.float64)
            positive = column == "bias_factor"
            bad = ~np.isfinite(values) | ((values <= 0) if positive else (values < 0))
            _note(problems, year, f"个水节点的 {column} 为空、非数或{'不为正' if positive else '为负'}", sorted(ids[bad]))
        basins = here["basin_code"]
        _note(problems, year, "个水节点没有流域码", sorted(ids[basins.isna()]))
        year_use = uses[uses["planning_year"].astype(int) == int(year)]
        codes = year_use["basin_code"].astype(str)
        _note(problems, year, "个流域没有生活与灌溉耗水", sorted(set(basins.dropna().astype(str)) - set(codes)))
        _note(problems, year, "个流域的耗水重复", sorted(set(codes[codes.duplicated()])))
        values = year_use[use_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        bad = (~np.isfinite(values) | (values < 0)).any(axis=1)
        _note(problems, year, "个流域的耗水为空、非数或为负", sorted(codes[bad]))
    return problems


def _note(problems: list[str], year: int, what: str, items: Sequence[str]) -> None:
    if items:
        problems.append(f"{year}：{len(items)} {what}（如 {', '.join(items[:3])}）")
