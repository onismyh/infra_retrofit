"""从厂级点源库构建工业点源清单。

来源：点源库目录 `POINT_SOURCE_DIR`（环境变量 `COAL_RETROFIT_POINT_SOURCE_DIR`，缺省为作者本机的
`D:\\6. Transfer\\PhD_tht\\point_source\\`）——读其中钢铁、水泥、合成氨与甲醇三个工作簿，覆盖五个部门，各自都带
WGS84 坐标、产能、产量、CO2，以及（EAF 除外）一列氢需求。

该库中的单位并不统一，一旦弄错，整个工业部门会被悄无声息地按比例放缩。编写本模块前
已对照全国总量核验：

    钢铁      产量以 kt 计      CO2 以 Mt 计   H2_DMD 以 kt 计   （939 Mt 粗钢）
    其他      产量以万吨计      CO2 以 Mt 计   H2_DMD 以万吨计   （52 Mt NH3，1 269 Mt 熟料）
    水泥      产能以 t/d 计（不是万吨）——9 000 t/d x 300 d = 2.7 Mt/yr，与它自己的
              产量列吻合，单位正是靠这项核对确定下来的。

全部统一为：产能与产量用 kt/yr，CO2 用 Mt/yr，氢用 kt/yr。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import write_csv
from ..constants_industry import (
    DEFAULT_SECTOR_HUB_COUNTS,
    INDUSTRY_CONSUMPTION_SHARE,
    SECTOR_AMMONIA,
    SECTOR_CEMENT,
    SECTOR_HAS_H2_ROUTE,
    SECTOR_LABELS_ZH,
    SECTOR_METHANOL,
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
    water_quota,
)
from ..paths import ProjectPaths
# 与煤电机组共用同一套球面距离矩阵：两类 hub 必须在完全相同的几何上聚类，否则它们的 hub 规模不可比。
from ..spatial import cluster_labels

logger = logging.getLogger(__name__)

# 点源库不入库；缺省是作者本机的路径，别的机器用环境变量指过去。
POINT_SOURCE_DIR = Path(os.environ.get("COAL_RETROFIT_POINT_SOURCE_DIR", r"D:\6. Transfer\PhD_tht\point_source"))

WAN_TO_KT = 10.0        # 万吨 -> kt
CEMENT_KILN_DAYS = 300  # t/d -> kt/yr，数据源隐含的年运行天数惯例

CANONICAL_COLUMNS = [
    "source_id", "sector", "sector_zh", "plant_name", "province",
    "longitude", "latitude", "capacity_kt_per_year", "production_kt_per_year",
    "co2_mt_per_year", "process_co2_mt_per_year", "h2_demand_kt_per_year",
    "feedstock", "water_m3_per_year", "status",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """取某列的数值视图；该列不存在时返回全零。"""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _text(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    """取某列的字符串视图；该列不存在时返回常数序列。

    `DataFrame.get(col, default)` 返回的是裸的默认值而不是 Series，所以这里不能用它——
    这种不匹配不会报错，直到对它调用 `.astype` 时才暴露。
    """
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].astype(str).fillna(default)


def _read(sheet_file: str, sheet: str) -> pd.DataFrame:
    path = POINT_SOURCE_DIR / sheet_file
    if not path.exists():
        raise FileNotFoundError(f"point-source workbook not found: {path}")
    return pd.read_excel(path, sheet_name=sheet)


def _load_steel() -> pd.DataFrame:
    """高炉-转炉（BOF）与电炉（EAF）装置。两张表，有意保留为两个部门。

    二者是不同的工艺：取水定额不同，排放因子不同（1.8 vs 0.4 tCO2/t 粗钢），两者之间
    也只有一条氢路线；把它们并成一个"钢铁"部门，会让下游每一个吨产品指标都变成两种
    不可比事物的混合。
    """
    frames = []
    bof = _read("steel_all.xlsx", "China_ope_cons_BOF")
    frames.append(pd.DataFrame({
        "source_id": "BOF_" + bof["ID"].astype(str),
        "sector": SECTOR_STEEL_BF,
        # `Name` 在 705 行炉子、237 个不同坐标上只有 60 个不同取值，因此无法标识厂址；
        # `GEM Plant ID` 可以，并且与 EAF 表所用的键一致。
        "plant_name": _text(bof, "GEM Plant ID", "unknown"),
        "province": bof["Province"].astype(str),
        "longitude": _num(bof, "lon"), "latitude": _num(bof, "lat"),
        "capacity_kt_per_year": _num(bof, "Current Capacity (ttpa)"),
        "production_kt_per_year": _num(bof, "BOF steel production"),
        "co2_mt_per_year": _num(bof, "CO2_Emi"),
        "process_co2_mt_per_year": 0.0,
        "h2_demand_kt_per_year": _num(bof, "H2_DMD"),
        "feedstock": "default",
        "status": _text(bof, "Unit Status", "operating"),
    }))
    eaf = _read("steel_all.xlsx", "China_ope_cons_EAF")
    frames.append(pd.DataFrame({
        "source_id": "EAF_" + eaf["GEM Unit ID"].astype(str),
        "sector": SECTOR_STEEL_EAF,
        "plant_name": eaf["GEM Plant ID"].astype(str),
        "province": eaf["Province"].astype(str),
        "longitude": _num(eaf, "lon"), "latitude": _num(eaf, "lat"),
        "capacity_kt_per_year": _num(eaf, "Current Capacity (ttpa)"),
        "production_kt_per_year": _num(eaf, "EAF steel production"),
        "co2_mt_per_year": _num(eaf, "CO2_Emi"),
        "process_co2_mt_per_year": 0.0,
        # 源数据中 EAF 没有 H2_DMD 列，这与 SECTOR_HAS_H2_ROUTE 一致：以废钢为原料的
        # 电弧炉没有可被替代的化石还原剂。
        "h2_demand_kt_per_year": 0.0,
        "feedstock": "default",
        "status": _text(eaf, "Unit Status", "operating"),
    }))
    return pd.concat(frames, ignore_index=True)


def _load_cement() -> pd.DataFrame:
    df = _read("Cement_all.xlsx", "水泥厂")
    return pd.DataFrame({
        "source_id": "CEM_" + df["ID"].astype(str),
        "sector": SECTOR_CEMENT,
        "plant_name": df["Name"].astype(str),
        "province": df["Province"].astype(str),
        "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
        # t/d -> kt/yr。唯一一个产能按日计的部门。
        "capacity_kt_per_year": _num(df, "Clinker Capacity") * CEMENT_KILN_DAYS / 1000.0,
        "production_kt_per_year": _num(df, "Clinker Production") * WAN_TO_KT,
        "co2_mt_per_year": _num(df, "CO2 Emission"),
        # 0.839 tCO2/t 熟料排放因子中煅烧（calcination）所占的份额。单独列出，因为这部分
        # 是任何燃料替代都动不了的（见 constants_industry.SECTOR_HAS_H2_ROUTE）。
        "process_co2_mt_per_year": _num(df, "CO2 Emission") * 0.63,
        "h2_demand_kt_per_year": 0.0,   # 无氢路线；源数据中该列只涉及燃料供热
        "feedstock": "default",
        "status": "operating",
    })


def _load_ammonia_methanol() -> pd.DataFrame:
    frames = []
    for sheet, sector, prefix in (("合成氨厂", SECTOR_AMMONIA, "NH3"),
                                  ("甲醇厂", SECTOR_METHANOL, "MEOH")):
        df = _read("Ammonia_Methanol.xlsx", sheet)
        frames.append(pd.DataFrame({
            "source_id": f"{prefix}_" + df["ID"].astype(str),
            "sector": sector,
            "plant_name": df["Company"].astype(str),
            "province": df["Province"].astype(str),
            "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
            "capacity_kt_per_year": _num(df, "Capacity") * WAN_TO_KT,
            "production_kt_per_year": _num(df, "Production") * WAN_TO_KT,
            "co2_mt_per_year": _num(df, "Emission"),
            "process_co2_mt_per_year": 0.0,
            "h2_demand_kt_per_year": _num(df, "H2_DMD") * WAN_TO_KT,
            "feedstock": _text(df, "Feedstock", "Coal"),
            "status": "operating",
        }))
    return pd.concat(frames, ignore_index=True)


def _apply_water(frame: pd.DataFrame) -> pd.DataFrame:
    """按 GB/T 18916 的单位产品取水量附上年取水量。"""
    frame = frame.copy()
    intensities = []
    for sector, feedstock in zip(frame["sector"], frame["feedstock"], strict=True):
        try:
            intensities.append(water_quota(str(sector), str(feedstock)))
        except ValueError:
            intensities.append(np.nan)
    intensity = pd.Series(intensities, index=frame.index, dtype=float)
    # 产量单位为 kt/yr；kt -> t 要 x1000。
    frame["water_m3_per_year"] = (
        intensity * frame["production_kt_per_year"] * 1000.0 * INDUSTRY_CONSUMPTION_SHARE
    )
    unsourced = sorted(frame.loc[intensity.isna(), "sector"].unique())
    if unsourced:
        logger.warning(
            "no water intake quota sourced for %s — %d plants carry NaN water use. "
            "Fill WATER_INTAKE_QUOTA_M3_PER_T from GB/T 18916 before solving.",
            unsourced, int(intensity.isna().sum()),
        )
    return frame


def build_industry_sources() -> pd.DataFrame:
    """五个部门的厂级工业点源，单位已统一。"""
    frame = pd.concat(
        [_load_steel(), _load_cement(), _load_ammonia_methanol()],
        ignore_index=True,
    )
    before = len(frame)
    frame = frame[
        frame["latitude"].between(3.0, 54.0) & frame["longitude"].between(73.0, 136.0)
    ].reset_index(drop=True)
    if len(frame) < before:
        logger.warning("dropped %d rows with missing or out-of-China coordinates",
                       before - len(frame))
    frame["sector_zh"] = frame["sector"].map(SECTOR_LABELS_ZH)
    frame.loc[~frame["sector"].map(SECTOR_HAS_H2_ROUTE), "h2_demand_kt_per_year"] = 0.0
    frame = _apply_water(frame)
    return frame[CANONICAL_COLUMNS]


def cluster_sector_hubs(
    sources: pd.DataFrame,
    hub_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """把点源聚类为 hub，且只在每个部门内部分别聚类。

    算法与煤电机组相同（`spatial.cluster_labels`，`builders/plants._cluster_plants_to_hubs` 也走它）：凝聚层次聚类、
    平均链接（average linkage）、预先算好的 haversine 距离。按部门聚类使每个 hub 在技术上
    同质——一个把水泥窑和合成氨厂混在一起的 hub，其用水强度、氢基准与捕集成本都没有意义。
    """
    counts = dict(DEFAULT_SECTOR_HUB_COUNTS if hub_counts is None else hub_counts)
    labelled = []
    for sector, group in sources.groupby("sector", sort=True):
        n_hubs = min(int(counts.get(str(sector), 30)), len(group))
        group = group.copy()
        if len(group) <= n_hubs:
            group["hub_id"] = [f"{sector}_{i:03d}" for i in range(len(group))]
        else:
            labels = cluster_labels(group["longitude"].to_numpy(), group["latitude"].to_numpy(), n_hubs)
            group["hub_id"] = [f"{sector}_{int(x):03d}" for x in labels]
        labelled.append(group)
    return pd.concat(labelled, ignore_index=True)


def aggregate_hubs(labelled: pd.DataFrame) -> pd.DataFrame:
    """把已打标签的点源合并为 hub 行，坐标按产量加权，其余量相加。"""
    rows = []
    for hub_id, group in labelled.groupby("hub_id", sort=True):
        output = float(group["production_kt_per_year"].sum())
        weights = group["production_kt_per_year"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(group))
        rows.append({
            "hub_id": str(hub_id),
            "sector": str(group["sector"].iloc[0]),
            "sector_zh": str(group["sector_zh"].iloc[0]),
            "n_plants": int(len(group)),
            "province": group["province"].mode().iat[0] if not group["province"].empty else "",
            "longitude": float(np.average(group["longitude"], weights=weights)),
            "latitude": float(np.average(group["latitude"], weights=weights)),
            "capacity_kt_per_year": float(group["capacity_kt_per_year"].sum()),
            "production_kt_per_year": output,
            "co2_mt_per_year": float(group["co2_mt_per_year"].sum()),
            "process_co2_mt_per_year": float(group["process_co2_mt_per_year"].sum()),
            "h2_demand_kt_per_year": float(group["h2_demand_kt_per_year"].sum()),
            "water_m3_per_year": float(group["water_m3_per_year"].sum(min_count=1)),
        })
    return pd.DataFrame(rows)


def write_industry_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 `industry_sources.csv` 与 `industry_hubs.csv` 写入 `inputs/`。"""
    paths.ensure_inputs_dir()
    sources = build_industry_sources()
    labelled = cluster_sector_hubs(sources)
    hubs = aggregate_hubs(labelled)
    write_csv(labelled, paths.inputs_dir / "industry_sources.csv")
    write_csv(hubs, paths.inputs_dir / "industry_hubs.csv")
    return labelled, hubs
