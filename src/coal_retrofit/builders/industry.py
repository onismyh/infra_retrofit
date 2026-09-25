"""从厂级点源库构建工业点源清单。

来源：`D:\\6. Transfer\\PhD_tht\\point_source\\`——六个工作簿，覆盖七个部门，各自都带
WGS84 坐标、产能、产量、CO2，以及（EAF 除外）一列氢需求。

该库中的单位并不统一，一旦弄错，整个工业部门会被悄无声息地按比例放缩。编写本模块前
已对照全国总量核验：

    钢铁      产量以 kt 计      CO2 以 Mt 计   H2_DMD 以 kt 计   （939 Mt 粗钢）
    其他      产量以万吨计      CO2 以 Mt 计   H2_DMD 以万吨计   （52 Mt NH3，678 Mt 原油，
                                                                   1 269 Mt 熟料）
    水泥      产能以 t/d 计（不是万吨）——9 000 t/d x 300 d = 2.7 Mt/yr，与它自己的
              产量列吻合，单位正是靠这项核对确定下来的。

全部统一为：产能与产量用 kt/yr，CO2 用 Mt/yr，氢用 kt/yr。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ..artifacts import write_csv
from ..constants import PLANT_YEAR_BASIS
from ..constants_industry import (
    ASSET_LIFETIME_YEARS,
    DEFAULT_SECTOR_HUB_COUNTS,
    INDUSTRY_CONSUMPTION_SHARE,
    SECTOR_AMMONIA,
    SECTOR_CEMENT,
    SECTOR_COAL_CHEM,
    SECTOR_HAS_H2_ROUTE,
    SECTOR_LABELS_ZH,
    SECTOR_METHANOL,
    SECTOR_REFINERY,
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
    SECTORS_OUT_OF_SCOPE,
    SECTORS_WITH_UNIFORM_RETIREMENT,
    water_quota,
)
from ..paths import ProjectPaths
# 复用煤电机组那套距离矩阵，而不是再写一套：两类 hub 必须在完全相同的几何上聚类，
# 否则它们的 hub 规模不可比。
from .plants import _haversine_distance_matrix

logger = logging.getLogger(__name__)

POINT_SOURCE_DIR = Path(r"D:\6. Transfer\PhD_tht\point_source")

WAN_TO_KT = 10.0        # 万吨 -> kt
CEMENT_KILN_DAYS = 300  # t/d -> kt/yr，数据源隐含的年运行天数惯例

CANONICAL_COLUMNS = [
    "source_id", "sector", "sector_zh", "plant_name", "province",
    "longitude", "latitude", "capacity_kt_per_year", "production_kt_per_year",
    "co2_mt_per_year", "process_co2_mt_per_year", "h2_demand_kt_per_year",
    "has_h2_route", "feedstock", "commission_year", "commission_year_observed",
    "asset_life_years", "water_intensity_m3_per_t", "water_m3_per_year", "status",
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


def _year(frame: pd.DataFrame, column: str) -> pd.Series:
    """投运年份，转为可空整数；'unknown' 与空白变为 NA。"""
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    raw = pd.to_numeric(frame[column], errors="coerce")
    raw = raw.where((raw >= 1900) & (raw <= 2100))
    return raw.astype("Int64")


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
        "commission_year": _year(bof, "Start Date"),
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
        "commission_year": _year(eaf, "Start Date"),
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
        "commission_year": _year(df, "Time"),
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
            "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "status": "operating",
        }))
    return pd.concat(frames, ignore_index=True)


def _load_refinery() -> pd.DataFrame:
    df = _read("Refinery_all.xlsx", "China")
    return pd.DataFrame({
        "source_id": "REF_" + df["ID"].astype(str),
        "sector": SECTOR_REFINERY,
        "plant_name": df["Name"].astype(str),
        "province": df["Province"].astype(str),
        "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
        "capacity_kt_per_year": _num(df, "Crude cap") * WAN_TO_KT,
        "production_kt_per_year": _num(df, "Crude input") * WAN_TO_KT,
        "co2_mt_per_year": _num(df, "Emission"),
        "process_co2_mt_per_year": 0.0,
        "h2_demand_kt_per_year": _num(df, "H2_dmd") * WAN_TO_KT,
        "feedstock": "default",
        "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "status": "operating",
    })


def _load_coal_chemical() -> pd.DataFrame:
    """现代煤化工。读四张分工艺表，而不是 `All` 汇总表。

    `All` 丢掉了产能与产量，而用水计算需要它们；分工艺表还把过程 CO2 与燃料 CO2 分开，
    这一点很重要，因为掺氢只能替代燃料那部分。
    """
    frames = []
    for sheet in ("煤制天然气", "煤制烯烃", "煤制油", "煤制乙二醇"):
        df = _read("Coal-to-chemicals.xlsx", sheet)
        frames.append(pd.DataFrame({
            "source_id": f"CTC_{sheet}_" + df["ID"].astype(str),
            "sector": SECTOR_COAL_CHEM,
            "plant_name": df["Project"].astype(str),
            "province": df["Province"].astype(str),
            "longitude": _num(df, "Longtitude"), "latitude": _num(df, "Latitude"),
            "capacity_kt_per_year": _num(df, "Capacity") * WAN_TO_KT,
            "production_kt_per_year": _num(df, "Production") * WAN_TO_KT,
            "co2_mt_per_year": _num(df, "Total Direct Emission"),
            "process_co2_mt_per_year": _num(df, "CO2 Emission-Process"),
            "h2_demand_kt_per_year": _num(df, "H2_DMD") * WAN_TO_KT,
            "feedstock": _text(df, "Feedstock", "Coal"),
            "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "status": "operating",
        }))
    return pd.concat(frames, ignore_index=True)


def _assign_missing_years(frame: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """补齐缺失的投运年份，补法取决于该部门掌握了多少信息。

    部分有观测的部门（水泥、钢铁）从它们自己的已观测年份分布中取值补缺：补出的值复现
    观测到的经验分位数，因此该部门的年龄结构得以保留，而不是被抹平。完全没有观测年份的
    部门（`SECTORS_WITH_UNIFORM_RETIREMENT`）把年龄均匀铺在 [0, life] 上。

    两种补法都是按 `source_id` 排序的确定性分位数阶梯——不是抽样——所以两次运行结果
    逐位相同（`.claude/rules/experiment-reproducibility.md`）。
    所有被补的行 `commission_year_observed` 都保持 False，因此任何读取厂龄的图都能把
    它们排除，而不是把合成的年份当成数据。
    """
    frame = frame.copy()
    frame["commission_year_observed"] = frame["commission_year"].notna()
    for sector, group in frame.groupby("sector", sort=True):
        life = ASSET_LIFETIME_YEARS[str(sector)]
        missing = group.index[group["commission_year"].isna()]
        if len(missing) == 0:
            continue
        ordered = frame.loc[missing].sort_values("source_id").index
        fractions = (np.linspace(0.0, 1.0, num=len(ordered)) if len(ordered) > 1
                     else np.array([0.5]))
        observed = group.loc[group["commission_year"].notna(), "commission_year"]
        if str(sector) in SECTORS_WITH_UNIFORM_RETIREMENT or observed.empty:
            years = base_year - np.rint(fractions * life)
            how = "uniform over %d-year life" % life
        else:
            # 经验分位数补法：复现该部门的观测分布。
            years = np.rint(np.quantile(observed.astype(float).to_numpy(), fractions))
            how = "empirical quantiles of %d observed years" % len(observed)
        frame.loc[ordered, "commission_year"] = years.astype("int64")
        logger.info("%s: filled %d/%d commissioning years by %s",
                    sector, len(missing), len(group), how)
    frame["commission_year"] = frame["commission_year"].astype("Int64")
    frame["asset_life_years"] = frame["sector"].map(ASSET_LIFETIME_YEARS).astype(int)
    return frame


def _apply_water(frame: pd.DataFrame) -> pd.DataFrame:
    """附上 GB/T 18916 的单位产品取水量，以及由此得到的年取水量。"""
    frame = frame.copy()
    intensities = []
    for sector, feedstock in zip(frame["sector"], frame["feedstock"], strict=True):
        try:
            intensities.append(water_quota(str(sector), str(feedstock)))
        except ValueError:
            intensities.append(np.nan)
    frame["water_intensity_m3_per_t"] = intensities
    # 产量单位为 kt/yr；kt -> t 要 x1000。
    frame["water_m3_per_year"] = (
        frame["water_intensity_m3_per_t"] * frame["production_kt_per_year"] * 1000.0
        * INDUSTRY_CONSUMPTION_SHARE
    )
    unsourced = sorted(frame.loc[frame["water_intensity_m3_per_t"].isna(), "sector"].unique())
    if unsourced:
        logger.warning(
            "no water intake quota sourced for %s — %d plants carry NaN water use. "
            "Fill WATER_INTAKE_QUOTA_M3_PER_T from GB/T 18916 before solving.",
            unsourced, int(frame["water_intensity_m3_per_t"].isna().sum()),
        )
    return frame


def build_industry_sources(base_year: int = int(PLANT_YEAR_BASIS)) -> pd.DataFrame:
    """全部七个部门的厂级工业点源，单位已统一。"""
    frame = pd.concat(
        [_load_steel(), _load_cement(), _load_ammonia_methanol(),
         _load_refinery(), _load_coal_chemical()],
        ignore_index=True,
    )
    frame = frame[~frame["sector"].isin(SECTORS_OUT_OF_SCOPE)].reset_index(drop=True)
    before = len(frame)
    frame = frame[
        frame["latitude"].between(3.0, 54.0) & frame["longitude"].between(73.0, 136.0)
    ].reset_index(drop=True)
    if len(frame) < before:
        logger.warning("dropped %d rows with missing or out-of-China coordinates",
                       before - len(frame))
    frame["sector_zh"] = frame["sector"].map(SECTOR_LABELS_ZH)
    frame["has_h2_route"] = frame["sector"].map(SECTOR_HAS_H2_ROUTE)
    frame.loc[~frame["has_h2_route"], "h2_demand_kt_per_year"] = 0.0
    frame = _assign_missing_years(frame, base_year)
    frame = _apply_water(frame)
    return frame[CANONICAL_COLUMNS]


def cluster_sector_hubs(
    sources: pd.DataFrame,
    hub_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """把点源聚类为 hub，且只在每个部门内部分别聚类。

    算法与煤电机组相同（`builders/plants._cluster_plants_to_hubs`）：凝聚层次聚类、
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
            distances = _haversine_distance_matrix(
                group["longitude"].to_numpy(), group["latitude"].to_numpy()
            )
            labels = AgglomerativeClustering(
                n_clusters=n_hubs, metric="precomputed", linkage="average"
            ).fit_predict(distances)
            group["hub_id"] = [f"{sector}_{int(x):03d}" for x in labels]
        labelled.append(group)
    return pd.concat(labelled, ignore_index=True)


def aggregate_hubs(labelled: pd.DataFrame) -> pd.DataFrame:
    """把已打标签的点源合并为 hub 行，强度按产量加权。"""
    rows = []
    for hub_id, group in labelled.groupby("hub_id", sort=True):
        output = float(group["production_kt_per_year"].sum())
        weights = group["production_kt_per_year"].to_numpy(dtype=float)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = np.ones(len(group))
            weight_sum = float(len(group))
        intensity = group["water_intensity_m3_per_t"].to_numpy(dtype=float)
        finite = np.isfinite(intensity)
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
            "has_h2_route": bool(group["has_h2_route"].iloc[0]),
            "water_intensity_m3_per_t": (
                float(np.average(intensity[finite], weights=weights[finite]))
                if finite.any() else np.nan
            ),
            "water_m3_per_year": float(group["water_m3_per_year"].sum(min_count=1)),
            "mean_commission_year": float(group["commission_year"].astype(float).mean()),
            "share_year_observed": float(group["commission_year_observed"].mean()),
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
