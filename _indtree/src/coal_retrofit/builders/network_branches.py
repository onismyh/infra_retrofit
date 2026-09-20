from __future__ import annotations

import re
from collections import Counter, defaultdict

import geopandas as gpd
import pandas as pd

from ..constants import NETWORK_DETOUR_FACTOR, TARGET_GEO_CRS
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km

PLANT_BRANCH_EDGE_CLASS = "hub_to_corridor_branch"
STORAGE_BRANCH_EDGE_CLASS = "corridor_to_storage_branch"
# 工业点源走和煤电完全相同的接入规则（同一 edge_class -> 同一 capex 乘子），
# 唯一的区别是它们以前只在求解时挂进来，因此没有参加三角剖分与去交叉。
INDUSTRY_BRANCH_EDGE_CLASS = PLANT_BRANCH_EDGE_CLASS
# 西藏不参与减排：4 个水泥点源（合计 6.95 Mt/yr）连同它们带出的备选管段一起去掉。
EXCLUDED_PROVINCES = {"xizang", "tibet", "西藏", "西藏自治区"}

_PROVINCE_STOPWORDS = {
    "autonomous",
    "administrative",
    "city",
    "province",
    "region",
    "special",
    "uigur",
    "uygur",
    "hui",
    "zhuang",
}

_ENGLISH_TO_CHINESE_PROVINCE_NAME = {
    "Anhui": "安徽省",
    "Beijing": "北京市",
    "Chongqing": "重庆市",
    "Fujian": "福建省",
    "Gansu": "甘肃省",
    "Guangdong": "广东省",
    "Guangxi Zhuang Autonomous Region": "广西壮族自治区",
    "Guizhou": "贵州省",
    "Hainan": "海南省",
    "Hebei": "河北省",
    "Heilongjiang": "黑龙江省",
    "Henan": "河南省",
    "Hong Kong Special Administrative Region": "香港特别行政区",
    "Hubei": "湖北省",
    "Hunan": "湖南省",
    "Inner Mongolia Autonomous Region": "内蒙古自治区",
    "Jiangsu": "江苏省",
    "Jiangxi": "江西省",
    "Jilin": "吉林省",
    "Liaoning": "辽宁省",
    "Macao Special Administrative Region": "澳门特别行政区",
    "Ningxia Hui Autonomous Region": "宁夏回族自治区",
    "Qinghai": "青海省",
    "Shaanxi": "陕西省",
    "Shandong": "山东省",
    "Shanghai": "上海市",
    "Shanxi": "山西省",
    "Sichuan": "四川省",
    "Taiwan": "台湾省",
    "Tianjin": "天津市",
    "Tibet Autonomous Region": "西藏自治区",
    "Xinjiang Uygur Autonomous Region": "新疆维吾尔族自治区",
    "Yunnan": "云南省",
    "Zhejiang": "浙江省",
}


def _normalize_province_name(value: str) -> str:
    tokens = re.findall(r"[a-z]+", str(value).lower())
    filtered = [token for token in tokens if token not in _PROVINCE_STOPWORDS]
    return "".join(filtered)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _province_centroid_lookup(paths: ProjectPaths) -> dict[str, tuple[float, float]]:
    provinces_path = paths.data_dir / "ChinaMap" / "provinces.shp"
    provinces = gpd.read_file(provinces_path)[["NAME", "geometry"]].copy()
    if provinces.crs is None:
        raise ValueError(f"{provinces_path} has no CRS defined")

    centroids = gpd.GeoSeries(provinces.geometry.centroid, crs=provinces.crs).to_crs(TARGET_GEO_CRS)
    chinese_lookup = {
        str(name): (float(point.x), float(point.y))
        for name, point in zip(provinces["NAME"], centroids, strict=True)
    }
    return {
        _normalize_province_name(english_name): chinese_lookup[chinese_name]
        for english_name, chinese_name in _ENGLISH_TO_CHINESE_PROVINCE_NAME.items()
    }


def _load_attachment_table(
    paths: ProjectPaths,
    filename: str,
    id_column: str,
    index_column: str,
    province_column: str,
    lon_candidates: list[str],
    lat_candidates: list[str],
    source_column: str = "source",
    year_column: str = "year_basis",
) -> pd.DataFrame:
    path = paths.inputs_dir / filename
    df = pd.read_csv(path)

    lon_column = _first_existing_column(df, lon_candidates)
    lat_column = _first_existing_column(df, lat_candidates)

    if lon_column is None or lat_column is None:
        if province_column not in df.columns:
            raise KeyError(f"{filename} is missing {province_column}; cannot infer coordinates")
        province_lookup = _province_centroid_lookup(paths)
        coords = df[province_column].map(lambda value: province_lookup[_normalize_province_name(value)])
        lon_series = coords.map(lambda item: item[0])
        lat_series = coords.map(lambda item: item[1])
    else:
        lon_series = pd.to_numeric(df[lon_column], errors="coerce")
        lat_series = pd.to_numeric(df[lat_column], errors="coerce")

    out = pd.DataFrame(
        {
            id_column: df[id_column],
            index_column: pd.to_numeric(df[index_column], errors="coerce").astype("Int64"),
            "lon": pd.to_numeric(lon_series, errors="coerce"),
            "lat": pd.to_numeric(lat_series, errors="coerce"),
            "province": df[province_column].astype(str).str.strip(),
            "source": df[source_column] if source_column in df.columns else paths.rel(path),
            "year_basis": df[year_column] if year_column in df.columns else "",
        }
    )
    return out


def load_plants(paths: ProjectPaths) -> pd.DataFrame:
    return _load_attachment_table(
        paths=paths,
        filename="plants.csv",
        id_column="plant_id",
        index_column="plant_index",
        province_column="province_mode",
        lon_candidates=["centroid_longitude", "lon", "longitude"],
        lat_candidates=["centroid_latitude", "lat", "latitude"],
    )


def load_storage_hubs(paths: ProjectPaths) -> pd.DataFrame:
    return _load_attachment_table(
        paths=paths,
        filename="storage_hubs.csv",
        id_column="storage_hub_id",
        index_column="storage_hub_index",
        province_column="province",
        lon_candidates=["lon", "longitude", "centroid_longitude"],
        lat_candidates=["lat", "latitude", "centroid_latitude"],
    )


def load_industry_hubs(paths: ProjectPaths) -> pd.DataFrame:
    """工业点源，剔除 EXCLUDED_PROVINCES；索引列 industry_index 由行序生成。"""
    path = paths.inputs_dir / "industry_hubs.csv"
    if not path.exists():
        return pd.DataFrame(columns=["hub_id", "industry_index", "lon", "lat", "province",
                                     "source", "year_basis"])
    df = pd.read_csv(path)
    keep = ~df["province"].astype(str).str.strip().str.lower().isin(EXCLUDED_PROVINCES)
    df = df.loc[keep].reset_index(drop=True)
    return pd.DataFrame(
        {
            "hub_id": df["hub_id"].astype(str),
            "industry_index": pd.Series(range(1, len(df) + 1), dtype="Int64"),
            "lon": pd.to_numeric(df["longitude"], errors="coerce"),
            "lat": pd.to_numeric(df["latitude"], errors="coerce"),
            "province": df["province"].astype(str).str.strip(),
            "source": df["source"] if "source" in df.columns else paths.rel(path),
            "year_basis": df["year_basis"] if "year_basis" in df.columns else "",
        }
    )


def _build_corridor_type_lookup(main_edges: pd.DataFrame) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in main_edges.itertuples(index=False):
        counts[row.from_node_id][str(row.corridor_type)] += 1
        counts[row.to_node_id][str(row.corridor_type)] += 1

    lookup: dict[str, str] = {}
    for node_id, counter in counts.items():
        lookup[node_id] = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return lookup


def _nearest_corridor_node(
    lon: float,
    lat: float,
    corridor_nodes: pd.DataFrame,
) -> tuple[str, float]:
    best_node_id = ""
    best_distance = float("inf")
    origin = (float(lon), float(lat))
    for row in corridor_nodes.itertuples(index=False):
        distance = geodesic_length_km([origin, (float(row.lon), float(row.lat))])
        if distance < best_distance:
            best_distance = distance
            best_node_id = str(row.node_id)
    return best_node_id, best_distance


def build_attachment_tables(
    paths: ProjectPaths,
    corridor_nodes: pd.DataFrame,
    main_edges: pd.DataFrame,
    next_node_index: int,
    next_edge_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    corridor_type_lookup = _build_corridor_type_lookup(main_edges)
    corridor_nodes = corridor_nodes[["node_id", "lon", "lat"]].reset_index(drop=True).copy()

    plant_hubs = load_plants(paths)
    storage_hubs = load_storage_hubs(paths)

    branch_node_rows: list[dict[str, object]] = []
    branch_edge_rows: list[dict[str, object]] = []
    corridor_degree_additions: dict[str, int] = defaultdict(int)

    def add_branch_node_and_edge(
        *,
        attachment_row: pd.Series,
        node_type: str,
        edge_class: str,
        from_corridor_to_attachment: bool,
        source: str,
        year_basis: str,
        index_column: str,
    ) -> None:
        nonlocal next_node_index, next_edge_index

        row_data = attachment_row._asdict()
        node_id = f"node_{next_node_index:05d}"
        next_node_index += 1
        attachment_lon = float(row_data["lon"])
        attachment_lat = float(row_data["lat"])
        corridor_node_id, distance_km = _nearest_corridor_node(attachment_lon, attachment_lat, corridor_nodes)
        corridor_degree_additions[corridor_node_id] += 1

        branch_node_rows.append(
            {
                "node_id": node_id,
                "lon": attachment_lon,
                "lat": attachment_lat,
                "node_type": node_type,
                "degree": 1,
                "source": source,
                "year_basis": year_basis,
                "plant_id": row_data.get("plant_id", pd.NA),
                "plant_index": row_data.get("plant_index", pd.NA),
                "storage_hub_id": row_data.get("storage_hub_id", pd.NA),
                "storage_hub_index": row_data.get("storage_hub_index", pd.NA),
                "industry_hub_id": row_data.get("hub_id", pd.NA),
                "industry_index": row_data.get("industry_index", pd.NA),
                "province": row_data.get("province", pd.NA),
            }
        )

        branch_edge_rows.append(
            {
                "edge_id": f"edge_{next_edge_index:05d}",
                "feature_index": int(row_data[index_column]) if pd.notna(row_data[index_column]) else 0,
                "part_index": 1,
                "from_node_id": corridor_node_id if from_corridor_to_attachment else node_id,
                "to_node_id": node_id if from_corridor_to_attachment else corridor_node_id,
                # Branches are straight-line attachments, so they carry the same detour factor
                # as the other straight-line candidates. Leaving them at the raw geodesic made a
                # corridor tie-in look cheaper than it is relative to the corridor itself, whose
                # length_km is a real routed polyline.
                "length_km": round(distance_km * NETWORK_DETOUR_FACTOR, 3),
                "direct_length_km": round(distance_km, 3),
                "tortuosity": NETWORK_DETOUR_FACTOR,
                "corridor_type": corridor_type_lookup.get(corridor_node_id, "main_corridor"),
                "existing_corridor_flag": 0,
                "edge_class": edge_class,
                "source": source,
                "year_basis": year_basis,
            }
        )
        next_edge_index += 1

    for row in plant_hubs.sort_values("plant_index").itertuples(index=False):
        add_branch_node_and_edge(
            attachment_row=row,
            node_type="plant",
            edge_class=PLANT_BRANCH_EDGE_CLASS,
            from_corridor_to_attachment=False,
            source=str(row.source),
            year_basis=str(row.year_basis),
            index_column="plant_index",
        )

    for row in storage_hubs.sort_values("storage_hub_index").itertuples(index=False):
        add_branch_node_and_edge(
            attachment_row=row,
            node_type="storage_hub",
            edge_class=STORAGE_BRANCH_EDGE_CLASS,
            from_corridor_to_attachment=True,
            source=str(row.source),
            year_basis=str(row.year_basis),
            index_column="storage_hub_index",
        )

    for row in load_industry_hubs(paths).sort_values("industry_index").itertuples(index=False):
        add_branch_node_and_edge(
            attachment_row=row,
            node_type="industry_hub",
            edge_class=INDUSTRY_BRANCH_EDGE_CLASS,
            from_corridor_to_attachment=False,
            source=str(row.source),
            year_basis=str(row.year_basis),
            index_column="industry_index",
        )

    branch_nodes = pd.DataFrame(branch_node_rows)
    branch_edges = pd.DataFrame(branch_edge_rows)
    return branch_nodes, branch_edges, dict(corridor_degree_additions)
