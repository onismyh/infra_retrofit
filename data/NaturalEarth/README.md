# NaturalEarth

`builders/network_offshore.py` 量候选管道的海上段时读的陆地图层：陆地 = `data/ChinaMap/provinces.shp` 各省的并 ∪ 本图层。
省界之外还有邻国陆地，只用省界会把穿过邻国的边（云南经老挝、越南北部到北部湾，内蒙古经蒙古国东部，新疆、缅甸方向）
算成海上段（2026-10-10 复核发现：1 559 条边里 9 条穿过邻国陆地，约 1 830 km）。

> Natural Earth 1:10m Physical Vectors — Land，v5.1.2，公有领域（<https://www.naturalearthdata.com/about/terms-of-use/>）。
> 原文件取自 <https://github.com/nvkelso/natural-earth-vector/tree/v5.1.2/10m_physical>（`ne_10m_land.shp`，
> SHA-256 `4cad3a49bc75c1a4c2f3d7efae04f2f8e63151c96764b2658effabf524331fa6`）。

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `ne_10m_land_60e150e_0n60n.gpkg` | 原图层（EPSG:4326）按 60–150°E、0–60°N 裁剪，只留 `featurecla`、`scalerank` 两列，图层名 `land` | `9d05e4e2537f228b908e03ba12d886cf01803d5cb809e5114a0e65f6d3c8c9e5` |

管网节点在 74.3–132.5°E、18.1–53.2°N，裁剪框外放 5° 以上。只作求交的掩膜，不用于出图，也不表示任何国界。
两套陆地边界不完全重合，取并集即任一算陆地就算陆地；与只用省界相比，除上面 9 条外还少约 215 km，都是每段不超过 10 km 的零碎段。
合计海上段 21 755 → 19 706 km，有海上段的边 184 → 171 条。
裁剪做法（geopandas）：读 `ne_10m_land.shp`（EPSG:4326），`clip(box(60, 0, 150, 60))`，去掉空几何，只留 `featurecla`、`scalerank`
两列，写成 GPKG、图层名 `land`。原图层 11 个要素、6 837 个部件，裁后 10 个要素、1 567 个部件。
重算输入表的海上段：`python scripts/build_network_inputs.py --offshore-only`。
