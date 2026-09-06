# 项目说明与工作规范

本文件对本仓库内所有 agent 生效，优先级高于默认行为。**默认使用中文。**

---

## 一、项目是什么

对**中国煤电机组与高耗能工业排放源**做机组/厂址级的**减排技术选择 + 源汇匹配**优化，
并把**气候变化影响下的水资源约束**作为一等约束纳入模型。

### 1.1 排放源侧（source）

| 源类型 | 说明 |
|---|---|
| 煤电机组 | 逐机组数据聚类为 350 个厂址级 hub（`AgglomerativeClustering`, `linkage="average"`, `metric="precomputed"`） |
| 高耗能工业 | 钢铁、水泥、合成氨、甲醇、乙二醇、烯烃、炼化等点源 |

### 1.2 改造路径（decision）

- **掺氢 / 掺氨**（co-firing，含掺烧比例决策）
- **掺生物质**（biomass co-firing，及其与 CCS 叠加形成 **BECCS**）
- **CCS 改造**（捕集率、捕集能耗惩罚、改造 capex）
- **湿冷→空冷改造**（air-cooling retrofit，水—能权衡的核心决策变量）
- **提前退役**（retirement）

### 1.3 汇侧（sink）与管网

- 103 个地质封存体，**DSA（深部咸水层）与 EOR 分开建模**，不做距离合并。
- 注入能力：栅格连通域 2 像素膨胀后 `(pixel_count/100) × 2 Mt/yr`，上限 200 Mt/yr，
  按 ACCA21 的 1,000–1,800 Mt/yr 全国注入能力标定。
- 候选管网：Delaunay 三角化 + 既有油气干线走廊 + 厂址到汇的直连弧（direct arc）。
- **绕行系数 1.136**：在中国 62 条油气干线上实测（直线 21,418 km vs 实际路由 24,338 km）。
  管长一律 `haversine × 1.136`，**不使用投影坐标算长度**（原因见 §4.1）。

### 1.4 水资源约束

- 流域尺度水—能—碳 nexus；水文强迫来自 CWatM / WaterGAP 的 ISIMIP 多 GCM 集合，
  情景 SSP1-2.6 / SSP3-7.0，区分枯水期。
- 处理组开关是 `existing_withdrawal_share = 0.85`（情景名后缀 `wd085`），
  即"存量取水权只有 85% 可被煤电继续占用"。

### 1.5 情景命名

```
WA_<水文源>_<SSP>_<季节>_<wd085>_<后缀>
   cwatm|wgap   126|370   dry     处理组开关   noair / air1000 / air1370 / capfree / nobias / seedN
```

`BASE` 为无水约束基准；`SA_*` 为敏感性。

---

## 二、建模与求解硬约定

1. **线程数固定为 8。** Gurobi 只在 (模型, 参数, **线程数**) 三者不变时才确定性可复现。
   任何要相减比较的两次求解必须共享线程数，否则差值里混入求解器噪声。
2. **MIPGap 分层。** 进主图的头部情景与 seed 复现族用 registry 的 1%；仅做背景的情景可放宽到 3%。
   **seed 族绝不能放宽**——它测的就是"只换随机种子答案能漂多远"，gap 放宽会把容差当成简并度。
3. **差值的置信区间必须用可证边界给**：`lo = (LB_t − INC_c)/INC_c`，`hi = (INC_t − LB_c)/LB_c`。
4. **简并度地板**：`1.96·√2·range/d2(k)`，d2 = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534}。
   任何差值必须显著高于地板才允许写进结论。已知结论：**只有目标函数与空冷转换量分辨得出来**，
   捕集量、CCS/BECCS/生物质容量、退役量、注入汇个数全部落在噪声内。
5. **并发求解最多 2 路**。`_biomass_access_matrices` 每次求解要分配 13,949 × 71,148 的 float64
   稠密矩阵（7.39 GiB），3 路并发必 OOM。
6. **不同输入版本的结果绝不混用**。v7 = 已发布输入版本（35 个合并汇、923 条候选边），
   v8 = 重建版本（103 个汇、连通性修复网络）。跨版本相减是本研究以前出过的事故。

---

## 三、绘图规范（顶刊标准，全中文）

风格与色系蒸馏自 `D:\6. Transfer\China-TIMES2.0\thesis.ipynb`。

### 3.1 字体与 rcParams

字体与 `thesis.ipynb` **完全一致，全仓库唯一一套，不允许任何脚本自行覆盖**。
thesis.ipynb 的原文就两行：

```python
plt.rcParams['font.family'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
```

本仓库把它落在 `scripts/plot_style.py` 的 `apply_style()` 里，其余脚本一律
`from plot_style import apply_style` 后调用，**不要再写 `font.family`**：

```python
plt.rcParams.update({
    "font.family": "SimHei",       # 单字体，不配 fallback 栈
    "axes.unicode_minus": False,
    "font.size": 8,
    "mathtext.default": "regular",
})
```

> 已完成迁移：`plot_style.py` 及另外 15 个自设字体的脚本（`plot_candidate_network`、
> `plot_spatial`、`plot_results`、`plot_sensitivity_tornado`、`visualize_*`、
> `draw_patent_figures*` 等）已全部从 Arial / Microsoft YaHei / sans-serif 栈改为 SimHei。
> 新脚本若再写 `"font.family": "Arial"`，就是破坏全局一致性。

**不配 fallback 栈是有意的**：一旦允许 `["SimHei", ..., "Arial"]` 这类回退，
同一个符号在有无 SimHei 的机器上会落到不同字形，图就不再是同一张图。
代价是 SimHei 的字符缺口必须靠写法绕开——见下表。

**SimHei 的字符缺口（已实测，matplotlib 3.10.8）：**

| 字体 | `³` `²` | `−`(U+2212) | `×` `°` `‰` |
|---|---|---|---|
| SimHei | ❌ 缺 | ❌ 缺 | ✅ |
| Microsoft YaHei | ✅ | ✅ | ✅ |
| SimSun | ✅ | ❌ 缺 | ✅ |

由此两条硬规则：

1. `axes.unicode_minus = False` **不是可选项**——SimHei 没有 U+2212，不关负号必出方框。
2. **所有上下标、单位、幂次一律走 mathtext，不要打字面的 `³`：**
   写 `r"取水量（$10^8$ m$^3$）"`、`r"Mt CO$_2$ yr$^{-1}$"`，
   不要写 `"取水量（10^8 m³）"`。已实测：mathtext 写法在纯 SimHei 下零缺字警告，
   字面 `³` 必出方框。**不要为了一个上标去换字体或加 fallback**，改写法就行。

seaborn 侧（thesis.ipynb 的原始写法）：

```python
import seaborn as sns
sns.set_theme(font_scale=1.2)
sns.set_style("ticks")          # 只留左/下轴线，无网格
```

注意 `sns.set_theme()` 会重置 rcParams，**字体设置必须放在它之后**。

### 3.2 版式

| 项 | 值 |
|---|---|
| 双栏宽 | 183 mm（`save_fig` 内已有宽度守卫，`bbox_inches="tight"` 会把画布撑大，注意 pad） |
| 单栏宽 | 89 mm |
| 最小字号 | **5 pt**，低于此排版社缩放后不可读 |
| 输出 | PDF + PNG，`dpi=300`，`bbox_inches="tight"` |
| 面板标号 | 加粗小写 `a b c`，不加括号，置于面板左上角外侧 |

thesis.ipynb 的典型 figsize 是 `(5, 3.5)` / `(10, 5)` / `(12, 4)` / `(15, 10)`——
**小图多面板**，不要做单张大图。

### 3.3 色系

三层，按用途选：

**(1) 离散分类 —— colorbm `npg`（Nature 系刊调色板），最多 10 类**

```python
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
```

**(2) 技术路径 —— 同族同色系，深端=带 CCS，浅端=不带 CCS**

这是 thesis.ipynb 配色表的核心逻辑（`Coal w/ CCS #636363` vs `w/o CCS #969696`，
`Biomass w/ CCS #31A354` vs `w/o CCS #74C476`），直接沿用到本项目：

| 路径 | 色值 | 来源色系 |
|---|---|---|
| 未改造燃煤 | `#969696` | Greys 浅端 |
| 煤电 + CCS 改造 | `#636363` | Greys 深端 |
| 生物质掺烧 | `#74C476` | Greens 浅端 |
| BECCS | `#31A354` | Greens 深端 |
| 掺氨 | `#FDAE6B` | Oranges 浅端 |
| 掺氨 + CCS | `#FD8D3C` | Oranges 深端 |
| 掺氢 | `#9E9AC8` | Purples 浅端 |
| 掺氢 + CCS | `#756BB1` | Purples 深端 |
| 退役 | `#D8DCE0` | 中性灰 |
| 空冷改造（叠加线） | `#CC3311` | 强调红 |
| CO₂ 管网 / DSA 封存 | `#3182BD` | Blues 深端 |
| EOR 封存 | `#9ECAE1` | Blues 浅端 |
| 取水 / 耗水 | `#6BAED6` / `#08519C` | Blues |

> 迁移说明：`plot_style.py` 现有的 `PATHWAY_COLORS` 是 Tol bright 系
> （`#4477AA`/`#228833`/`#CCBB44`…），与上表不一致。**新图一律用上表**；
> 老图重绘时一并替换，不要两套并存。

**(3) 连续 / 分段 —— 显式 BoundaryNorm，不要用默认连续色带**

```python
from matplotlib.colors import BoundaryNorm, ListedColormap
boundaries = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
cmap = ListedColormap(["#FEBEBD", "#FDFF73", "#A8FF00", "#00E6A8", "#0085A7", "#700049"])
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)
# colorbar 用 spacing="proportional"，最后一档标签写成 ">0.5"
```

多层级嵌套分类（如"大类 × 子类"）用 thesis.ipynb 的复合色表写法：

```python
base_cmaps = ['Reds_r', 'Greens_r', 'Blues_r', 'Purples_r']   # 一个大类一条 ramp
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.4, 0.85, N[i]))
                         for i, name in enumerate(base_cmaps)])
cmap = ListedColormap(colors)
```

顺序色带备选：`GnBu_r_d`、`YlGn`、`Spectral`、`RdYlGn_r`、`flare`、`rocket_r`；
分类色带备选：`Set2`、`tab20c`、`Paired`。

---

## 四、地图绘制规范

方法蒸馏自 `D:\6. Transfer\数据-昊天\氢管网论文EST\给晋辉\GIS_layer\plot.ipynb`。
本仓库 `scripts/plot_candidate_network.py` 已按此实现，可直接抄。

### 4.1 投影：EPSG:2380（强制）

```python
TARGET_CRS = "EPSG:2380"       # Xian 1980 / 3-degree Gauss-Kruger CM 105E，单位 m
gdf = gdf.to_crs(TARGET_CRS)
```

所有图层——底图、点、线、栅格——都要 `to_crs`/`rio.reproject` 到 2380 后再画，
`bound` 定位点也必须走同一次投影，否则 `set_xlim` 对不上。

> **仅用于显示。** 2380 的官方适用带只有 103.5°–106.5°E，把全国摊在这一个 3° 带上，
> 新疆/黑龙江方向的尺度畸变可达 7% 量级。**面积、距离、长度一律不得在 2380 下计算**：
> 管长用 haversine × 1.136（§1.3），面积用等积 Albers 或地理坐标下的大地线。

### 4.2 九段线（强制出现）

本仓库 `data/ChinaMap/boundary.shp` **已包含九段线**：`GBCODE == 26100`，261 条线段，
经纬度范围 111.44–119.71°E / 3.85–23.78°N。

> ⚠️ **但不要整幅画 `boundary.shp`。** 它的 7 个 GBCODE 语义完全不同（要素数与范围实测）：
>
> | GBCODE | 要素数 | 范围 | 是什么 |
> |---|---|---|---|
> | 61010 | 82 | 73.45–135.08°E / 3.41–53.56°N | **国界 + 海岸线** |
> | 26100 | 261 | 111.44–119.71°E / 3.85–23.78°N | **九段线** |
> | 26010 | 943 | 108.00–124.58°E / 6.32–41.06°N | 沿海岛屿轮廓 |
> | 26080 | 92 | 109.57–117.82°E / 5.45–16.25°N | 南海岛礁 |
> | 62010 / 61020 / 99001 | 4 | — | 港澳界、零星未定界 |
>
> 26010 + 26080 合计 **1 035 条**岛礁轮廓，在 183 mm 幅面上每条画不满一个像素，
> 叠起来就是南海一片黑斑加东部海岸毛刺 —— 那不是信息，是噪点。
> **主图只画 61010 + 26100；岛礁只在南海小图里画**，那个尺度上它们才是内容。

直接用 `plot_style` 的封装，不要各图自己拼图层：

```python
from plot_style import draw_china_basemap, mainland_extent, add_scs_inset, to_map_xy

draw_china_basemap(ax, facecolor="#F7F8F9")   # 省界 0.20 + 国界/九段线 0.75
x, y = to_map_xy(lon, lat)                    # 散点/折线坐标 -> EPSG:2380
mainland_extent(ax)                           # 范围裁到 17°N，见下方说明
add_scs_inset(ax.get_figure(), ax, draw=业务图层回调)
```

`load_country()` 内含断言，换底图后九段线静默消失会直接抛异常。

> **`provinces.shp` 自身延伸到 6.32°N**（含南海要素），所以 `provinces.total_bounds`
> 给出的是 6.3–53.6°N 的画框，比大陆高出近 280 km。英文图看不出来（那片什么都不画），
> 补上九段线后大陆就被压扁到画面上半部。`mainland_extent()` 把南边裁到 17°N
> （海南最南 18.15°N，完整保留），九段线主体交给小图 —— 这正是小图必需的原因。

备用独立图层（EPSG:4326，10 条 LineString）：
`D:\6. Transfer\PhD_tht\GIS_layer\china-shapefiles\china_nine_dotted_line.shp`

### 4.3 主图 + 南海小图的标准骨架

```python
from shapely.geometry import Point

# 四个定位点：[主图西南, 主图东北, 南海小图西南, 南海小图东北]
BOUND_LONLAT = [(80, 15), (150, 50), (106.5, 2.8), (123, 24.5)]
bound = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in BOUND_LONLAT], crs="EPSG:4326"
).to_crs(TARGET_CRS).geometry

provinces = gpd.read_file(ROOT / "data" / "ChinaMap" / "provinces.shp").to_crs(TARGET_CRS)

def draw_basemap(ax):
    provinces.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.2, zorder=0)
    country.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.75, zorder=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
draw_basemap(ax)
# ... 业务图层 zorder >= 2 ...
ax.set_axis_off()
ax.set_xlim(bound[0].x, bound[1].x)
ax.set_ylim(bound[0].y, bound[1].y)

# 南海小图：与主图同底图、同业务图层、同色标，只换 xlim/ylim
ax_child = fig.add_axes([0.72, 0.25, 0.25, 0.2])
draw_basemap(ax_child)
# ... 重画一遍业务图层 ...
ax_child.set_xlim(bound[2].x, bound[3].x)
ax_child.set_ylim(bound[2].y, bound[3].y)
ax_child.set_xticks([]); ax_child.set_yticks([])
ax_child.set_title(""); ax_child.set_xlabel(""); ax_child.set_ylabel("")
```

小图也可用 `inset_axes` 定位（多面板时更稳）：

```python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_child = inset_axes(ax, width="30%", height="30%", loc="lower left",
                      bbox_to_anchor=(0.75, 0.05, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
```

### 4.4 图层与线宽约定

| 图层 | facecolor | edgecolor | linewidth | zorder |
|---|---|---|---|---|
| 省界 `provinces.shp` | none | black | 0.20 | 0 |
| 国界 + 九段线 `boundary.shp` | none | black | 0.75 | 1 |
| 流域 / 分区填充 | 浅色 | none | — | 1 |
| 管网 / 流量线 | — | 按情景 | `np.sqrt(flow)/scale` | 2–3 |
| 源点 / 汇点 | 按类别 | white | 0.3 | 4+ |

**流量线宽 = `sqrt(流量)/scale`**（面积正比于流量）。图例不能直接用数据线宽，
必须用 `Line2D` 按代表值另建分档图例（如 `<0.5 / 0.5–1.0 / 1.0–2.0 / >2.0`），
并在图例标题写清单位。

### 4.5 地图输出

`ax.set_axis_off()`，无经纬网格，无标题（标题放面板标号或图注）；
存 PDF/SVG 矢量 + PNG 300 dpi。

### 4.6 出图前自检（三条，缺一不可）

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)   # 缺字警告直接抛异常，别让方框混进 PDF
    fig.savefig(path, dpi=300, bbox_inches="tight")
```

1. **缺字**：按上面把 glyph 警告升级为异常。
2. **九段线**：`assert (country["GBCODE"] == 26100).sum() > 0`（§4.2）。
3. **图幅宽度**：`save_fig` 内的宽度守卫必须通过；被撑宽通常是 `pad_inches`
   或画到轴外的 artist 造成的，先降 pad，仍超宽再查具体 artist。

---

## 五、图幅版本管理

`results/figures/v1 … v8/`，每个版本目录必须有 `README.md`，写明：

1. 这批图基于哪个**输入版本**（汇的个数、候选边条数）与哪些**已求解情景**；
2. 哪些图**能画**、哪些**因为缺情景不能画**，逐图列表；
3. 相对上一版**哪些数字变了、变了多少**；
4. **哪些结论不许从这批图里读**（例如 v8：除空冷转换外的任何容量级结论都在简并噪声内）。

跨版本比较必须在同一版本内完成，不同版本的图不得相减、不得拼进同一张图。

---

## 六、Agent 工作准则

1. **从第一性原理出发。** 识别真实目标；区分事实、假设与偏好；质疑隐含前提。
2. **不为"看起来在干活"而优化。** 不写废话，不假装完整，不引入无谓复杂度，不做表演式重构。
3. **以证据为准。** 动手前先读真实的代码、文件与上下文；结论必须落到仓库证据上；不确定处明确标出。
4. **控制风险。** 优先最小正确改动；不做无关修改；考虑回归风险与边界情况。
5. **面向可交付。** 计划必须可执行，代码必须可维护，评审必须指出真问题。
6. **显式表达。** 说清假设、说清取舍、说清做过哪些验证、说清尚未解决的风险。
7. **不偷懒。** 不停在表层理解，不套用通用模板，不含糊带过关键细节。
8. **不说空话。** 不奉承，不含糊其辞，不编造事实，不给虚假的确定性。

### 6.1 论文文本

**论文正文不写。** 本仓库的产出是模型、结果与图，正文由作者本人撰写。

### 6.2 其他规则文件

`.claude/rules/` 下的 `coding-style.md`、`security.md`、
`experiment-reproducibility.md`、`agents.md` 同样有效，与本文件冲突时以本文件为准。
