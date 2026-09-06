---
name: academic-plotting
description: Publication-quality figure generation for research papers — covers both spatial maps (China) and standard charts (bar, line, waterfall). Enforces Arial font, Nature/Science sizing, colorblind-safe palettes, and vector output.
tags: [Visualization, Academic, Research, GIS, Matplotlib]
---

# Academic Plotting Skill

Generate publication-quality figures for academic papers. Covers two domains:
1. **Standard charts** — bar, stacked bar, line, waterfall, heatmap, tornado
2. **Spatial maps** — China base map with point/line/raster overlays and inset

## Global Style Configuration

Every figure script MUST start with this block. No exceptions.

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    # ── Font ──
    "font.family": "Arial",
    "font.size": 8,
    "mathtext.default": "regular",  # Math text also in Arial
    # ── Axes ──
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.facecolor": "white",
    # ── Ticks ──
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # ── Legend ──
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    # ── Lines ──
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    # ── Output ──
    "figure.dpi": 150,       # Screen preview
    "savefig.dpi": 300,      # Publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
})
```

## Figure Sizing Rules

Target journal column widths (Nature/Science family):

| Layout | Width (inches) | Width (cm) | Use case |
|--------|---------------|------------|----------|
| Single column | 3.5 | 8.9 | Simple bar/line charts |
| 1.5 column | 5.5 | 14.0 | Two-panel figures |
| Double column | 7.0 | 17.8 | Complex multi-panel, maps |

Height: keep aspect ratio ≤ 1.2 for charts, ≤ 1.0 for maps.

```python
# Single column chart
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Two-panel figure
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Four-panel grid
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

# China map (always double-column)
fig, ax = plt.subplots(figsize=(7.0, 6.0))
```

## Color Palettes

### Categorical (5 pathways)
```python
PATHWAY_COLORS = {
    "retire":  "#636363",  # Gray
    "ccs":     "#3182bd",  # Blue
    "biomass": "#31a354",  # Green
    "beccs":   "#756bb1",  # Purple
    "ammonia": "#e6550d",  # Orange
}
```

### Sequential (warm → cool, colorblind-safe)
```python
# For 6-level discrete colorbars (resource maps)
SEQUENTIAL_6 = ["#FEBEBD", "#FDFF73", "#A8FF00", "#00E6A8", "#0085A7", "#700049"]

# For diverging (cost: negative=blue, positive=red)
DIVERGING = {"negative": "#4575b4", "positive": "#d73027"}
```

### Scenario comparison (3 levels)
```python
SCENARIO_3 = {
    "low":  "#4575b4",   # Blue
    "base": "#636363",   # Gray
    "high": "#d73027",   # Red
}
```

### Rules
- Never use rainbow/jet colormap
- Always test with colorblind simulator (e.g., Coblis)
- Use `alpha=0.7` for overlapping points
- Categorical: max 7 colors; beyond that, use shape+color encoding

## Standard Chart Patterns

### Stacked Bar Chart
```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(years))
bottom = np.zeros(len(years))
bar_width = 0.55

for category, color in categories.items():
    vals = [data[y][category] for y in years]
    ax.bar(x, vals, bar_width, bottom=bottom, label=category,
           color=color, edgecolor="white", linewidth=0.3)
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylabel("Share (%)")
ax.set_ylim(0, 105)
ax.legend(ncol=2, bbox_to_anchor=(0, 1.02), loc="lower left")
```

### Horizontal Waterfall (cost breakdown)
```python
fig, ax = plt.subplots(figsize=(3.5, 4.0))
labels = list(cost_items.keys())
values = list(cost_items.values())
colors = [DIVERGING["positive"] if v > 0 else DIVERGING["negative"] for v in values]
y_pos = np.arange(len(labels))

ax.barh(y_pos, values, height=0.6, color=colors, edgecolor="white", linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.axvline(0, color="black", linewidth=0.4)
ax.set_xlabel("Billion CNY")
```

### Multi-scenario Line Chart
```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
markers = ["o", "s", "^", "D"]

for i, (label, series) in enumerate(scenarios.items()):
    ax.plot(years, values, marker=markers[i % 4], color=colors[i],
            label=label, markersize=4, markeredgecolor="white", markeredgewidth=0.3)

ax.set_xlabel("Year")
ax.set_ylabel("Metric (%)")
ax.legend()
```

### Grouped Bar Comparison
```python
fig, ax = plt.subplots(figsize=(5.5, 3.0))
n_groups = len(scenarios)
group_width = 0.8 / n_groups
x = np.arange(len(categories))

for i, (label, values) in enumerate(scenarios.items()):
    offset = (i - (n_groups - 1) / 2) * group_width
    ax.bar(x + offset, values, group_width * 0.9, label=label,
           color=colors[i], edgecolor="white", linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
```

## Panel Labeling

Always use bold lowercase letters: **(a)**, **(b)**, **(c)**

```python
# Option 1: In title
ax.set_title("(a) Panel description", loc="left", fontweight="bold", fontsize=9)

# Option 2: As text annotation (preferred for maps)
ax.text(0.02, 0.98, "(a)", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top")
```

## China Map Patterns

### Base Map Setup
```python
import geopandas as gpd
from shapely.geometry import Point

plt.rcParams['font.family'] = 'Arial'

# Projection: Albers Equal-Area for China
CHINA_CRS = "EPSG:2380"

# Bounds (in projected coordinates)
bound_pts = gpd.GeoDataFrame({'x': [80, 150, 106.5, 123], 'y': [15, 50, 2.8, 24.5]})
bound_pts.geometry = bound_pts.apply(lambda r: Point(r['x'], r['y']), axis=1)
bound_pts.crs = "EPSG:4326"
bound_pts = bound_pts.to_crs(CHINA_CRS)

provinces = gpd.read_file("data/ChinaMap/provinces.shp").to_crs(CHINA_CRS)
country = gpd.read_file("data/ChinaMap/country.shp").to_crs(CHINA_CRS)

fig, ax = plt.subplots(figsize=(7.0, 6.0))
provinces.plot(facecolor="none", edgecolor="black", linewidth=0.2, ax=ax)
country.plot(facecolor="none", edgecolor="black", linewidth=0.75, ax=ax)
ax.set_axis_off()
ax.set_xlim(bound_pts.geometry[0].x, bound_pts.geometry[1].x)
ax.set_ylim(bound_pts.geometry[0].y, bound_pts.geometry[1].y)
```

### South China Sea Inset (REQUIRED for all China maps)
```python
ax_inset = fig.add_axes([0.74, 0.20, 0.22, 0.22])
provinces.plot(facecolor="none", edgecolor="black", linewidth=0.2, ax=ax_inset)
country.plot(facecolor="none", edgecolor="black", linewidth=0.75, ax=ax_inset)
# ... plot data on inset ...
ax_inset.set_xlim(bound_pts.geometry[2].x, bound_pts.geometry[3].x)
ax_inset.set_ylim(bound_pts.geometry[2].y, bound_pts.geometry[3].y)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_title("")
ax_inset.set_xlabel("")
ax_inset.set_ylabel("")
```

### Discrete Colorbar (resource maps)
```python
from matplotlib.colors import BoundaryNorm, ListedColormap

boundaries = [0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5]
cmap = ListedColormap(SEQUENTIAL_6)
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

mappable = raster.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
cbar = fig.colorbar(mappable, ax=ax, boundaries=boundaries,
                    ticks=boundaries, spacing="uniform",
                    orientation="horizontal", pad=0.03, aspect=40, shrink=0.8)
cbar.set_label("Technical Potential (unit)", fontsize=8)
```

### Flow Width Encoding (pipeline maps)
```python
from matplotlib.lines import Line2D

# Line width = sqrt(flow) / scale
scale = 25
gdf.plot(ax=ax, color='#15FCAF', linewidth=np.sqrt(gdf["flow"]) / scale, alpha=1)

# Custom legend for width
def add_flow_legend(ax, scale, color, title, reps=(500, 1000, 1500, 2500)):
    labels = ["< 0.5", "0.5–1.0", "1.0–2.0", "> 2.0"]
    handles = [Line2D([0], [0], color=color, lw=np.sqrt(v)/scale,
                      solid_capstyle="round") for v in reps]
    ax.legend(handles=handles, labels=labels, title=title,
              loc="upper right", fontsize=7, frameon=False,
              bbox_to_anchor=(1.1, 0.85))
```

## Output Rules

```python
# ALWAYS save both vector and raster
fig.savefig("figure_name.pdf")   # Primary: vector for journal submission
fig.savefig("figure_name.png")   # Secondary: for preview and presentation

plt.close(fig)  # Free memory
```

- **PDF**: primary format for journal submission (vector, small file)
- **PNG**: for quick preview and presentations (300 DPI)
- **SVG**: for further editing in Illustrator/Inkscape
- Never use JPEG for scientific figures (lossy compression)

## Checklist Before Saving

1. ☐ Font is Arial everywhere (including colorbar, legend, annotations)
2. ☐ No top/right spines (unless heatmap)
3. ☐ Axis labels include units in parentheses
4. ☐ Panel labels (a), (b), (c) are bold and consistent
5. ☐ Legend has no frame, positioned to avoid data overlap
6. ☐ Figure width matches target column width
7. ☐ Colors are colorblind-safe (no red-green only distinction)
8. ☐ Text is readable at print size (≥7pt)
9. ☐ Grid is off (or very light alpha=0.15 if needed)
10. ☐ Saved as PDF + PNG
11. ☐ `plt.close(fig)` called after saving

## Anti-Patterns (DO NOT)

- ❌ `plt.show()` in scripts (use `Agg` backend)
- ❌ Default matplotlib colors (use defined palettes)
- ❌ `jet` or `rainbow` colormap
- ❌ Grid with alpha > 0.2
- ❌ Title on top of figure (use panel labels instead)
- ❌ `figsize` wider than 7.0 inches
- ❌ Font size < 7pt anywhere
- ❌ Thick spines/ticks (> 0.6pt)
- ❌ Legend inside data-dense area
- ❌ Emoji or special Unicode in labels
