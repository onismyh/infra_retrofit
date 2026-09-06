"""Visualize retrofit pathway choices on China map."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_pathway_map(result_dir, year=2060):
    """Plot dominant pathway for each hub on China map."""
    # Load data
    provinces = gpd.read_file("data/ChinaMap/provinces.shp")
    if provinces.crs is None or str(provinces.crs) != "EPSG:4326":
        provinces = provinces.to_crs("EPSG:4326")
    provinces_proj = provinces.to_crs("EPSG:2380")

    hubs = pd.read_csv("inputs/plants_hub_200.csv")
    pathway_shares = pd.read_csv(Path(result_dir) / "artifacts/pathway_shares.csv")

    # Filter for target year and find dominant pathway
    year_data = pathway_shares[pathway_shares['year'] == year].copy()
    dominant = year_data.loc[year_data.groupby('hub_id')['share'].idxmax()]

    # Merge with hub coordinates
    hub_pathways = hubs[['hub_id', 'centroid_longitude', 'centroid_latitude', 'total_capacity_mw']].merge(
        dominant[['hub_id', 'pathway', 'share']], on='hub_id'
    )

    # Convert to GeoDataFrame and project
    hub_gdf = gpd.GeoDataFrame(
        hub_pathways,
        geometry=gpd.points_from_xy(hub_pathways['centroid_longitude'],
                                     hub_pathways['centroid_latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:2380")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot China base map
    provinces_proj.boundary.plot(ax=ax, color='black', linewidth=0.8)
    provinces_proj.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5, alpha=0.3)

    # Color mapping for pathways
    colors = {
        'retire': '#E74C3C',
        'biomass': '#27AE60',
        'beccs': '#16A085',
        'ammonia': '#3498DB',
        'ccs': '#9B59B6'
    }

    # Plot hubs by pathway
    for pathway, color in colors.items():
        pathway_hubs = hub_gdf[hub_gdf['pathway'] == pathway]
        if len(pathway_hubs) > 0:
            ax.scatter(
                pathway_hubs.geometry.x,
                pathway_hubs.geometry.y,
                s=pathway_hubs['total_capacity_mw'] / 10,
                c=color,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.8,
                label=f'{pathway.upper()} ({len(pathway_hubs)} hubs)'
            )

    ax.set_title(f'Coal Retrofit Pathway Distribution ({year})\nSize: Capacity, Color: Dominant Pathway',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    output_path = Path(result_dir) / f"artifacts/pathway_map_{year}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nPathway distribution for {year}:")
    for pathway in colors.keys():
        count = len(hub_gdf[hub_gdf['pathway'] == pathway])
        capacity = hub_gdf[hub_gdf['pathway'] == pathway]['total_capacity_mw'].sum()
        print(f"  {pathway.upper()}: {count} hubs, {capacity:.0f} MW")

if __name__ == "__main__":
    result_dir = "results/test_updated_costs/EXP-B1/20260326_123527_284078/scenarios/multi-period-baseline"

    print("Generating pathway maps...")
    for year in [2040, 2050, 2060]:
        plot_pathway_map(result_dir, year)
    print("Done!")

