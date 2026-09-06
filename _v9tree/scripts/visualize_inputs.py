"""Visualize Phase A input data on China map."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely import wkt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

# Set font to Arial for publication quality
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_hubs():
    """Plot plant hubs on China map with EPSG:2380 projection."""
    # Load and project China provinces
    provinces = gpd.read_file("data/ChinaMap/provinces.shp")
    if provinces.crs is None or str(provinces.crs) != "EPSG:4326":
        provinces = provinces.to_crs("EPSG:4326")
    provinces_proj = provinces.to_crs("EPSG:2380")

    # Load hub data
    hubs = pd.read_csv("inputs/plants_hub_200.csv")
    hubs_gdf = gpd.GeoDataFrame(
        hubs,
        geometry=gpd.points_from_xy(hubs['centroid_longitude'], hubs['centroid_latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:2380")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot China base map
    provinces_proj.boundary.plot(ax=ax, color='black', linewidth=0.8)
    provinces_proj.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5, alpha=0.3)

    # Plot hubs with size and color encoding
    if 'weighted_water_intensity_m3_per_mwh' in hubs.columns:
        scatter = ax.scatter(
            hubs_gdf.geometry.x,
            hubs_gdf.geometry.y,
            s=hubs['total_capacity_mw'] / 5,
            c=hubs['weighted_water_intensity_m3_per_mwh'],
            cmap='RdYlBu_r',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8,
            vmin=0.3, vmax=2.0
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Water Intensity (m³/MWh)', fontsize=12)
    else:
        ax.scatter(
            hubs_gdf.geometry.x,
            hubs_gdf.geometry.y,
            s=hubs['total_capacity_mw'] / 5,
            c='#1f77b4',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8
        )

    ax.set_title('Coal Power Plant Hubs (n=200)\nSize: Capacity, Color: Water Intensity',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('inputs/figures/plant_hubs_map.png', dpi=300, bbox_inches='tight')
    print("Saved: inputs/figures/plant_hubs_map.png")
    plt.close()

def plot_biomass():
    """Plot biomass supply distribution with discrete colorbar."""
    provinces = gpd.read_file("data/ChinaMap/provinces.shp")
    if provinces.crs is None or str(provinces.crs) != "EPSG:4326":
        provinces = provinces.to_crs("EPSG:4326")
    provinces_proj = provinces.to_crs("EPSG:2380")

    biomass = pd.read_csv("inputs/biomass_supply_curve.csv")
    biomass_gdf = gpd.GeoDataFrame(
        biomass,
        geometry=gpd.points_from_xy(biomass['longitude'], biomass['latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:2380")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot China base map
    provinces_proj.boundary.plot(ax=ax, color='black', linewidth=0.8)
    provinces_proj.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5, alpha=0.3)

    # Discrete colorbar for biomass potential (GJ)
    supply_values = biomass['biomass_supply_gj'] / 1e9  # Convert to billion GJ
    boundaries = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    colors = ["#FEBEBD", "#FDFF73", "#A8FF00", "#00E6A8", "#0085A7"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    scatter = ax.scatter(
        biomass_gdf.geometry.x,
        biomass_gdf.geometry.y,
        s=30,
        c=supply_values,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors='none'
    )

    cbar = plt.colorbar(scatter, ax=ax, boundaries=boundaries, ticks=boundaries,
                        spacing="proportional", shrink=0.8, pad=0.02)
    cbar.set_label('Biomass Supply (Billion GJ/yr)', fontsize=12)

    ax.set_title('Biomass Supply Distribution\nColor: Supply Potential',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('inputs/figures/biomass_supply_map.png', dpi=300, bbox_inches='tight')
    print("Saved: inputs/figures/biomass_supply_map.png")
    plt.close()

def plot_network():
    """Plot CO2 pipeline network with actual geometry lines."""
    provinces = gpd.read_file("data/ChinaMap/provinces.shp")
    if provinces.crs is None or str(provinces.crs) != "EPSG:4326":
        provinces = provinces.to_crs("EPSG:4326")
    provinces_proj = provinces.to_crs("EPSG:2380")

    nodes = pd.read_csv("inputs/pipeline_nodes.csv")
    edges = pd.read_csv("inputs/pipeline_candidate_edges.csv")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot China base map
    provinces_proj.boundary.plot(ax=ax, color='black', linewidth=0.8)
    provinces_proj.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5, alpha=0.3)

    # Plot pipeline edges from WKT geometry
    if 'geometry_wkt' in edges.columns:
        edges_valid = edges.dropna(subset=['geometry_wkt'])
        if len(edges_valid) > 0:
            edges_geom = edges_valid['geometry_wkt'].apply(wkt.loads)
            edges_gdf = gpd.GeoDataFrame(edges_valid, geometry=edges_geom, crs="EPSG:4326").to_crs("EPSG:2380")
            edges_gdf.plot(ax=ax, color='#FF6B6B', linewidth=0.5, alpha=0.6, zorder=2)

    # Plot nodes
    nodes_gdf = gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes['lon'], nodes['lat']),
        crs="EPSG:4326"
    ).to_crs("EPSG:2380")
    ax.scatter(nodes_gdf.geometry.x, nodes_gdf.geometry.y, s=8, c='#C92A2A',
               alpha=0.8, zorder=3, edgecolors='none')

    ax.set_title('CO2 Pipeline Candidate Network\nLines: Corridors, Points: Nodes',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('inputs/figures/pipeline_network_map.png', dpi=300, bbox_inches='tight')
    print("Saved: inputs/figures/pipeline_network_map.png")
    plt.close()

if __name__ == "__main__":
    Path("inputs/figures").mkdir(exist_ok=True)

    print("Generating visualizations...")
    plot_hubs()
    plot_biomass()
    plot_network()
    print("Done!")
