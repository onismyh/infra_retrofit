"""Visualize optimization results."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_pathway_evolution(result_dir):
    """Plot technology pathway evolution over time."""
    summary_path = Path(result_dir) / "artifacts/summary.md"

    # Parse pathway shares from summary
    years = [2040, 2050, 2060]
    pathways = ['retire', 'biomass', 'beccs', 'ammonia', 'ccs']
    shares = {p: [] for p in pathways}

    with open(summary_path) as f:
        for line in f:
            if '/' in line and any(p in line for p in pathways):
                parts = line.strip().split('/')
                if len(parts) == 2:
                    year_part = parts[0].strip('- ')
                    pathway_part = parts[1].split(':')[0].strip()
                    if pathway_part in shares and ':' in parts[1]:
                        value_str = parts[1].split('`')[1]
                        try:
                            value = float(value_str)
                            shares[pathway_part].append(value * 100)
                        except ValueError:
                            continue

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'retire': '#E74C3C',
        'biomass': '#27AE60',
        'beccs': '#16A085',
        'ammonia': '#3498DB',
        'ccs': '#9B59B6'
    }

    bottom = np.zeros(len(years))
    for pathway in pathways:
        if pathway in shares and len(shares[pathway]) == 3:
            ax.bar(years, shares[pathway], bottom=bottom,
                   label=pathway.upper(), color=colors[pathway], width=8)
            bottom += shares[pathway]

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Technology Share (%)', fontsize=12)
    ax.set_title('Coal Retrofit Technology Pathway Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(result_dir) / "artifacts/pathway_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_cost_breakdown(result_dir):
    """Plot cost breakdown by year."""
    overview = pd.read_csv(Path(result_dir) / "artifacts/overview.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    years = overview['year'].values
    costs = overview['objective_cny'].values / 1e12  # Convert to trillion CNY

    ax.bar(years, costs, color='#3498DB', width=8, alpha=0.8, edgecolor='black', linewidth=1)

    for i, (year, cost) in enumerate(zip(years, costs)):
        ax.text(year, cost + 0.3, f'{cost:.1f}T', ha='center', fontsize=11)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Cost (Trillion CNY)', fontsize=12)
    ax.set_title('Optimization Cost by Year', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(result_dir) / "artifacts/cost_by_year.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    result_dir = "results/test_refactored_hub200/EXP-B1/20260326_005046_166287/scenarios/multi-period-baseline"

    print("Generating result visualizations...")
    plot_pathway_evolution(result_dir)
    plot_cost_breakdown(result_dir)
    print("Done!")
