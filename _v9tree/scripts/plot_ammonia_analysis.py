"""Ammonia break-even analysis and carbon price × ammonia price phase diagram.

Expert review P1-1: quantify ammonia's zero marginal value with analytical
cost comparisons — the paper's most citable finding.

Uses analytical LCOE formulas (no model runs needed).

Usage:
    python scripts/plot_ammonia_analysis.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Style (match plot_results.py) ────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False, "font.size": 8, "mathtext.default": "regular",
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "results" / "figures"

# ── Model parameters (from scenario.py / OptimizationAssumptions) ────────────
HEAT_RATE = 9.0          # GJ/MWh
NH3_LHV = 0.0186         # GJ/kg
COAL_FUEL_COST = 35.0    # CNY/GJ
COAL_EF = 0.82           # tCO2/MWh (unabated)
CCS_CAPTURE_RATE = 0.90
CAPACITY_FACTOR = 0.55
CCS_CAPEX = 1200.0       # CNY/kW
CCS_OM = 50.0            # CNY/MWh (incremental O&M)
CCS_EFF_PENALTY = 0.08   # absolute efficiency loss
BIOMASS_OM = 30.0         # CNY/MWh
HOURS_PER_YEAR = 8760
DISCOUNT_RATE = 0.06
AMORT_YEARS = 20
CRF = DISCOUNT_RATE * (1 + DISCOUNT_RATE)**AMORT_YEARS / ((1 + DISCOUNT_RATE)**AMORT_YEARS - 1)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


# ── Analytical cost functions ────────────────────────────────────────────────

def ammonia_abatement_cost(carbon_price: np.ndarray, nh3_price_per_kg: float,
                           blend: float = 0.20) -> np.ndarray:
    """Levelized cost of ammonia co-firing per MWh (net of coal savings + carbon credit).

    Args:
        carbon_price: CNY/tCO2 array
        nh3_price_per_kg: ammonia price in CNY/kg
        blend: ammonia blend fraction (energy basis)

    Returns:
        net cost per MWh (CNY/MWh) — lower is better
    """
    # Ammonia fuel needed per MWh
    nh3_kg_per_mwh = blend * HEAT_RATE / NH3_LHV  # kg/MWh
    fuel_cost = nh3_kg_per_mwh * nh3_price_per_kg
    # Coal savings
    coal_savings = blend * HEAT_RATE * COAL_FUEL_COST
    # CO2 reduction
    co2_reduction = blend * COAL_EF  # tCO2/MWh
    carbon_credit = co2_reduction * carbon_price
    # Fixed O&M for ammonia pathway
    om_cost = 80.0  # CNY/MWh
    return fuel_cost - coal_savings + om_cost - carbon_credit


def biomass_abatement_cost(carbon_price: np.ndarray,
                           blend: float = 0.15) -> np.ndarray:
    """Levelized cost of biomass co-firing per MWh."""
    # Biomass delivered cost: ~50 CNY/GJ (purchase 22 + pretreatment 12.3 + transport ~16)
    biomass_cost_per_gj = 50.0
    fuel_cost = blend * HEAT_RATE * biomass_cost_per_gj
    coal_savings = blend * HEAT_RATE * COAL_FUEL_COST
    co2_reduction = blend * COAL_EF
    carbon_credit = co2_reduction * carbon_price
    om_cost = BIOMASS_OM
    return fuel_cost - coal_savings + om_cost - carbon_credit


def ccs_abatement_cost(carbon_price: np.ndarray) -> np.ndarray:
    """Levelized cost of CCS retrofit per MWh."""
    # Annualized CAPEX using capital recovery factor at 6% discount rate
    annual_capex = CCS_CAPEX * 1000 * CRF / (CAPACITY_FACTOR * HOURS_PER_YEAR)  # CNY/MWh
    # Energy penalty cost
    eff_penalty_cost = CCS_EFF_PENALTY * HEAT_RATE * COAL_FUEL_COST
    # CO2 captured
    co2_captured = CCS_CAPTURE_RATE * COAL_EF
    carbon_credit = co2_captured * carbon_price
    # Storage + transport (~60 CNY/t × captured)
    storage_transport = 60.0 * co2_captured
    return annual_capex + CCS_OM + eff_penalty_cost + storage_transport - carbon_credit


# ── Fig: Ammonia break-even price ────────────────────────────────────────────

def fig_ammonia_breakeven():
    """Break-even ammonia price vs carbon price, compared with biomass and CCS."""
    carbon_prices = np.linspace(0, 1500, 300)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # For each competitor, find the NH3 price that makes ammonia equally expensive
    # ammonia_cost(cp, nh3_price, blend=0.20) = competitor_cost(cp)
    # Solve for nh3_price:
    # nh3_kg * nh3_price - coal_savings + om - carbon_credit = competitor_cost
    # nh3_price = (competitor_cost + coal_savings - om + carbon_credit) / nh3_kg

    blend_nh3 = 0.20
    nh3_kg = blend_nh3 * HEAT_RATE / NH3_LHV
    coal_sav = blend_nh3 * HEAT_RATE * COAL_FUEL_COST
    co2_red_nh3 = blend_nh3 * COAL_EF
    om_nh3 = 80.0

    competitors = [
        ("Biomass (15%)", biomass_abatement_cost(carbon_prices, 0.15), "#228833", "-"),
        ("Biomass (30%)", biomass_abatement_cost(carbon_prices, 0.30), "#228833", "--"),
        ("CCS retrofit", ccs_abatement_cost(carbon_prices), "#4477AA", "-"),
    ]

    for label, comp_cost, color, ls in competitors:
        # break-even: ammonia_cost = comp_cost
        # nh3_price = (comp_cost + coal_sav - om_nh3 + co2_red_nh3 * cp) / nh3_kg
        breakeven_price = (comp_cost + coal_sav - om_nh3 + co2_red_nh3 * carbon_prices) / nh3_kg
        ax.plot(carbon_prices, breakeven_price, color=color, linestyle=ls, label=label)

    # Current price markers
    ax.axhline(3.0, color="#EE6677", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.text(1480, 3.1, "Current NH$_3$\nprice", fontsize=5.5, color="#EE6677",
            ha="right", va="bottom")

    ax.axvline(80, color="#888888", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.text(90, 4.8, "China\nETS", fontsize=5.5, color="#888888", va="top")

    # Shade region where ammonia is competitive (below ALL lines)
    bio15_be = (biomass_abatement_cost(carbon_prices, 0.15) + coal_sav - om_nh3 +
                co2_red_nh3 * carbon_prices) / nh3_kg
    ccs_be = (ccs_abatement_cost(carbon_prices) + coal_sav - om_nh3 +
              co2_red_nh3 * carbon_prices) / nh3_kg
    floor = np.minimum(bio15_be, ccs_be)
    ax.fill_between(carbon_prices, 0, np.clip(floor, 0, None),
                    alpha=0.08, color="#EE6677", label="NH$_3$ competitive")

    ax.set_xlabel("Carbon price (CNY/tCO$_2$)")
    ax.set_ylabel("Break-even NH$_3$ price (CNY/kg)")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=6, loc="upper left")
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_ammonia_breakeven")


# ── Fig: 2D phase diagram ────────────────────────────────────────────────────

def fig_phase_diagram():
    """Carbon price × ammonia price phase space showing cheapest pathway."""
    cp_range = np.linspace(0, 1500, 200)
    nh3_range = np.linspace(0, 5, 200)
    CP, NH3 = np.meshgrid(cp_range, nh3_range)

    # Cost per MWh for each pathway at each (carbon_price, nh3_price) point
    cost_bio = biomass_abatement_cost(CP, blend=0.15)
    cost_ccs = ccs_abatement_cost(CP)

    # Ammonia cost varies with NH3 price
    blend_nh3 = 0.20
    nh3_kg = blend_nh3 * HEAT_RATE / NH3_LHV
    coal_sav = blend_nh3 * HEAT_RATE * COAL_FUEL_COST
    co2_red = blend_nh3 * COAL_EF
    cost_nh3 = nh3_kg * NH3 - coal_sav + 80.0 - co2_red * CP

    # Determine winner: 0=biomass, 1=CCS, 2=ammonia
    costs = np.stack([cost_bio, cost_ccs, cost_nh3], axis=0)
    winner = np.argmin(costs, axis=0)

    # Custom colormap: green=biomass, blue=CCS, red=ammonia
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#228833", "#4477AA", "#EE6677"])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.pcolormesh(CP, NH3, winner, cmap=cmap, shading="auto", alpha=0.5)

    # Add contour boundaries
    ax.contour(CP, NH3, winner, levels=[0.5, 1.5], colors="black",
               linewidths=0.6, linestyles="-")

    # Region labels
    ax.text(200, 1.0, "Biomass", fontsize=8, fontweight="bold", color="#228833")
    ax.text(900, 4.0, "CCS", fontsize=8, fontweight="bold", color="#4477AA")
    # Find ammonia region centroid (if it exists)
    nh3_wins = winner == 2
    if nh3_wins.any():
        # Place label at approximate center of ammonia region
        nh3_ys, nh3_xs = np.where(nh3_wins)
        cx = cp_range[int(np.mean(nh3_xs))]
        cy = nh3_range[int(np.mean(nh3_ys))]
        ax.text(cx, cy, "NH$_3$", fontsize=7, fontweight="bold", color="#EE6677",
                ha="center", va="center")
    else:
        ax.text(1400, 0.3, "NH$_3$\n(not\nviable)", fontsize=5.5, color="#EE6677",
                ha="right", va="bottom", fontstyle="italic")

    # Policy reference lines
    for cp_val, label, va in [(80, "China ETS\n(2024)", "bottom"),
                                (700, "EU ETS", "bottom"),
                                (1200, "SCC", "bottom")]:
        ax.axvline(cp_val, color="#888888", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.text(cp_val + 15, 4.8, label, fontsize=5, color="#888888", va=va, rotation=90)

    # Current price point
    ax.plot(80, 3.0, "k*", markersize=8, zorder=10)
    ax.annotate("Current\nprices", xy=(80, 3.0), xytext=(200, 3.8),
                fontsize=6, ha="left",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    ax.set_xlabel("Carbon price (CNY/tCO$_2$)")
    ax.set_ylabel("Ammonia price (CNY/kg)")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 5)
    ax.text(0.0, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_phase_diagram")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating ammonia analysis figures...")
    fig_ammonia_breakeven()
    fig_phase_diagram()
    print(f"\nAmmonia figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
