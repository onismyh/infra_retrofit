"""读入 `inputs/` 各表并按情景加工成 `PreparedInputs`：机组、封存、生物质、氨、水、工业、部门目标、管网。

逐年系数矩阵在 `year_matrices.py`，资源/水链路矩阵在 `resource_access.py` / `water_access.py`。
"""
from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import pandas as pd

from ..paths import ProjectPaths
from ._shared import PreparedInputs
from .network import build_runtime_network
from .resource_access import _coarsen_resource_nodes, _haversine_distances_km
from .scenario import OptimizationAssumptions, OptimizationScenario

logger = logging.getLogger(__name__)


def _paths_df_from_assumptions(assumptions: OptimizationAssumptions, scenario: OptimizationScenario) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field_info in fields(assumptions):
        rows.append({"scope": "assumption", "key": field_info.name, "value": getattr(assumptions, field_info.name)})
    for field_info in fields(scenario):
        rows.append({"scope": "scenario", "key": field_info.name, "value": getattr(scenario, field_info.name)})
    return pd.DataFrame(rows)


def _prepare_plants(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    plants["province_name"] = plants["province_mode"].astype(str)
    # Use province-specific operating hours for generation calculation
    plants["province_cf"] = plants["province_name"].map(
        lambda prov: assumptions.province_cf(prov)
    )
    plants["annual_generation_mwh"] = plants["total_capacity_mw"].astype(float) * plants["province_cf"] * 8760.0
    # Capacity-weighted current fleet hours, the denominator of
    # `scenario.operating_hours_scale`. Stored on every row so `_build_year_matrices` can
    # read it without recomputing the weighting.
    capacity = plants["total_capacity_mw"].astype(float)
    plants["fleet_hours_now"] = float(
        (capacity * plants["province_cf"] * 8760.0).sum() / max(float(capacity.sum()), 1e-9)
    )
    plants["baseline_emissions_mt"] = (
        plants["annual_generation_mwh"].astype(float) * assumptions.coal_emission_factor_t_per_mwh / 1_000_000.0
    )
    plants["effective_cooling_technology"] = (
        str(scenario.forced_cooling_technology)
        if scenario.forced_cooling_technology
        else plants["dominant_cooling_technology"].astype(str)
    )
    # Water that has to come from the freshwater system, on the basis the scenario selects.
    # `builders/plants.py` writes four capacity-weighted intensities per site from the Wang
    # (2023) table, looked up per unit by steam cycle and cooling system, with coastal
    # seawater condensers already zeroed out. Older inputs without those columns fall back
    # to the flat per-cooling consumption values.
    # Three bases, three jobs. They are NOT interchangeable and the model uses all three:
    #
    #   consumption  what the basin actually loses      -> the availability constraint
    #   quota        what China meters and charges       -> the water tariff
    #   withdrawal   everything diverted, incl. the flow returned downstream -> reported only
    #
    # Constraining on withdrawal was tried and abandoned: an environmental-flow allowance is
    # a rule about DEPLETION, and applying it to once-through condenser flow — which returns
    # to the river a few kilometres on — put the Yangtze basin 155% over its limit purely
    # because 45.6% of its coal is once-through. What actually limits a once-through intake
    # is instantaneous channel discharge at the intake, which needs routed flow (`dis`) at
    # reach scale; until that exists, withdrawal is a diagnostic, never a constraint.
    base_column = "consumption_intensity_m3_per_mwh"
    ccs_column = "consumption_ccs_intensity_m3_per_mwh"
    has_table = base_column in plants.columns and ccs_column in plants.columns
    if has_table and not scenario.forced_cooling_technology:
        plants["baseline_water_intensity_m3_per_mwh"] = plants[base_column].astype(float)
        plants["capture_water_intensity_m3_per_mwh"] = plants[ccs_column].astype(float)
    else:
        if not has_table:
            logger.warning("plants.csv lacks the per-technology water table; using flat cooling values")
        flat = plants["effective_cooling_technology"].map(assumptions.cooling_baseline_water_intensity)
        plants["baseline_water_intensity_m3_per_mwh"] = flat
        plants["capture_water_intensity_m3_per_mwh"] = flat * assumptions.ccs_water_multiplier
    plants["water_basis"] = "consumption"
    # Charge ratio: the tariff is levied on the metered quota volume, but the variable the
    # solver tracks is consumption, so the delivered price is scaled by quota/consumption per
    # plant. Capture increments are charged at the base ratio — a <=25% approximation on a
    # term worth ~5% of system cost. Falls back to 1.0 wherever the quota column is absent.
    if "quota_intensity_m3_per_mwh" in plants.columns:
        consumption = plants["baseline_water_intensity_m3_per_mwh"].astype(float)
        ratio = plants["quota_intensity_m3_per_mwh"].astype(float) / consumption.where(consumption > 0)
        plants["water_charge_ratio"] = ratio.fillna(1.0).clip(lower=0.0, upper=20.0)
    else:
        logger.warning("plants.csv lacks quota_intensity_m3_per_mwh; charging water at the consumption volume")
        plants["water_charge_ratio"] = 1.0
    # Reported, never constrained.
    for column in ("withdrawal_intensity_m3_per_mwh", "withdrawal_ccs_intensity_m3_per_mwh"):
        if column not in plants.columns:
            plants[column] = 0.0
    plants["all_source_water_intensity_m3_per_mwh"] = plants["effective_cooling_technology"].map(
        assumptions.cooling_baseline_water_intensity
    )
    plants["source_dataset"] = "inputs/plants.csv"
    # Level-1 basin of the hub's own location, for the official-quota cap. Attribution is by
    # location rather than by the basin of the node a hub draws from, because that is how the
    # withdrawal permit is issued. Computed once here: it is a spatial join, and redoing it per
    # planning year per scenario would cost more than the whole rest of the preparation.
    if str(assumptions.water_budget) == "official_quota":
        from ..builders.water import _assign_basin_codes

        located = plants.rename(columns={"centroid_latitude": "latitude",
                                         "centroid_longitude": "longitude"})
        plants["basin_code"] = _assign_basin_codes(paths, located)
    return plants


def _prepare_basin_caps(paths: ProjectPaths, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    """Official 用水总量控制指标 per basin per planning year; empty unless that budget is on."""
    if str(assumptions.water_budget) != "official_quota":
        return pd.DataFrame(columns=["basin_code", "planning_year", "residual_m3_per_year"])
    path = paths.inputs_dir / "water_basin_caps.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. water_budget='official_quota' needs it; run "
            "scripts/build_water_basin_caps.py."
        )
    caps = pd.read_csv(path)
    missing = {"basin_code", "planning_year", "residual_m3_per_year"} - set(caps.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return caps


def _prepare_sector_targets(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """部门残余排放上限，各组自身 2030 基线的比例。"""
    columns = ["sector_group", "planning_year", "cap_fraction_of_2030"]
    path = paths.inputs_dir / f"sector_targets_{scenario.sector_target_source}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found for sector_target_source={scenario.sector_target_source!r}; "
            "run scripts/build_sector_targets.py"
        )
    table = pd.read_csv(path)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return table[columns].copy()


def _prepare_output_index(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[tuple[str, int], float]:
    """{(sector, year): output index, 2030 = 1}; empty dict means output is held flat."""
    source = scenario.effective_output_index_source
    if not source:
        return {}
    path = paths.inputs_dir / f"industry_output_index_{source}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found for industry_output_index_source={source!r}; "
            "run scripts/build_sector_targets.py"
        )
    table = pd.read_csv(path)
    missing = {"sector", "planning_year", "output_index"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return {
        (str(row.sector), int(row.planning_year)): float(row.output_index)
        for row in table.itertuples(index=False)
    }


def _prepare_storages(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    storages = pd.read_csv(paths.inputs_dir / "storage_hubs.csv").copy()
    if scenario.storage_scope == "dsa_only":
        storages = storages[storages["storage_type"].astype(str) == "dsa"].copy()
    storages["available_capacity_mt"] = storages["storage_all_mt"].astype(float).clip(lower=0.0)
    # Buildable injection rate from candidate-site density, not from summing the raster's
    # per-cell geological rates (see OptimizationAssumptions.storage_site_block_pixels).
    # One project per 50x50 km block of 5 km cells, at real project scale. The raster sum is
    # still applied as a geological ceiling so a hub can never exceed what the formation
    # accepts. Everything is resolved BEFORE injectivity_multiplier so SA_injectivity_half
    # still bites.
    geological_ceiling = (
        storages["injectivity_dsa_avg_mtpa"].astype(float) + storages["injectivity_eor_avg_mtpa"].astype(float)
    ).clip(lower=0.0)
    if "pixel_count" in storages.columns:
        site_count = storages["pixel_count"].astype(float) / max(1e-9, assumptions.storage_site_block_pixels)
        buildable = (site_count * assumptions.storage_site_project_rate_mtpa).clip(
            lower=0.0, upper=assumptions.max_hub_injectivity_mtpa
        )
    else:
        # Inputs without the raster cell count (e.g. hand-written test fixtures) fall back
        # to the raw rate under the ceiling.
        logger.warning("storage_hubs.csv has no pixel_count column; using raw injectivity under the hub ceiling")
        buildable = geological_ceiling.clip(upper=assumptions.max_hub_injectivity_mtpa)
    storages["injectivity_mtpa"] = (
        np.minimum(buildable, geological_ceiling) * scenario.injectivity_multiplier
    )
    # DSA capacity pays the base storage cost; EOR capacity earns the revenue credit against it.
    # Priced on the CAPACITY SPLIT rather than on `storage_type`, because that label is only a
    # majority vote whenever a hub holds both. Sinks built without a distance merge are pure, so
    # the share is 0 or 1 and this reduces to the old rule exactly; under a merge it stops
    # 7 232 Mt of EOR capacity from silently losing its credit by being outvoted.
    dsa_mt = storages["storage_dsa_mt"].astype(float).clip(lower=0.0)
    eor_mt = storages["storage_eor_mt"].astype(float).clip(lower=0.0)
    eor_share = (eor_mt / (dsa_mt + eor_mt).replace(0.0, np.nan)).fillna(0.0)
    storages["storage_cost_cny_per_t"] = (
        assumptions.storage_cost_cny_per_t - eor_share * assumptions.eor_credit_cny_per_t
    )
    storages = storages.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    storages = storages[(storages["available_capacity_mt"] > 0) | (storages["injectivity_mtpa"] > 0)].reset_index(drop=True)
    return storages


def _prepare_biomass(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, plants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    biomass = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv").copy()
    biomass["available_gj"] = biomass["available_gj"].astype(float) * scenario.biomass_supply_multiplier
    biomass["cost_cny_per_gj"] = biomass["base_cost_cny_per_gj"].astype(float) * scenario.biomass_cost_multiplier

    # Coarsen grid if configured
    if assumptions.biomass_coarse_grid_degrees > 0:
        biomass = _coarsen_resource_nodes(
            biomass, assumptions.biomass_coarse_grid_degrees,
            supply_col="available_gj", cost_col="cost_cny_per_gj",
            id_col="biomass_node_id", id_prefix="BC",
        )

    node_lons = biomass["longitude"].astype(float).to_numpy()
    node_lats = biomass["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats,
        )
        matched = np.flatnonzero(distances_km <= assumptions.resource_match_radius_km)
        for node_idx in matched:
            node = biomass.iloc[int(node_idx)]
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "biomass_node_id": str(node["biomass_node_id"]),
                "distance_km": round(float(distances_km[node_idx]), 3),
                "cost_cny_per_gj": float(node["cost_cny_per_gj"]),
            })
    biomass_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["plant_id", "biomass_node_id", "distance_km", "cost_cny_per_gj"]
    )
    logger.info("Biomass links: %d", len(biomass_links))
    return biomass, biomass_links


def _prepare_ammonia_supply(
    paths: ProjectPaths,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plants: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ammonia = pd.read_csv(paths.inputs_dir / "ammonia_supply_curve.csv").copy()
    ammonia = ammonia.replace([np.inf, -np.inf], np.nan)
    ammonia = ammonia.dropna(subset=["ammonia_node_id", "year", "nh3_supply_kg_per_year", "nh3_cost_lb_usd_per_kg"])
    ammonia["cost_cny_per_kg"] = (
        (ammonia["nh3_cost_lb_usd_per_kg"].astype(float) + scenario.ammonia_transport_adder_usd_per_kg)
        * assumptions.usd_to_cny
        * scenario.ammonia_cost_multiplier
    )

    # Coarsen per year group if configured
    coarse_deg = assumptions.ammonia_coarse_grid_degrees
    if coarse_deg > 0:
        coarsened_parts = []
        for year, year_grp in ammonia.groupby("year", sort=True):
            coarsened = _coarsen_resource_nodes(
                year_grp.reset_index(drop=True), coarse_deg,
                supply_col="nh3_supply_kg_per_year", cost_col="cost_cny_per_kg",
                id_col="ammonia_node_id", id_prefix="AC",
            )
            coarsened["year"] = int(year)
            # Carry forward nh3_cost_lb_usd_per_kg for downstream compatibility
            coarsened["nh3_cost_lb_usd_per_kg"] = coarsened["cost_cny_per_kg"] / (assumptions.usd_to_cny * scenario.ammonia_cost_multiplier) if scenario.ammonia_cost_multiplier != 0 else 0.0
            coarsened_parts.append(coarsened)
        ammonia = pd.concat(coarsened_parts, ignore_index=True)

    link_rows = []
    for year, year_nodes in ammonia.groupby("year", sort=True):
        year_nodes = year_nodes.reset_index(drop=True)
        node_lons = year_nodes["longitude"].astype(float).to_numpy()
        node_lats = year_nodes["latitude"].astype(float).to_numpy()
        for plant in plants.itertuples(index=False):
            distances_km = _haversine_distances_km(
                float(plant.centroid_longitude), float(plant.centroid_latitude),
                node_lons, node_lats,
            )
            for node_idx in np.flatnonzero(distances_km <= assumptions.resource_match_radius_km):
                node = year_nodes.iloc[int(node_idx)]
                link_rows.append({
                    "year": int(year),
                    "plant_id": str(plant.plant_id),
                    "ammonia_node_id": str(node["ammonia_node_id"]),
                    "distance_km": round(float(distances_km[node_idx]), 3),
                    "cost_cny_per_kg": float(node["cost_cny_per_kg"]),
                })
    ammonia_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["year", "plant_id", "ammonia_node_id", "distance_km", "cost_cny_per_kg"]
    )
    logger.info("Ammonia links: %d (across all years)", len(ammonia_links))
    return ammonia.sort_values(["year", "ammonia_node_id"]).reset_index(drop=True), ammonia_links


def _prepare_water(paths: ProjectPaths, plants: pd.DataFrame, assumptions: OptimizationAssumptions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from ..constants import WATER_MATCH_BUFFER_KM
    water_nodes = pd.read_csv(paths.inputs_dir / "water_nodes.csv").copy()
    water_availability = pd.read_csv(paths.inputs_dir / "water_availability.csv").copy()

    # Coarsening is done once at build time (`builders.water.coarsen_water_inputs`), which
    # groups by (basin, lon bin, lat bin) so basin budgets survive it. The runtime version
    # that used to live here grouped by bin alone and, worse, aggregated availability on
    # (node, year, scenario_family) — summing every climate member of a family into one
    # number. It was never enabled (the parameter defaults to 0 and no scenario sets it),
    # so removing it changes no result; leaving it would have been a loaded gun.
    if assumptions.water_coarse_grid_degrees > 0:
        raise ValueError(
            "water_coarse_grid_degrees is no longer honoured at solve time; set "
            "constants.WATER_COARSE_GRID_DEGREES and rebuild inputs instead."
        )

    node_lons = water_nodes["longitude"].astype(float).to_numpy()
    node_lats = water_nodes["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats,
        )
        matches = np.flatnonzero(distances_km <= WATER_MATCH_BUFFER_KM)
        sorted_indices = matches[np.argsort(distances_km[matches])]
        for rank, node_idx in enumerate(sorted_indices, 1):
            node = water_nodes.iloc[int(node_idx)]
            dist = float(distances_km[int(node_idx)])
            delivered_cost = assumptions.water_extraction_cost_cny_per_m3 + assumptions.water_transport_cost_cny_per_m3_km * dist
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "water_node_id": str(node["water_node_id"]),
                "distance_km": round(dist, 2),
                "distance_rank": rank,
                "delivered_cost_cny_per_m3": round(delivered_cost, 4),
            })
    water_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["plant_id", "water_node_id", "distance_km", "distance_rank", "delivered_cost_cny_per_m3"]
    )
    logger.info("Water links: %d", len(water_links))
    return water_nodes, water_links, water_availability


def prepare_inputs(
    paths: ProjectPaths,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
) -> PreparedInputs:
    plants = _prepare_plants(paths, scenario, assumptions)
    storages = _prepare_storages(paths, scenario, assumptions)
    biomass, biomass_links = _prepare_biomass(paths, scenario, assumptions, plants)
    ammonia_supply, ammonia_links = _prepare_ammonia_supply(paths, scenario, assumptions, plants)
    water_nodes, water_links, water_availability = _prepare_water(paths, plants, assumptions)
    water_basin_caps = _prepare_basin_caps(paths, assumptions)
    sector_targets = _prepare_sector_targets(paths, scenario)
    available_ammonia_years = tuple(sorted(ammonia_supply["year"].astype(int).unique().tolist()))
    from .industry import prepare_industry, prepare_industry_h2_links

    industry = prepare_industry(paths, assumptions, output_index=_prepare_output_index(paths, scenario))
    industry_h2_links = prepare_industry_h2_links(
        industry.hubs, ammonia_supply, float(assumptions.resource_match_radius_km)
    )
    network = build_runtime_network(
        paths, plants, storages, scenario, assumptions, industry_hubs=industry.hubs,
    )
    return PreparedInputs(
        plants=plants,
        storages=storages,
        biomass=biomass,
        biomass_links=biomass_links,
        ammonia_supply=ammonia_supply,
        ammonia_links=ammonia_links,
        water_nodes=water_nodes,
        water_links=water_links,
        water_availability=water_availability,
        water_basin_caps=water_basin_caps,
        network=network,
        available_ammonia_years=available_ammonia_years,
        industry=industry,
        sector_targets=sector_targets,
        industry_h2_links=industry_h2_links,
        inputs_dir=paths.inputs_dir,
    )

