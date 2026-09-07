from __future__ import annotations

from dataclasses import fields

import logging

import numpy as np
import pandas as pd

from ..constants import PLANNING_YEARS
from ..paths import ProjectPaths
from .network import build_runtime_network
from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import PreparedInputs, SolveState, PATHWAY_INDEX, _nearest_year

logger = logging.getLogger(__name__)


def _paths_df_from_assumptions(assumptions: OptimizationAssumptions, scenario: OptimizationScenario) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field_info in fields(assumptions):
        rows.append({"scope": "assumption", "key": field_info.name, "value": getattr(assumptions, field_info.name)})
    for field_info in fields(scenario):
        rows.append({"scope": "scenario", "key": field_info.name, "value": getattr(scenario, field_info.name)})
    return pd.DataFrame(rows)


def _haversine_distances_km(origin_lon: float, origin_lat: float, target_lons: np.ndarray, target_lats: np.ndarray) -> np.ndarray:
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)
    target_lons_rad = np.radians(target_lons.astype(np.float64))
    target_lats_rad = np.radians(target_lats.astype(np.float64))
    delta_lon = target_lons_rad - origin_lon_rad
    delta_lat = target_lats_rad - origin_lat_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(origin_lat_rad) * np.cos(target_lats_rad) * np.sin(delta_lon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _coarsen_resource_nodes(
    nodes: pd.DataFrame,
    cell_degrees: float,
    supply_col: str,
    cost_col: str,
    id_col: str,
    id_prefix: str,
    extra_sum_cols: tuple[str, ...] = (),
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Aggregate fine-grained resource nodes into coarser grid cells.

    Args:
        nodes: DataFrame with at least lon, lat, supply, cost, id columns.
        cell_degrees: Target grid cell size in degrees. 0 = no coarsening.
        supply_col: Column name for supply quantity (summed within cell).
        cost_col: Column name for unit cost (supply-weighted average).
        id_col: Column name for node IDs (replaced by coarse cell IDs).
        id_prefix: Prefix for new coarse node IDs (e.g. "BC", "AC", "WC").
        extra_sum_cols: Additional columns to sum within each cell.

    Returns:
        Coarsened DataFrame with same essential columns but fewer rows.
    """
    if cell_degrees <= 0 or nodes.empty:
        return nodes

    df = nodes.copy()
    lons = df[lon_col].astype(float)
    lats = df[lat_col].astype(float)
    df["_ci"] = np.floor(lons / cell_degrees).astype(int)
    df["_cj"] = np.floor(lats / cell_degrees).astype(int)
    supply = df[supply_col].astype(float).clip(lower=0.0)

    rows: list[dict[str, object]] = []
    for (ci, cj), grp in df.groupby(["_ci", "_cj"]):
        w = grp[supply_col].astype(float).clip(lower=0.0)
        total_supply = float(w.sum())
        if total_supply > 0:
            avg_cost = float((w * grp[cost_col].astype(float)).sum() / total_supply)
            avg_lon = float((w * grp[lon_col].astype(float)).sum() / total_supply)
            avg_lat = float((w * grp[lat_col].astype(float)).sum() / total_supply)
        else:
            avg_cost = float(grp[cost_col].astype(float).mean())
            avg_lon = float(grp[lon_col].astype(float).mean())
            avg_lat = float(grp[lat_col].astype(float).mean())
        new_id = f"{id_prefix}_{ci:04d}_{cj:04d}"
        row: dict[str, object] = {
            id_col: new_id,
            lon_col: avg_lon,
            lat_col: avg_lat,
            supply_col: total_supply,
            cost_col: avg_cost,
        }
        # Carry province from the largest-supply node in the cell
        if "province_name" in grp.columns:
            max_idx = w.idxmax() if total_supply > 0 else grp.index[0]
            row["province_name"] = str(grp.loc[max_idx, "province_name"])
        for col in extra_sum_cols:
            if col in grp.columns:
                row[col] = float(grp[col].astype(float).sum())
        rows.append(row)

    result = pd.DataFrame(rows)
    n_before = len(nodes)
    n_after = len(result)
    logger.info("Coarsened %s: %d -> %d nodes (cell=%.1f deg)", id_prefix, n_before, n_after, cell_degrees)
    return result


def _prepare_plants(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    plants["province_name"] = plants["province_mode"].astype(str)
    # Use province-specific operating hours for generation calculation
    plants["province_cf"] = plants["province_name"].map(
        lambda prov: assumptions.province_cf(prov)
    )
    plants["annual_generation_mwh"] = plants["total_capacity_mw"].astype(float) * plants["province_cf"] * 8760.0
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
    if str(getattr(assumptions, "water_budget", "runoff")) == "official_quota":
        from ..builders.water import _assign_basin_codes

        located = plants.rename(columns={"centroid_latitude": "latitude",
                                         "centroid_longitude": "longitude"})
        plants["basin_code"] = _assign_basin_codes(paths, located)
    return plants


def _prepare_basin_caps(paths: ProjectPaths, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    """Official 用水总量控制指标 per basin per planning year; empty unless that budget is on."""
    if str(getattr(assumptions, "water_budget", "runoff")) != "official_quota":
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
    available_ammonia_years = tuple(sorted(ammonia_supply["year"].astype(int).unique().tolist()))
    network = build_runtime_network(paths, plants, storages, scenario, assumptions)
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
    )


def _water_available_by_node(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    nodes: pd.DataFrame,
) -> np.ndarray:
    """Water available to power plants at each node in `year` (m3/yr).

        available = renewable runoff x (1 - environmental flow) x (1 - existing withdrawals)

    The input table now carries one row per (node, year, climate member), where a member is
    a hydrology model x GCM x SSP combination. `scenario.water_scenario_id` selects a single
    member; leaving it empty falls back to the first member of the requested family, sorted
    by id, so a run is always reproducible. Selecting a family without naming a member used
    to silently keep whichever duplicate row landed last in the dict.
    """
    from ..constants import WATER_EXTRACTABLE_FRACTION

    frame = prepared.water_availability
    frame = frame[frame["planning_year"].astype(int) == int(year)]
    wanted = str(getattr(scenario, "water_scenario_id", "") or "")
    if wanted:
        frame = frame[frame["scenario_id"].astype(str) == wanted]
        if frame.empty:
            raise ValueError(f"water_scenario_id {wanted!r} not present in water_availability.csv")
    else:
        family = _water_scenario_family(scenario.water_mode)
        frame = frame[frame["scenario_family"].astype(str) == family]
        members = sorted(frame["scenario_id"].astype(str).unique())
        if not members:
            raise ValueError(f"No water availability rows for family {family!r} in {year}")
        if len(members) > 1:
            logger.info("water: family %s has %d members; using %s", family, len(members), members[0])
        frame = frame[frame["scenario_id"].astype(str) == members[0]]

    season = str(getattr(scenario, "water_season", "annual")).lower()
    column = "dry_season_water_m3_per_year" if season == "dry" else "available_water_m3_per_year"
    if column not in frame.columns:
        logger.warning("water: column %s missing, falling back to annual mean", column)
        column = "available_water_m3_per_year"

    lookup = frame.set_index("water_node_id")[column].to_dict()
    # Bias-correction on/off. The correction (basin_bias_factors) is baked into the
    # availability columns; this column holds the multiplicative factor, so undoing it is
    # exact for both the annual and dry-season budgets -- the correction multiplies only the
    # level, never the seasonality. Off is the "raw modelled runoff" counterfactual used to
    # quantify how much of the constraint strength owes to the correction.
    if not float(getattr(assumptions, "apply_bias_correction", True)):
        bias = frame.set_index("water_node_id")["bias_factor"].to_dict()
        lookup = {
            node_id: val / bias.get(str(node_id), 1.0)
            for node_id, val in lookup.items()
            if float(bias.get(str(node_id), 1.0)) > 0
        }
    # Depletion allowance. The constraint acts on consumption only (see `_prepare_plants`),
    # so this is the environmental-flow rule and nothing else.
    # THESE TWO PARAMETERS ARE ALIASED, AND THAT LIMITS WHAT THE MODEL CAN ATTRIBUTE.
    # The product is the only thing the solver ever sees:
    #     s = 0.85 at an extractable fraction of 0.20   ->  0.0300
    #     s = 0.00 at an extractable fraction of 0.03   ->  0.0300
    # are the same model. So a contrast built by moving `existing_withdrawal_share`
    # CANNOT distinguish "the allocation rule is what binds" from "the environmental-flow
    # standard is what binds" -- which is exactly the distinction the paper draws when it
    # reports that the institution and not the climate decides whether water binds.
    # The defensible statement is about the SIZE of the residual share left to power,
    # however that share is arrived at; the attribution between its two factors is not
    # identified by any experiment in this study and must not be claimed.
    # Fig 2(b) is the right place for this: its x-axis is the residual share, and it
    # should be read as such rather than as a pure allocation-policy axis.
    # Under the official-quota budget the allocation rule has moved out of here and into the
    # basin cap constraint, which reads it off 国办发〔2013〕2号 rather than assuming it. What
    # is left at the node is the environmental-flow rule alone -- and that de-aliases the two
    # factors the comment above says cannot otherwise be told apart.
    if str(getattr(assumptions, "water_budget", "runoff")) == "official_quota":
        usable = WATER_EXTRACTABLE_FRACTION
    else:
        usable = WATER_EXTRACTABLE_FRACTION * (1.0 - float(assumptions.existing_withdrawal_share))
    return np.array(
        [float(lookup.get(str(node_id), 0.0)) * usable * scenario.water_multiplier
         for node_id in nodes["water_node_id"].astype(str)],
        dtype=np.float64,
    )


def _withdrawal_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    water_intensity: np.ndarray,
    air_water_intensity: np.ndarray,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Per-pathway WITHDRAWAL intensity, m3/MWh, for the basin cap. None when it is inactive.

    Mirrors the consumption matrices pathway for pathway, so the two constraints see the same
    fleet through two different meters. The capture pathways take the table's own with-capture
    withdrawal rather than a multiplier, for the same reason the consumption matrix does.

    Retire is zero on both bases. Biomass and ammonia co-firing keep the base cooling system, so
    they inherit the base withdrawal scaled by the same co-firing multipliers used for
    consumption -- the fuel changes, the condenser does not.
    """
    from ..builders.water_quota import calibrated_withdrawal_intensities

    if str(getattr(assumptions, "water_budget", "runoff")) != "official_quota":
        return None, None, 1.0
    if scenario.water_mode == "no_water":
        return None, None, 1.0

    plants = prepared.plants
    hours = plants["province_mode"].astype(str).map(assumptions.province_operating_hours)
    hours = hours.fillna(assumptions.capacity_factor * 8760.0)
    generation = plants["total_capacity_mw"].astype(float) * hours
    base, capture, air_base, air_capture, factor = calibrated_withdrawal_intensities(
        plants, generation
    )
    base_np = base.to_numpy()
    capture_np = capture.to_numpy()

    withdrawal = np.zeros_like(water_intensity)
    withdrawal[:, PATHWAY_INDEX["unabated"]] = base_np
    withdrawal[:, PATHWAY_INDEX["retire"]] = 0.0
    withdrawal[:, PATHWAY_INDEX["ccs"]] = capture_np * scenario.ccs_water_multiplier_adjustment
    withdrawal[:, PATHWAY_INDEX["biomass"]] = base_np * assumptions.biomass_water_multiplier
    withdrawal[:, PATHWAY_INDEX["beccs"]] = capture_np * scenario.beccs_water_multiplier_adjustment
    withdrawal[:, PATHWAY_INDEX["ammonia"]] = base_np * assumptions.ammonia_water_multiplier

    air_base_np = air_base.to_numpy()
    air_capture_np = air_capture.to_numpy()
    air_withdrawal = np.zeros_like(withdrawal)
    air_withdrawal[:, PATHWAY_INDEX["unabated"]] = air_base_np
    air_withdrawal[:, PATHWAY_INDEX["retire"]] = 0.0
    air_withdrawal[:, PATHWAY_INDEX["ccs"]] = air_capture_np * scenario.ccs_water_multiplier_adjustment
    air_withdrawal[:, PATHWAY_INDEX["biomass"]] = air_base_np * assumptions.biomass_water_multiplier
    air_withdrawal[:, PATHWAY_INDEX["beccs"]] = air_capture_np * scenario.beccs_water_multiplier_adjustment
    air_withdrawal[:, PATHWAY_INDEX["ammonia"]] = air_base_np * assumptions.ammonia_water_multiplier
    air_withdrawal = np.minimum(air_withdrawal, withdrawal)

    logger.info(
        "water: official-quota budget active for %d, once-through calibration k=%.3f, "
        "fleet withdrawal/consumption at unabated = %.1fx",
        year, factor,
        float(withdrawal[:, PATHWAY_INDEX["unabated"]].sum()
              / max(water_intensity[:, PATHWAY_INDEX["unabated"]].sum(), 1e-9)),
    )
    return withdrawal, air_withdrawal, factor


def _basin_cap_data(
    prepared: PreparedInputs,
    assumptions: OptimizationAssumptions,
    scenario: OptimizationScenario,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """(membership, residual_m3, basin codes) for the basin cap. (None, None, []) when inactive.

    Plants are attributed to basins by LOCATION (`plants.basin_code`, set in `_prepare_plants`),
    not by the basin of the node they draw from. That is how the withdrawal permit is actually
    issued -- a plant in Hebei counts against the Hai cap whichever side of a basin line its
    intake sits on -- and it keeps the constraint exact in the pathway dimension, which
    attributing through links could not be: link flows are not split by pathway, and the
    withdrawal/consumption ratio moves ~80x between once-through and dry cooling.
    """
    if str(getattr(assumptions, "water_budget", "runoff")) != "official_quota":
        return None, None, []
    if scenario.water_mode == "no_water":
        return None, None, []

    caps = prepared.water_basin_caps
    caps = caps[caps["planning_year"].astype(int) == int(year)]
    if caps.empty:
        raise ValueError(f"water_basin_caps.csv has no rows for planning year {year}")

    plant_basins = prepared.plants["basin_code"].astype(str).to_numpy()
    codes = [str(code) for code in caps["basin_code"]]
    membership = np.zeros((len(codes), len(plant_basins)), dtype=np.float64)
    for row, code in enumerate(codes):
        membership[row, :] = (plant_basins == code).astype(np.float64)
    unmatched = int(len(plant_basins) - membership.sum())
    if unmatched:
        raise ValueError(f"{unmatched} hubs fell outside every basin in water_basin_caps.csv")
    residual = caps["residual_m3_per_year"].astype(float).to_numpy() * scenario.water_multiplier
    return membership, residual, codes


def _water_scenario_family(mode: str) -> str:
    if str(mode) == "high_water_stress":
        return "high_pressure"
    return "baseline"  # Used for both "base_water" and "grid_supply"


def _water_access_data(prepared: PreparedInputs, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, year: int) -> dict[str, object]:
    from ..constants import WATER_EXTRACTABLE_FRACTION, WATER_FLOW_SCALE
    nodes = prepared.water_nodes.copy().reset_index(drop=True)
    links = prepared.water_links.copy().reset_index(drop=True)
    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    node_index = {str(node_id): idx for idx, node_id in enumerate(nodes["water_node_id"].astype(str))}
    link_count = len(links)
    hub_membership = np.zeros((len(prepared.plants), link_count), dtype=np.float64)
    node_membership = np.zeros((len(nodes), link_count), dtype=np.float64)
    # Cost matrix: delivered cost per m3 for each link, scaled to the CHARGED volume. The
    # flow variable carries consumption (what the constraint acts on) while the tariff is
    # levied on the metered quota volume, so each link is priced at its plant's
    # quota/consumption ratio. The parametric sweep adder is applied to the physical volume,
    # not the metered one — it is a shadow price, not a tariff.
    charge_ratio = (
        prepared.plants["water_charge_ratio"].astype(float).to_numpy()
        if "water_charge_ratio" in prepared.plants.columns
        else np.ones(len(prepared.plants), dtype=np.float64)
    )
    link_cost_cny_per_m3 = np.zeros(link_count, dtype=np.float64)
    for link_idx, link in enumerate(links.itertuples(index=False)):
        if str(link.plant_id) not in plant_index or str(link.water_node_id) not in node_index:
            continue
        plant_idx = plant_index[str(link.plant_id)]
        hub_membership[plant_idx, link_idx] = 1.0
        node_membership[node_index[str(link.water_node_id)], link_idx] = 1.0
        if hasattr(link, "delivered_cost_cny_per_m3"):
            link_cost_cny_per_m3[link_idx] = float(link.delivered_cost_cny_per_m3) * charge_ratio[plant_idx]
        link_cost_cny_per_m3[link_idx] += float(scenario.water_price_adder_cny_per_m3)

    if scenario.water_mode == "no_water":
        available = None
    else:
        available = _water_available_by_node(prepared, scenario, assumptions, year, nodes)
    return {
        "nodes": nodes[["water_node_id", "province_name"]].copy(),
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "available_m3": available / WATER_FLOW_SCALE if available is not None else None,
        "link_cost_cny_per_m3": link_cost_cny_per_m3 * WATER_FLOW_SCALE,
        "water_flow_scale": WATER_FLOW_SCALE,
    }


def _biomass_access_matrices(prepared: PreparedInputs, assumptions: OptimizationAssumptions) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    from ..constants import BIOMASS_FLOW_SCALE
    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    node_index = {str(node_id): idx for idx, node_id in enumerate(prepared.biomass["biomass_node_id"].astype(str))}
    link_count = len(prepared.biomass_links)
    hub_membership = np.zeros((len(prepared.plants), link_count), dtype=np.float64)
    node_membership = np.zeros((len(prepared.biomass), link_count), dtype=np.float64)
    link_costs = np.zeros(link_count, dtype=np.float64)
    for link_idx, link in enumerate(prepared.biomass_links.itertuples(index=False)):
        if str(link.plant_id) not in plant_index or str(link.biomass_node_id) not in node_index:
            continue
        hub_membership[plant_index[str(link.plant_id)], link_idx] = 1.0
        node_membership[node_index[str(link.biomass_node_id)], link_idx] = 1.0
        # Delivered cost = purchase + pretreatment + transport(fixed + distance×variable)
        purchase_cost = float(link.cost_cny_per_gj)
        dist_km = float(link.distance_km) if hasattr(link, "distance_km") else 0.0
        transport_cost = (
            assumptions.biomass_pretreatment_cost_cny_per_gj
            + assumptions.biomass_transport_fixed_cost_cny_per_gj
            + assumptions.biomass_transport_variable_cost_cny_per_gj_km * dist_km
        )
        # Scale: cost_per_GJ * SCALE → cost per TJ
        link_costs[link_idx] = (purchase_cost + transport_cost) * BIOMASS_FLOW_SCALE
    # Scale supply: GJ → TJ (÷ SCALE)
    available_scaled = prepared.biomass["available_gj"].astype(float).to_numpy() / BIOMASS_FLOW_SCALE
    return (
        hub_membership,
        node_membership,
        available_scaled,
        link_costs,
        BIOMASS_FLOW_SCALE,
    )


def _ammonia_access_data(prepared: PreparedInputs, year: int, assumptions: OptimizationAssumptions) -> dict[str, object]:
    from ..constants import AMMONIA_FLOW_SCALE
    source_year = _nearest_year(year, prepared.available_ammonia_years)
    nodes = prepared.ammonia_supply[prepared.ammonia_supply["year"].astype(int) == int(source_year)].copy()
    links = prepared.ammonia_links[prepared.ammonia_links["year"].astype(int) == int(source_year)].copy()
    node_index = {str(node_id): idx for idx, node_id in enumerate(nodes["ammonia_node_id"].astype(str))}
    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    link_count = len(links)
    hub_membership = np.zeros((len(prepared.plants), link_count), dtype=np.float64)
    node_membership = np.zeros((len(nodes), link_count), dtype=np.float64)
    link_costs = np.zeros(link_count, dtype=np.float64)
    for link_idx, link in enumerate(links.itertuples(index=False)):
        if str(link.ammonia_node_id) not in node_index or str(link.plant_id) not in plant_index:
            continue
        hub_membership[plant_index[str(link.plant_id)], link_idx] = 1.0
        node_membership[node_index[str(link.ammonia_node_id)], link_idx] = 1.0
        # Delivered cost = production cost + distance-based truck transport
        base_cost = float(link.cost_cny_per_kg)
        dist_km = float(link.distance_km) if hasattr(link, "distance_km") else 0.0
        transport_cost = assumptions.ammonia_transport_cost_cny_per_kg_km * dist_km
        # Scale: cost_per_kg * SCALE → cost per scaled unit (kt)
        link_costs[link_idx] = (base_cost + transport_cost) * AMMONIA_FLOW_SCALE
    # Scale supply: kg → kt (÷ SCALE)
    available_scaled = (
        nodes["nh3_supply_kg_per_year"].astype(float).to_numpy() / AMMONIA_FLOW_SCALE
        if not nodes.empty else np.zeros(0, dtype=np.float64)
    )
    return {
        "year": source_year,
        "nodes": nodes,
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "available_kg": available_scaled,
        "link_cost_cny_per_kg": link_costs,
        "ammonia_flow_scale": AMMONIA_FLOW_SCALE,
    }


def _offshore_edge_mask(prepared: PreparedInputs) -> np.ndarray:
    """True for edges that touch an offshore storage hub (subsea pipeline / platform)."""
    edges = prepared.network.edges
    storages = prepared.storages
    if "offshore" not in storages.columns:
        return np.zeros(len(edges), dtype=bool)
    offshore_hubs = {
        str(hub_id)
        for hub_id, flag in zip(storages["storage_hub_id"], storages["offshore"].astype(bool))
        if flag
    }
    offshore_nodes = {
        node_id
        for hub_id, node_id in prepared.network.storage_node_ids.items()
        if str(hub_id) in offshore_hubs
    }
    if not offshore_nodes:
        return np.zeros(len(edges), dtype=bool)
    return (
        edges["from_node_id"].astype(str).isin(offshore_nodes)
        | edges["to_node_id"].astype(str).isin(offshore_nodes)
    ).to_numpy()


def _edge_capex_multiplier(edge_class: str, existing_flag: int, assumptions: OptimizationAssumptions) -> float:
    label = str(edge_class)
    if label == "runtime_direct_fallback":
        return assumptions.direct_fallback_capex_multiplier
    if existing_flag and label == "existing_main_corridor":
        return assumptions.corridor_capex_multiplier
    if label == "triangulation_candidate":
        return 1.0
    return assumptions.branch_capex_multiplier


def _build_year_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    state: SolveState,
) -> dict[str, object]:
    biomass_hub_membership, biomass_node_membership, biomass_available, biomass_link_costs, biomass_flow_scale = _biomass_access_matrices(prepared, assumptions)
    ammonia_data = _ammonia_access_data(prepared, year, assumptions)
    water_data = _water_access_data(prepared, scenario, assumptions, year)
    water_base = prepared.plants["baseline_water_intensity_m3_per_mwh"].astype(float).to_numpy()

    generation = prepared.plants["annual_generation_mwh"].astype(float).to_numpy()
    emissions_mt = prepared.plants["baseline_emissions_mt"].astype(float).to_numpy()
    # Retrofit CF boost: retrofitted pathways get boosted generation (priority dispatch)
    generation_retrofit = generation * scenario.retrofit_cf_boost
    generation_by_pathway = np.column_stack([
        generation,                    # unabated
        np.zeros_like(generation),     # retire: no generation
        generation_retrofit,           # ccs
        generation_retrofit,           # biomass
        generation_retrofit,           # beccs
        generation_retrofit,           # ammonia
    ])  # shape: (plant_count, len(PATHWAYS))
    design_retirement_year = prepared.plants["retirement_year"].astype(int).to_numpy()
    # Plants past their design life that choose site rebuild operate at rebuild_efficiency
    # (USC). eff_ratio < 1 scales their heat rate and emission intensity vs the baseline
    # fleet. Non-expired plants keep ratio 1.0.
    eff_ratio = np.where(
        year >= design_retirement_year,
        assumptions.coal_plant_base_efficiency / max(float(scenario.rebuild_efficiency), 1e-9),
        1.0,
    )
    heat_rate_eff = assumptions.heat_rate_gj_per_mwh * eff_ratio
    # Emission bases: baseline (retire avoidance + target denominator), operating
    # (efficiency-adjusted, unabated ops), retrofit (efficiency-adjusted x CF boost).
    emissions_operating_mt = emissions_mt * eff_ratio
    emissions_retrofit_mt = emissions_operating_mt * scenario.retrofit_cf_boost
    # Cost-basis generation: retire keeps baseline generation (retirement cost is priced
    # per MWh of former generation); physical-operation quantities instead use
    # generation_by_pathway, whose retire column is zero.
    generation_cost_basis = generation_by_pathway.copy()
    generation_cost_basis[:, PATHWAY_INDEX["retire"]] = generation
    pathway_fixed_costs = np.array(
        [
            assumptions.fixed_cost_cny_per_mwh("unabated"),
            assumptions.fixed_cost_cny_per_mwh("retire"),
            assumptions.fixed_cost_cny_per_mwh("ccs") * scenario.ccs_cost_multiplier,
            assumptions.fixed_cost_cny_per_mwh("biomass"),
            assumptions.fixed_cost_cny_per_mwh("beccs") * scenario.ccs_cost_multiplier,
            assumptions.fixed_cost_cny_per_mwh("ammonia"),
        ],
        dtype=np.float64,
    )
    # --- Energy penalty (efficiency loss) cost matrix ---
    # CCS penalty is fixed (depends only on share); biomass penalty is blend-level-dependent
    # and handled via McCormick products in constraints.py
    # Physical meaning: efficiency drop → more coal per MWh → extra fuel cost
    # Formula: (ε / η) × heat_rate × coal_price_per_GJ  [CNY/MWh]
    eta_coal = assumptions.coal_plant_base_efficiency
    # Province-specific coal price vector (CNY/GJ per plant)
    coal_price_per_plant = np.array([
        assumptions.province_coal_cost(str(prov))
        for prov in prepared.plants["province_name"]
    ], dtype=np.float64)
    # CCS energy penalty, expressed directly as extra fuel per unit of output and declining
    # over time (see OptimizationAssumptions.ccs_energy_penalty_ratio_by_year):
    #   extra coal cost = ratio(year) × hr_eff × coal_price   [CNY/MWh output, per plant]
    eps_ratio_ccs = assumptions.ccs_energy_penalty_ratio(year)
    ccs_penalty_per_mwh_per_plant = eps_ratio_ccs * heat_rate_eff * coal_price_per_plant
    energy_penalty_per_pathway = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    energy_penalty_matrix = generation_cost_basis * (ccs_penalty_per_mwh_per_plant[:, None] * energy_penalty_per_pathway[None, :])
    # Per-level biomass efficiency penalty coefficient for McCormick linearization (per plant)
    # penalty = G_p × β_b × (ε_per_ratio / η) × hr_eff × coal_price_p
    # ε_per_ratio is per-unit blend ratio (e.g., 0.0037 means 0.37% efficiency drop per 1% blend)
    biomass_penalty_coeff_per_level = (
        assumptions.biomass_efficiency_penalty_per_ratio / eta_coal * heat_rate_eff * coal_price_per_plant
    )

    # --- Energy penalty EMISSIONS (Mt) ---
    # The extra coal burned to cover the efficiency loss is fired in the SAME boiler, so its
    # CO2 passes through the same capture train and only the uncaptured share is vented.
    # Fan et al. 2023 (Nat Clim Change) SI eq. (S42) applies exactly this 0.1 uncaptured rate
    # to the energy-penalty emissions. Charging it at 100% (the previous behaviour) dropped
    # the effective capture rate of the CCS pathway from the nominal 90% to ~71%.
    uncaptured = 1.0 - float(scenario.capture_rate)
    emission_factor_t_per_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    ccs_penalty_emissions_per_mwh_per_plant = (
        eps_ratio_ccs * heat_rate_eff * emission_factor_t_per_gj * uncaptured / 1_000_000.0
    )
    ccs_penalty_emissions_matrix = (
        generation_cost_basis
        * (ccs_penalty_emissions_per_mwh_per_plant[:, None] * energy_penalty_per_pathway[None, :])
    )
    # Biomass co-firing penalty fuel: fully vented on the unabated-biomass pathway, but
    # captured at `capture_rate` on BECCS (same boiler, same capture train).
    biomass_penalty_emissions_coeff_per_level = (
        assumptions.biomass_efficiency_penalty_per_ratio / eta_coal * heat_rate_eff * emission_factor_t_per_gj / 1_000_000.0
    )
    beccs_penalty_emissions_coeff_per_level = biomass_penalty_emissions_coeff_per_level * uncaptured

    # --- CCS/BECCS retrofit CAPEX matrix — with learning curve ---
    lf = assumptions.ccs_learning_factor(year)
    ccs_capex_per_mw = np.array([
        0.0,
        0.0,
        assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * scenario.ccs_cost_multiplier * lf,   # ccs
        0.0,
        assumptions.beccs_retrofit_capex_cny_per_kw * 1000.0 * scenario.ccs_cost_multiplier * lf,  # beccs
        0.0,
    ], dtype=np.float64)
    capacity_mw = prepared.plants["total_capacity_mw"].astype(float).to_numpy()
    ccs_retrofit_capex_matrix = capacity_mw[:, None] * ccs_capex_per_mw[None, :]

    # --- CCS O&M matrix — with learning curve (O&M ∝ CAPEX) ---
    # Fixed annual O&M = ccs_om_fraction × (learning-adjusted) retrofit CAPEX, i.e. a
    # CNY/(kW·yr) quantity exactly as reported by An et al. 2025 SI Table 7. It is charged
    # per MW of retrofitted capacity, NOT per MWh: the previous per-MWh form divided by the
    # global fallback capacity_factor (0.55) while generation used province-specific hours,
    # so the realised O&M ranged from 0.7% (Beijing) to 5.3% (Zhejiang) of CAPEX per year.
    ccs_om_per_mw = np.array([
        0.0,
        0.0,
        assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * lf * assumptions.ccs_om_fraction,
        0.0,
        assumptions.beccs_retrofit_capex_cny_per_kw * 1000.0 * lf * assumptions.ccs_om_fraction,
        0.0,
    ], dtype=np.float64)
    ccs_om_matrix = capacity_mw[:, None] * ccs_om_per_mw[None, :]

    fixed_cost_matrix = generation_cost_basis * pathway_fixed_costs[None, :]

    # Per-pathway water intensity. Capture pathways take the table's own with-capture value
    # rather than a single multiplier: the capture increment differs by cooling system
    # (Wang 2023: x1.9-2.2 on towers and dry cooling, x1.2-1.5 on once-through), so one
    # global multiplier cannot represent both.
    capture_base = prepared.plants["capture_water_intensity_m3_per_mwh"].astype(float).to_numpy()
    water_intensity = np.zeros((len(prepared.plants), len(PATHWAYS)), dtype=np.float64)
    water_intensity[:, PATHWAY_INDEX["unabated"]] = water_base
    water_intensity[:, PATHWAY_INDEX["retire"]] = 0.0
    water_intensity[:, PATHWAY_INDEX["ccs"]] = capture_base * scenario.ccs_water_multiplier_adjustment
    water_intensity[:, PATHWAY_INDEX["biomass"]] = water_base * assumptions.biomass_water_multiplier
    water_intensity[:, PATHWAY_INDEX["beccs"]] = capture_base * scenario.beccs_water_multiplier_adjustment
    water_intensity[:, PATHWAY_INDEX["ammonia"]] = water_base * assumptions.ammonia_water_multiplier

    # Same matrix for the dry-cooled counterfactual: what each pathway would consume if this
    # hub's condensers were converted to air cooling. Retire stays zero. The retrofit is
    # modelled as a per-pathway air-cooled generation share (see `constraints`), so both
    # matrices are needed, not a scalar multiplier.
    if "air_consumption_intensity_m3_per_mwh" in prepared.plants.columns:
        air_base = prepared.plants["air_consumption_intensity_m3_per_mwh"].astype(float).to_numpy()
        air_capture = prepared.plants["air_consumption_ccs_intensity_m3_per_mwh"].astype(float).to_numpy()
    else:
        air_base = water_base.copy()
        air_capture = capture_base.copy()
    air_water_intensity = np.zeros_like(water_intensity)
    air_water_intensity[:, PATHWAY_INDEX["unabated"]] = air_base
    air_water_intensity[:, PATHWAY_INDEX["retire"]] = 0.0
    air_water_intensity[:, PATHWAY_INDEX["ccs"]] = air_capture * scenario.ccs_water_multiplier_adjustment
    air_water_intensity[:, PATHWAY_INDEX["biomass"]] = air_base * assumptions.biomass_water_multiplier
    air_water_intensity[:, PATHWAY_INDEX["beccs"]] = air_capture * scenario.beccs_water_multiplier_adjustment
    air_water_intensity[:, PATHWAY_INDEX["ammonia"]] = air_base * assumptions.ammonia_water_multiplier
    # Converting cannot be a way to consume MORE water; a seawater hub would show a negative
    # saving because its freshwater draw is already zero.
    air_water_intensity = np.minimum(air_water_intensity, water_intensity)

    # Withdrawal twins of the two matrices above, on the 水资源公报 scale. Only built when the
    # basin cap is active: it is the only constraint that acts on withdrawal, because it is the
    # only one written against a quantity that counts once-through condenser flow.
    withdrawal_intensity, air_withdrawal_intensity, once_through_factor = _withdrawal_matrices(
        prepared, scenario, assumptions, water_intensity, air_water_intensity, year
    )
    basin_membership, basin_residual, basin_codes = _basin_cap_data(
        prepared, assumptions, scenario, year
    )

    # Fraction of each hub already dry-cooled: it needs no conversion and pays no capex.
    already_air_share = (
        prepared.plants["already_air_share"].astype(float).clip(0.0, 1.0).to_numpy()
        if "already_air_share" in prepared.plants.columns
        else np.zeros(len(prepared.plants), dtype=np.float64)
    )
    # `air_share` is the share of a hub's generation priced at the ALL-dry intensity, so
    # driving it to 1 converts only the part that is still wet-cooled — the hub's baseline
    # intensity already blends in whatever is dry today. Capex and the backpressure penalty
    # therefore both scale by the still-wet fraction, or a hub that is 87% dry (Ningxia)
    # would be charged as if all of it had to be rebuilt.
    still_wet = 1.0 - already_air_share
    # Lump sum on the newly converted fraction, the same convention as the CCS retrofit
    # capex, so the two retrofit decisions are priced comparably.
    air_retrofit_capex_per_plant = (
        capacity_mw * 1000.0 * float(assumptions.air_retrofit_capex_cny_per_kw) * still_wet
    )
    # Dry cooling raises backpressure: more coal, and more CO2, per MWh delivered. Expressed
    # against the plant's own baseline emissions so it scales with the fuel actually burnt.
    penalty_ratio = float(assumptions.air_retrofit_efficiency_penalty_pp) / max(
        1e-6, float(assumptions.coal_plant_base_efficiency)
    )
    air_penalty_emissions_matrix = (
        generation_by_pathway
        * assumptions.coal_emission_factor_t_per_mwh / 1_000_000.0
        * penalty_ratio
        * still_wet[:, None]
    )
    air_penalty_emissions_matrix[:, PATHWAY_INDEX["retire"]] = 0.0
    # The same extra coal also has to be bought, not only paid for at the carbon price. At
    # the 2060 carbon price the CO2 term dominates roughly 5:1, but omitting the fuel would
    # understate the penalty by ~17% and make conversion look cheaper than it is.
    air_penalty_cost_matrix = (
        generation_cost_basis
        * (penalty_ratio * heat_rate_eff * coal_price_per_plant)[:, None]
        * still_wet[:, None]
    )
    air_penalty_cost_matrix[:, PATHWAY_INDEX["retire"]] = 0.0

    edge_base_stock = (
        prepared.network.edges["existing_corridor_flag"].fillna(0).astype(float).to_numpy() * assumptions.existing_corridor_capacity_mtpa
        + state.edge_added_stock_mtpa
    )
    edge_max_total = np.full(len(prepared.network.edges), assumptions.standard_pipe_capacity_mtpa * assumptions.max_parallel_pipes)
    edge_max_new = np.maximum(0.0, edge_max_total - edge_base_stock)
    edge_length_km = prepared.network.edges["length_km"].astype(float).to_numpy()
    offshore_edges = _offshore_edge_mask(prepared)
    offshore_factor = np.where(offshore_edges, assumptions.offshore_transport_multiplier, 1.0)
    edge_capex_coeff = (
        edge_length_km
        * assumptions.pipe_capex_cny_per_mtpa_km
        * prepared.network.edges.apply(
            lambda row: _edge_capex_multiplier(str(row["edge_class"]), int(row["existing_corridor_flag"]), assumptions),
            axis=1,
        ).astype(float).to_numpy()
        * offshore_factor
    )
    # Per-edge transport O&M coefficient (CNY per Mt of flow over the whole edge): the
    # offshore factor makes subsea routes more expensive to operate as well as to build.
    edge_route_opex_coeff = (
        edge_length_km * assumptions.route_opex_cny_per_t_km * 1_000_000.0 * offshore_factor
    )

    # --- Net system cost framework data ---
    elec_price_year = scenario.electricity_price_for_year(year)
    # Province-specific baseline operating cost (per plant); rebuilt (post-design-life)
    # plants use their improved heat rate via heat_rate_eff.
    net_operating_cost_per_mwh = (
        heat_rate_eff * coal_price_per_plant
        + assumptions.baseline_om_cost_cny_per_mwh
        - elec_price_year
    )
    # Matrix form: unabated column at baseline generation, retrofit columns carry the
    # CF boost, retire column is zero (no operation, hence no net operating cost).
    baseline_net_matrix = generation_by_pathway * net_operating_cost_per_mwh[:, None]

    carbon_price_year = scenario.carbon_price_for_year(year)

    remaining_life = np.maximum(0, design_retirement_year - year)
    fraction_remaining = np.minimum(1.0, remaining_life / max(1, assumptions.stranded_asset_accounting_life))
    capacity_mw = prepared.plants["total_capacity_mw"].astype(float).to_numpy()
    stranded_per_plant = capacity_mw * assumptions.stranded_asset_base_cny_per_kw * 1000.0 * fraction_remaining

    coal_savings_per_gj = coal_price_per_plant * biomass_flow_scale  # scaled: CNY/TJ instead of CNY/GJ

    return {
        "generation": generation,
        "generation_by_pathway": generation_by_pathway,
        "emissions_mt": emissions_mt,
        "emissions_operating_mt": emissions_operating_mt,
        "emissions_retrofit_mt": emissions_retrofit_mt,
        "heat_rate_eff": heat_rate_eff,
        "capacity_mw": capacity_mw,
        "fixed_cost_matrix": fixed_cost_matrix,
        "energy_penalty_matrix": energy_penalty_matrix,
        "biomass_penalty_coeff_per_level": biomass_penalty_coeff_per_level,
        "ccs_penalty_emissions_matrix": ccs_penalty_emissions_matrix,
        "biomass_penalty_emissions_coeff_per_level": biomass_penalty_emissions_coeff_per_level,
        "beccs_penalty_emissions_coeff_per_level": beccs_penalty_emissions_coeff_per_level,
        "ccs_retrofit_capex_matrix": ccs_retrofit_capex_matrix,
        "ccs_om_matrix": ccs_om_matrix,
        "baseline_net_matrix": baseline_net_matrix,
        "carbon_price": carbon_price_year,
        "stranded_per_plant": stranded_per_plant,
        "coal_savings_per_gj": coal_savings_per_gj,
        "biomass_link_hub_membership": biomass_hub_membership,
        "biomass_link_node_membership": biomass_node_membership,
        "biomass_available": biomass_available,
        "biomass_link_cost_cny_per_gj": biomass_link_costs,
        "biomass_flow_scale": biomass_flow_scale,
        "biomass_nodes": prepared.biomass[["biomass_node_id", "province_name"]].copy(),
        "ammonia_year": int(ammonia_data["year"]),
        "ammonia_nodes": ammonia_data["nodes"][["ammonia_node_id", "province_name"]].copy() if not ammonia_data["nodes"].empty else pd.DataFrame(columns=["ammonia_node_id", "province_name"]),
        "ammonia_links": ammonia_data["links"],
        "ammonia_link_hub_membership": ammonia_data["hub_membership"],
        "ammonia_link_node_membership": ammonia_data["node_membership"],
        "ammonia_available_kg": ammonia_data["available_kg"],
        "ammonia_link_cost_cny_per_kg": ammonia_data["link_cost_cny_per_kg"],
        "water_nodes": water_data["nodes"],
        "water_links": water_data["links"],
        "water_link_hub_membership": water_data["hub_membership"],
        "water_link_node_membership": water_data["node_membership"],
        "water_available_m3": water_data["available_m3"],
        "water_link_cost_cny_per_m3": water_data["link_cost_cny_per_m3"],
        "water_intensity": water_intensity,
        "air_water_intensity": air_water_intensity,
        # Withdrawal twins + the basin cap. All None/empty unless water_budget is
        # 'official_quota'; the solver adds the basin constraint only when they are present.
        "withdrawal_intensity": withdrawal_intensity,
        "air_withdrawal_intensity": air_withdrawal_intensity,
        "once_through_calibration": once_through_factor,
        "water_basin_membership": basin_membership,
        "water_basin_available_m3": basin_residual,
        "water_basin_codes": basin_codes,
        "already_air_share": already_air_share,
        "air_retrofit_capex_per_plant": air_retrofit_capex_per_plant,
        "air_penalty_emissions_matrix": air_penalty_emissions_matrix,
        "air_penalty_cost_matrix": air_penalty_cost_matrix,
        "allow_air_cooling_retrofit": bool(assumptions.allow_air_cooling_retrofit),
        "ammonia_flow_scale": ammonia_data.get("ammonia_flow_scale", 1.0),
        "water_flow_scale": water_data.get("water_flow_scale", 1.0),
        "edge_base_stock_mtpa": edge_base_stock,
        "edge_max_new_mtpa": edge_max_new,
        "edge_min_build_mtpa": np.minimum(edge_max_new, assumptions.standard_pipe_capacity_mtpa),
        "edge_capex_coeff": edge_capex_coeff,
        "edge_route_opex_coeff": edge_route_opex_coeff,
        "edge_offshore_mask": offshore_edges,
    }
