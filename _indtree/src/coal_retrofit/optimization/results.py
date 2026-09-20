from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE
from ..constants_industry import INDUSTRY_ROUTES
from ..experiments.scenario import ScenarioRunContext
from .emissions import blend_level_to_ratio, reduction_fraction
from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import PreparedInputs, SolveState, PATHWAY_INDEX
from .industry import CCS as _CCS, H2 as _H2, UNABATED as _UNABATED


def _build_cost_breakdown(year: int, breakdown: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"year": year, "category": category, "cost_cny": cost} for category, cost in breakdown.items()])


def _build_pathway_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    captured_mt_by_plant: np.ndarray | None = None,
    blend_level_b: np.ndarray | None = None,
    blend_level_a: np.ndarray | None = None,
    year_data: dict[str, object] | None = None,
    plant_reduction_mt: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per plant x pathway shares, generation, emissions and abatement for one year.

    Generation and baseline emissions are the YEAR's values (utilisation trajectory applied)
    when `year_data` is given. `abatement_mt` is anchored to the solver's own per-plant
    reduction when `plant_reduction_mt` is given: the classic per-pathway reduction fractions
    only fix the SPLIT between a plant's pathways, and the plant total is rescaled to the
    constraint's value, which carries the CF boost and every penalty fuel. Before 2026-09-10
    the column was the unanchored classic formula and had been seen to sum to 111.7% of the
    baseline.
    """
    rows: list[dict[str, object]] = []
    gen_year = (
        np.asarray(year_data["generation"], dtype=np.float64) if year_data is not None
        else prepared.plants["annual_generation_mwh"].astype(float).to_numpy()
    )
    em_year = (
        np.asarray(year_data["emissions_mt"], dtype=np.float64) if year_data is not None
        else prepared.plants["baseline_emissions_mt"].astype(float).to_numpy()
    )
    for plant_idx, plant in enumerate(prepared.plants.itertuples(index=False)):
        baseline_emissions_mt = float(em_year[plant_idx])
        # Use solver's actual captured value if available
        actual_captured = float(captured_mt_by_plant[plant_idx]) if captured_mt_by_plant is not None else None
        bio_blend = blend_level_to_ratio(
            blend_level_b[plant_idx] if blend_level_b is not None else 0.0,
            scenario.biomass_blend_levels,
        )
        amm_blend = blend_level_to_ratio(
            blend_level_a[plant_idx] if blend_level_a is not None else 0.0,
            scenario.ammonia_blend_levels,
        )
        classic = np.array([
            baseline_emissions_mt
            * reduction_fraction(pathway, scenario.capture_rate, bio_blend, amm_blend)
            * float(share_values[plant_idx, path_idx])
            for path_idx, pathway in enumerate(PATHWAYS)
        ], dtype=np.float64)
        classic_total = float(classic.sum())
        if plant_reduction_mt is not None and abs(classic_total) > 1e-9:
            abatement = classic * (float(plant_reduction_mt[plant_idx]) / classic_total)
        elif plant_reduction_mt is not None:
            # Nothing abated on the classic view (all unabated): put the solver's value, which
            # is then a penalty-fuel correction, on the unabated column so the total is right.
            abatement = np.zeros(len(PATHWAYS))
            abatement[PATHWAY_INDEX["unabated"]] = float(plant_reduction_mt[plant_idx])
        else:
            abatement = classic
        for path_idx, pathway in enumerate(PATHWAYS):
            share = float(share_values[plant_idx, path_idx])
            # Distribute actual captured proportionally among CCS/BECCS pathways
            if actual_captured is not None and pathway in ("ccs", "beccs"):
                ccs_share = float(share_values[plant_idx, PATHWAY_INDEX["ccs"]])
                beccs_share = float(share_values[plant_idx, PATHWAY_INDEX["beccs"]])
                total_capture_share = ccs_share + beccs_share
                captured_mt = actual_captured * (share / total_capture_share) if total_capture_share > 1e-12 else 0.0
            else:
                captured_mt = 0.0
            rows.append(
                {
                    "year": year,
                    "plant_id": plant.plant_id,
                    "province_name": plant.province_name,
                    "pathway": pathway,
                    "share": share,
                    "annual_generation_mwh": float(gen_year[plant_idx]) * share,
                    "baseline_emissions_mt": baseline_emissions_mt * share,
                    "abatement_mt": float(abatement[path_idx]),
                    "captured_mt": captured_mt,
                    "enabled": scenario.path_enabled(pathway),
                }
            )
    return pd.DataFrame(rows)


def _build_province_table(pathways: pd.DataFrame) -> pd.DataFrame:
    grouped = pathways.groupby(["year", "province_name", "pathway"], as_index=False)[
        ["annual_generation_mwh", "baseline_emissions_mt", "abatement_mt", "captured_mt"]
    ].sum()
    totals = grouped.groupby(["year", "province_name"], as_index=False)["annual_generation_mwh"].sum().rename(
        columns={"annual_generation_mwh": "province_generation_mwh"}
    )
    merged = grouped.merge(totals, on=["year", "province_name"], how="left")
    merged["generation_share"] = np.where(
        merged["province_generation_mwh"] > 0,
        merged["annual_generation_mwh"] / merged["province_generation_mwh"],
        0.0,
    )
    return merged


def _build_edge_table(
    prepared: PreparedInputs,
    year: int,
    edge_flow_mtpa: np.ndarray,
    build_edge: np.ndarray,
    new_cap_mtpa: np.ndarray,
    state_before: SolveState,
    assumptions: OptimizationAssumptions,
    pipe_count: np.ndarray | None = None,
    pipe_tiers: tuple[float, ...] = (),
) -> pd.DataFrame:
    edges = prepared.network.edges.copy()
    edges["year"] = year
    edges["available_stock_before_mtpa"] = (
        edges["existing_corridor_flag"].fillna(0).astype(float) * assumptions.existing_corridor_capacity_mtpa
        + state_before.edge_added_stock_mtpa
    )
    edges["edge_flow_mtpa"] = edge_flow_mtpa
    edges["build_selected"] = np.rint(np.asarray(build_edge, dtype=np.float64)).astype(int)
    edges["new_capacity_mtpa"] = new_cap_mtpa
    edges["total_capacity_mtpa"] = edges["available_stock_before_mtpa"] + edges["new_capacity_mtpa"]
    edges["edge_active"] = ((edges["edge_flow_mtpa"] > 1e-6) | (edges["new_capacity_mtpa"] > 1e-6)).astype(int)
    edges["num_pipe_new"] = edges["new_capacity_mtpa"] / assumptions.standard_pipe_capacity_mtpa
    edges["num_pipe_stock"] = edges["total_capacity_mtpa"] / assumptions.standard_pipe_capacity_mtpa
    # Whole pipes laid this year by diameter tier, e.g. "2x2|1x20" -- what was actually built.
    if pipe_count is not None and len(pipe_tiers):
        counts = np.rint(np.asarray(pipe_count, dtype=np.float64)).astype(int)
        edges["pipes_new_by_tier"] = [
            "|".join(f"{int(counts[e, k])}x{pipe_tiers[k]:g}" for k in range(len(pipe_tiers)) if counts[e, k] > 0)
            for e in range(len(edges))
        ]
    else:
        edges["pipes_new_by_tier"] = ""
    return edges[
        [
            "year",
            "edge_id",
            "from_node_id",
            "to_node_id",
            "length_km",
            "corridor_type",
            "existing_corridor_flag",
            "edge_class",
            "source",
            "year_basis",
            "available_stock_before_mtpa",
            "build_selected",
            "edge_active",
            "edge_flow_mtpa",
            "new_capacity_mtpa",
            "total_capacity_mtpa",
            "num_pipe_new",
            "num_pipe_stock",
            "pipes_new_by_tier",
        ]
    ]


def _build_storage_table(
    prepared: PreparedInputs,
    year: int,
    storage_use_mtpa: np.ndarray,
    state_before: SolveState,
    interval_years: int,
    injectivity_mtpa: np.ndarray | None = None,
) -> pd.DataFrame:
    table = prepared.storages.copy()
    table["year"] = year
    if injectivity_mtpa is not None:
        # The year's deployed rate (buildable rate x ramp), which is what the constraint used.
        table["injectivity_mtpa"] = np.asarray(injectivity_mtpa, dtype=np.float64)
    table["storage_use_mtpa"] = storage_use_mtpa
    table["remaining_capacity_before_mt"] = state_before.remaining_storage_mt
    table["remaining_capacity_after_mt"] = np.maximum(0.0, state_before.remaining_storage_mt - storage_use_mtpa * interval_years)
    table["injectivity_utilization"] = np.where(
        table["injectivity_mtpa"].astype(float) > 0,
        storage_use_mtpa / table["injectivity_mtpa"].astype(float),
        0.0,
    )
    return table[
        [
            "year",
            "storage_hub_id",
            "province",
            "injectivity_mtpa",
            "available_capacity_mt",
            "remaining_capacity_before_mt",
            "storage_use_mtpa",
            "remaining_capacity_after_mt",
            "injectivity_utilization",
            "source",
            "year_basis",
        ]
    ]


def _build_supply_table(
    prepared: PreparedInputs,
    year: int,
    year_data: dict[str, object],
    biomass_flow_gj: np.ndarray,
    ammonia_flow_kg: np.ndarray,
    water_flow_m3: np.ndarray,
    basin_use_m3: np.ndarray | None = None,
) -> pd.DataFrame:
    biomass_links = prepared.biomass_links[["biomass_node_id"]].copy()
    biomass_links["used"] = np.asarray(biomass_flow_gj, dtype=np.float64)
    biomass_grouped = biomass_links.groupby("biomass_node_id", as_index=False)["used"].sum()
    biomass_table = prepared.biomass[["biomass_node_id", "province_name", "available_gj"]].copy()
    biomass_table = biomass_table.merge(biomass_grouped, on="biomass_node_id", how="left").fillna({"used": 0.0})
    biomass_table["year"] = year
    biomass_table["resource_type"] = "biomass"
    biomass_table["region"] = biomass_table["biomass_node_id"]
    biomass_table["available"] = biomass_table["available_gj"].astype(float)
    biomass_table["competition_scope"] = "shared_biomass_node"
    biomass_table["unit"] = "GJ/yr"

    ammonia_links = year_data["ammonia_links"][["ammonia_node_id"]].copy()
    ammonia_links["used"] = np.asarray(ammonia_flow_kg, dtype=np.float64)
    ammonia_grouped = ammonia_links.groupby("ammonia_node_id", as_index=False)["used"].sum()
    ammonia_table = year_data["ammonia_nodes"].copy()
    # `used` comes back from the solver in physical units but the availability vector is still
    # in the solver's scaled units, so it has to be un-scaled or utilisation reads 1e6.
    ammonia_table["available"] = np.asarray(year_data["ammonia_available_kg"], dtype=np.float64) * AMMONIA_FLOW_SCALE
    ammonia_table = ammonia_table.merge(ammonia_grouped, on="ammonia_node_id", how="left").fillna({"used": 0.0})
    ammonia_table["year"] = year
    ammonia_table["resource_type"] = "ammonia"
    ammonia_table["region"] = ammonia_table["ammonia_node_id"]
    ammonia_table["competition_scope"] = "shared_ammonia_node"
    ammonia_table["unit"] = "kg/yr"

    water_links = year_data["water_links"][["water_node_id"]].copy()
    water_links["used"] = np.asarray(water_flow_m3, dtype=np.float64)
    water_grouped = water_links.groupby("water_node_id", as_index=False)["used"].sum()
    water_table = year_data["water_nodes"].copy()
    water_table["available"] = np.asarray(year_data["water_available_m3"], dtype=np.float64) * WATER_FLOW_SCALE
    water_table = water_table.merge(water_grouped, on="water_node_id", how="left").fillna({"used": 0.0})
    water_table["year"] = year
    water_table["resource_type"] = "water"
    water_table["region"] = water_table["water_node_id"]
    water_table["competition_scope"] = "shared_water_node"
    water_table["unit"] = "m3/yr"

    # Official-quota basin cap: the institutional half, on the WITHDRAWAL basis. Empty
    # unless water_budget='official_quota'. Kept as its own resource_type so nothing
    # aggregates it together with the consumption-basis `water` rows above -- the two are
    # different meters and summing them is meaningless.
    frames = [
        biomass_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
        ammonia_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
        water_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
    ]
    basin_codes = list(year_data.get("water_basin_codes") or [])
    if basin_use_m3 is not None and len(basin_codes):
        used = np.asarray(basin_use_m3, dtype=np.float64)
        available = np.asarray(year_data["water_basin_available_m3"], dtype=np.float64)
        frames.append(pd.DataFrame({
            "year": year,
            "resource_type": "water_basin_quota",
            "region": basin_codes[:len(used)],
            "province_name": "",
            "competition_scope": "basin_withdrawal_cap",
            "used": used,
            "available": available[:len(used)],
            "unit": "m3/yr",
        }))
    output = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    output["utilization"] = np.where(output["available"] > 0, output["used"] / output["available"], 0.0)
    return output


def _build_sanity_checks(
    year: int,
    slacks: dict[str, object],
    pathways: pd.DataFrame,
    province_table: pd.DataFrame,
) -> pd.DataFrame:
    total_generation = float(pathways["annual_generation_mwh"].sum())
    path_shares = pathways.groupby("pathway", as_index=False)["annual_generation_mwh"].sum()
    max_path_share = float(path_shares["annual_generation_mwh"].max() / total_generation) if total_generation > 0 else 0.0
    province_generation = province_table.groupby("province_name", as_index=False)["annual_generation_mwh"].sum()
    province_peak = (
        float(province_generation["annual_generation_mwh"].max() / province_generation["annual_generation_mwh"].sum())
        if not province_generation.empty and float(province_generation["annual_generation_mwh"].sum()) > 0
        else 0.0
    )
    rows = [
        {"year": year, "check_name": "target_shortfall", "status": "fail" if slacks["target_shortfall_mt"] > 1e-6 else "pass", "metric": "mt", "value": slacks["target_shortfall_mt"], "threshold": 0.0, "detail": "Emission target slack should remain zero."},
    ]
    # Under sector targets, one row per group so the report says WHICH cap was missed.
    for group, value in sorted((slacks.get("target_shortfall_by_group") or {}).items()):
        rows.append({
            "year": year, "check_name": f"target_shortfall_{group}",
            "status": "fail" if float(value) > 1e-6 else "pass", "metric": "mt",
            "value": float(value), "threshold": 0.0,
            "detail": f"Residual cap of sector group '{group}' should be met without slack.",
        })
    rows += [
        {"year": year, "check_name": "biomass_overuse", "status": "warn" if float(np.sum(slacks["biomass_slack_gj"])) > 1e-3 else "pass", "metric": "GJ", "value": float(np.sum(slacks["biomass_slack_gj"])), "threshold": 0.0, "detail": "Biomass use should fit shared biomass-node availability within hub buffers."},
        {"year": year, "check_name": "ammonia_overuse", "status": "warn" if float(np.sum(slacks["ammonia_slack_kg"])) > 1e-3 else "pass", "metric": "kg", "value": float(np.sum(slacks["ammonia_slack_kg"])), "threshold": 0.0, "detail": "Ammonia use should fit shared ammonia-node availability under hub competition."},
        {"year": year, "check_name": "water_overuse", "status": "warn" if float(np.sum(slacks["water_slack_m3"])) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks["water_slack_m3"])), "threshold": 0.0, "detail": "Consumptive water use should fit the shared grid-water-node availability proxy under local competition."},
        # Official-quota basin cap. Always emitted -- with a zero value when the budget is off --
        # so a reader can tell "the cap held" apart from "the cap was never applied", which an
        # absent row cannot do. Any positive value means a basin's 用水总量控制指标 was breached
        # and the model paid the big-M rather than complying.
        {"year": year, "check_name": "water_basin_quota_breach", "status": "warn" if float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))), "threshold": 0.0, "detail": "Basin withdrawal should fit the official 用水总量控制指标 net of non-power use."},
        {"year": year, "check_name": "storage_or_network_stress", "status": "warn" if float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])) > 1e-6 else "pass", "metric": "aggregate_slack", "value": float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])), "threshold": 0.0, "detail": "Transport and storage slacks indicate infeasible corridor or sink assumptions."},
        {"year": year, "check_name": "single_route_lock_in", "status": "warn" if max_path_share > 0.80 else "pass", "metric": "share", "value": max_path_share, "threshold": 0.80, "detail": "A single route dominating the annual mix may indicate lock-in."},
        {"year": year, "check_name": "province_concentration", "status": "warn" if province_peak > 0.35 else "pass", "metric": "share", "value": province_peak, "threshold": 0.35, "detail": "A single province carrying too much of the result should be reviewed."},
    ]
    return pd.DataFrame(rows)


def _render_summary_markdown(
    context: ScenarioRunContext,
    scenario: OptimizationScenario,
    costs: pd.DataFrame,
    pathways: pd.DataFrame,
    sanity: pd.DataFrame,
) -> str:
    lines = [
        "# Scenario Summary",
        "",
        f"- Experiment: `{context.experiment.experiment_id}`",
        f"- Scenario: `{context.scenario.scenario_id}`",
        f"- Label: {context.scenario.label}",
        f"- Solve mode: `{scenario.solve_mode}`",
        f"- Planning years: `{', '.join(str(year) for year in scenario.planning_years)}`",
        "",
        "## Total Cost by Year",
        "",
    ]
    cost_totals = costs.groupby("year", as_index=False)["cost_cny"].sum()
    for row in cost_totals.itertuples(index=False):
        lines.append(f"- {row.year}: `{row.cost_cny:,.0f}` CNY")
    lines.extend(["", "## Pathway Share by Year", ""])
    pathway_share = pathways.groupby(["year", "pathway"], as_index=False)["annual_generation_mwh"].sum()
    year_totals = pathway_share.groupby("year", as_index=False)["annual_generation_mwh"].sum().rename(columns={"annual_generation_mwh": "total"})
    pathway_share = pathway_share.merge(year_totals, on="year", how="left")
    pathway_share["share"] = np.where(pathway_share["total"] > 0, pathway_share["annual_generation_mwh"] / pathway_share["total"], 0.0)
    for row in pathway_share.itertuples(index=False):
        lines.append(f"- {row.year} / {row.pathway}: `{row.share:.3f}`")
    lines.extend(["", "## Sanity Checks", ""])
    for row in sanity.itertuples(index=False):
        lines.append(f"- {row.year} / {row.check_name}: `{row.status}` ({row.value:.6g})")
    lines.append("")
    return "\n".join(lines)


def _build_plant_detail_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    captured_mt: np.ndarray,
    biomass_use_gj: np.ndarray,
    ammonia_use_kg: np.ndarray,
    water_use_m3: np.ndarray,
    blend_level_b: np.ndarray,
    blend_level_a: np.ndarray,
    air_share: np.ndarray | None = None,
    year_data: dict[str, object] | None = None,
    plant_reduction_mt: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-plant high-resolution detail: pathway shares, resource use, blend levels, storage proximity."""
    plants = prepared.plants
    gen_year = (
        np.asarray(year_data["generation"], dtype=np.float64) if year_data is not None
        else plants["annual_generation_mwh"].astype(float).to_numpy()
    )
    em_year = (
        np.asarray(year_data["emissions_mt"], dtype=np.float64) if year_data is not None
        else plants["baseline_emissions_mt"].astype(float).to_numpy()
    )

    # Pre-compute min distance to storage per plant via network edges
    plant_to_min_storage_km: dict[str, float] = {}
    edges = prepared.network.edges
    storage_node_set = set(prepared.network.storage_node_ids.values())
    for _, edge in edges.iterrows():
        from_id, to_id = str(edge["from_node_id"]), str(edge["to_node_id"])
        length = float(edge["length_km"])
        # Check if either end is a storage node
        for plant_id, node_id in prepared.network.plant_node_ids.items():
            if node_id == from_id and to_id in storage_node_set:
                plant_to_min_storage_km[plant_id] = min(plant_to_min_storage_km.get(plant_id, 1e9), length)
            elif node_id == to_id and from_id in storage_node_set:
                plant_to_min_storage_km[plant_id] = min(plant_to_min_storage_km.get(plant_id, 1e9), length)

    rows: list[dict[str, object]] = []
    for p in range(len(plants)):
        plant = plants.iloc[p]
        pid = str(plant["plant_id"])
        dominant_path_idx = int(np.argmax(share_values[p, :]))
        rows.append({
            "year": year,
            "plant_id": pid,
            "province_name": plant["province_name"],
            "capacity_mw": float(plant["total_capacity_mw"]),
            "centroid_longitude": float(plant["centroid_longitude"]),
            "centroid_latitude": float(plant["centroid_latitude"]),
            "retirement_year": int(plant["retirement_year"]),
            "dominant_cooling": str(plant.get("dominant_cooling_technology", "")),
            "annual_generation_mwh": float(gen_year[p]),
            "baseline_emissions_mt": float(em_year[p]),
            "reduction_mt": float(plant_reduction_mt[p]) if plant_reduction_mt is not None else float("nan"),
            # Pathway shares
            "share_unabated": float(share_values[p, PATHWAY_INDEX["unabated"]]),
            "share_retire": float(share_values[p, PATHWAY_INDEX["retire"]]),
            "share_ccs": float(share_values[p, PATHWAY_INDEX["ccs"]]),
            "share_biomass": float(share_values[p, PATHWAY_INDEX["biomass"]]),
            "share_beccs": float(share_values[p, PATHWAY_INDEX["beccs"]]),
            "share_ammonia": float(share_values[p, PATHWAY_INDEX["ammonia"]]),
            "dominant_pathway": PATHWAYS[dominant_path_idx],
            # Resource consumption
            "captured_mt": float(captured_mt[p]),
            "biomass_use_gj": float(biomass_use_gj[p]),
            "ammonia_use_kg": float(ammonia_use_kg[p]),
            "water_use_m3": float(water_use_m3[p]),
            # Fraction of this hub's generation running on dry cooling after conversion.
            # Zero everywhere when the retrofit is disabled or was not worth its capex.
            "air_cooled_share": float(air_share[p, :].sum()) if air_share is not None else 0.0,
            "already_air_share": float(plant.get("already_air_share", 0.0)),
            # Blend levels
            "biomass_blend_level": float(blend_level_b[p]),
            "ammonia_blend_level": float(blend_level_a[p]),
            # Spatial context
            "min_distance_to_storage_km": plant_to_min_storage_km.get(pid, float("nan")),
        })
    return pd.DataFrame(rows)


def _build_biomass_flow_table(
    prepared: PreparedInputs,
    year: int,
    biomass_flow_gj: np.ndarray,
) -> pd.DataFrame:
    """Per-link biomass flow: which node supplies which plant, how much."""
    links = prepared.biomass_links.copy()
    links["year"] = year
    links["flow_gj"] = np.asarray(biomass_flow_gj, dtype=np.float64)
    # Only keep active links
    active = links[links["flow_gj"] > 1e-3].copy()
    # Add plant location for mapping
    plant_loc = prepared.plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    # Add node location
    node_loc = prepared.biomass[["biomass_node_id", "longitude", "latitude", "province_name"]].copy()
    node_loc.columns = ["biomass_node_id", "node_longitude", "node_latitude", "node_province"]
    node_loc["biomass_node_id"] = node_loc["biomass_node_id"].astype(str)
    active["biomass_node_id"] = active["biomass_node_id"].astype(str)
    active = active.merge(node_loc, on="biomass_node_id", how="left")
    return active


def _build_ammonia_flow_table(
    year_data: dict[str, object],
    year: int,
    ammonia_flow_kg: np.ndarray,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """Per-link ammonia flow: which node supplies which plant, how much."""
    links = year_data["ammonia_links"].copy()
    links["year"] = year
    links["flow_kg"] = np.asarray(ammonia_flow_kg, dtype=np.float64)
    active = links[links["flow_kg"] > 1e-3].copy()
    if active.empty:
        return active
    plant_loc = plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    return active


def _build_water_flow_table(
    year_data: dict[str, object],
    year: int,
    water_flow_m3: np.ndarray,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """Per-link water flow: which water node supplies which plant, how much."""
    links = year_data["water_links"].copy()
    links["year"] = year
    links["flow_m3"] = np.asarray(water_flow_m3, dtype=np.float64)
    active = links[links["flow_m3"] > 1e-3].copy()
    if active.empty:
        return active
    plant_loc = plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    return active


def _build_slack_detail_table(
    prepared: PreparedInputs,
    year: int,
    year_data: dict[str, object],
    slacks: dict[str, object],
) -> pd.DataFrame:
    """Per-node slack values for all resource/infrastructure constraints."""
    rows: list[dict[str, object]] = []

    bio_slack = np.asarray(slacks["biomass_slack_gj"], dtype=np.float64)
    for i, node in enumerate(prepared.biomass.itertuples(index=False)):
        if bio_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "biomass_supply", "node_id": str(node.biomass_node_id), "province": str(node.province_name), "slack_value": float(bio_slack[i]), "unit": "GJ"})

    amm_slack = np.asarray(slacks["ammonia_slack_kg"], dtype=np.float64)
    amm_nodes = year_data["ammonia_nodes"]
    for i in range(len(amm_nodes)):
        if amm_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "ammonia_supply", "node_id": str(amm_nodes.iloc[i]["ammonia_node_id"]), "province": str(amm_nodes.iloc[i].get("province_name", "")), "slack_value": float(amm_slack[i]), "unit": "kg"})

    water_slack = np.asarray(slacks["water_slack_m3"], dtype=np.float64)
    water_nodes = year_data["water_nodes"]
    for i in range(len(water_nodes)):
        if water_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "water_supply", "node_id": str(water_nodes.iloc[i]["water_node_id"]), "province": str(water_nodes.iloc[i].get("province_name", "")), "slack_value": float(water_slack[i]), "unit": "m3"})

    # Official-quota basin cap. Absent (zero-length) unless water_budget='official_quota'.
    # Reported separately from `water_supply` because it is a different rule on a different
    # basis: allocation on withdrawal, against the environmental-flow limit on consumption.
    basin_slack = np.asarray(slacks.get("water_basin_slack_m3", np.zeros(0)), dtype=np.float64)
    basin_codes = list(slacks.get("water_basin_codes") or year_data.get("water_basin_codes") or [])
    for i, code in enumerate(basin_codes[:len(basin_slack)]):
        if basin_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "water_basin_quota", "node_id": str(code),
                         "province": "", "slack_value": float(basin_slack[i]), "unit": "m3"})

    inj_slack = np.asarray(slacks["injectivity_slack_mtpa"], dtype=np.float64)
    for i, hub in enumerate(prepared.storages.itertuples(index=False)):
        if inj_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "storage_injectivity", "node_id": str(hub.storage_hub_id), "province": str(hub.province), "slack_value": float(inj_slack[i]), "unit": "Mtpa"})

    cap_slack = np.asarray(slacks["storage_slack_mt"], dtype=np.float64)
    for i, hub in enumerate(prepared.storages.itertuples(index=False)):
        if cap_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "storage_capacity", "node_id": str(hub.storage_hub_id), "province": str(hub.province), "slack_value": float(cap_slack[i]), "unit": "Mt"})

    edge_slack = np.asarray(slacks["edge_slack_mtpa"], dtype=np.float64)
    for i, edge in enumerate(prepared.network.edges.itertuples(index=False)):
        if edge_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "edge_capacity", "node_id": str(edge.edge_id), "province": "", "slack_value": float(edge_slack[i]), "unit": "Mtpa"})

    if slacks["target_shortfall_mt"] > 1e-9:
        rows.append({"year": year, "constraint_type": "emission_target", "node_id": "global", "province": "", "slack_value": float(slacks["target_shortfall_mt"]), "unit": "Mt"})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["year", "constraint_type", "node_id", "province", "slack_value", "unit"])


def _build_co2_flow_direction_table(
    prepared: PreparedInputs,
    year: int,
    co2_flow_fwd: np.ndarray,
    co2_flow_bwd: np.ndarray,
) -> pd.DataFrame:
    """CO2 flow direction on each active edge: resolved to source→sink with magnitude."""
    edges = prepared.network.edges
    fwd = np.asarray(co2_flow_fwd, dtype=np.float64)
    bwd = np.asarray(co2_flow_bwd, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for i in range(len(edges)):
        net = fwd[i] - bwd[i]
        total = fwd[i] + bwd[i]
        if total < 1e-9:
            continue
        edge = edges.iloc[i]
        if net >= 0:
            source_node, sink_node = str(edge["from_node_id"]), str(edge["to_node_id"])
        else:
            source_node, sink_node = str(edge["to_node_id"]), str(edge["from_node_id"])
        rows.append({
            "year": year, "edge_id": str(edge["edge_id"]),
            "source_node": source_node, "sink_node": sink_node,
            "flow_fwd_mtpa": float(fwd[i]), "flow_bwd_mtpa": float(bwd[i]),
            "net_flow_mtpa": float(abs(net)), "length_km": float(edge["length_km"]),
            "corridor_type": str(edge.get("corridor_type", "")),
            "edge_class": str(edge.get("edge_class", "")),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["year", "edge_id", "source_node", "sink_node", "flow_fwd_mtpa", "flow_bwd_mtpa", "net_flow_mtpa", "length_km", "corridor_type", "edge_class"])


def _build_plant_cost_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    year_data: dict[str, object],
    share_values: np.ndarray,
    captured_mt: np.ndarray,
    biomass_use_gj: np.ndarray,
    water_use_m3: np.ndarray,
    blend_level_b: np.ndarray | None = None,
    blend_level_a: np.ndarray | None = None,
    prev_share_values: np.ndarray | None = None,
    retrofit_installed: np.ndarray | None = None,
    prev_retrofit_installed: np.ndarray | None = None,
    capex_pathway_indices: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Per-plant cost decomposition: computed from solved variable values.

    Undiscounted per-year view. One-time CAPEX columns are model-consistent:
    stranded asset is charged on newly retired share increments, and CCS retrofit
    CAPEX on installed-stock (max historical share) increments including the
    learning-curve cost factor — pass the previous year's share/installed values
    via prev_share_values / prev_retrofit_installed (None for the first year).
    """
    plants = prepared.plants
    n = len(plants)
    gen = np.asarray(year_data["generation"], dtype=np.float64)
    emissions = np.asarray(year_data["emissions_mt"], dtype=np.float64)
    capacity_mw = plants["total_capacity_mw"].astype(float).to_numpy()
    carbon_price = float(year_data["carbon_price"])
    retire_idx = PATHWAY_INDEX["retire"]
    # Capture-island / BECCS-increment stock coefficients (see solver); fall back to the
    # per-pathway matrix for results written before the two-stock form existed.
    stock_coeff = year_data.get("retrofit_stock_capex")

    rows: list[dict[str, object]] = []
    for p in range(n):
        share = share_values[p]
        g = float(gen[p])
        e = float(emissions[p])
        cap = float(capacity_mw[p])
        bio_blend = blend_level_to_ratio(
            blend_level_b[p] if blend_level_b is not None else 0.0,
            scenario.biomass_blend_levels,
        )
        amm_blend = blend_level_to_ratio(
            blend_level_a[p] if blend_level_a is not None else 0.0,
            scenario.ammonia_blend_levels,
        )

        # Baseline net cost: per-pathway (fuel+om-elec) matrix row × shares
        # (retrofit columns carry the CF boost, retire column is zero)
        baseline_net = sum(
            float(year_data["baseline_net_matrix"][p, k]) * float(share[k])
            for k in range(len(PATHWAYS))
        )
        # Carbon cost: price × residual emissions (approximate reporting: retrofit
        # pathways use the efficiency × CF-boost emission basis, retire avoids baseline)
        e_rt = float(year_data["emissions_retrofit_mt"][p])
        reduction_mt = (
            e * float(share[retire_idx])
            + sum(
                e_rt
                * reduction_fraction(pathway, scenario.capture_rate, bio_blend, amm_blend)
                * float(share[k])
                for k, pathway in enumerate(PATHWAYS)
                if k != retire_idx
            )
        )
        residual_mt = e - reduction_mt
        carbon_cost = carbon_price * 1e6 * residual_mt if carbon_price > 0 else 0.0
        # Coal savings (coal_savings_per_gj is scaled to CNY/TJ; biomass_use_gj is unscaled GJ)
        _bio_scale = float(year_data.get("biomass_flow_scale", 1.0))
        _cspg = year_data["coal_savings_per_gj"]
        _cspg_val = float(_cspg[p]) if hasattr(_cspg, '__getitem__') and not isinstance(_cspg, (int, float)) else float(_cspg)
        coal_savings = (_cspg_val / _bio_scale) * float(biomass_use_gj[p])
        # Incremental O&M
        incr_om = sum(float(year_data["fixed_cost_matrix"][p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # Energy penalty
        energy_pen = sum(float(year_data["energy_penalty_matrix"][p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # CCS O&M
        ccs_om = sum(float(year_data["ccs_om_matrix"][p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # Stranded asset (model-consistent: charged on newly retired share increment)
        prev_retire = float(prev_share_values[p, retire_idx]) if prev_share_values is not None else 0.0
        stranded = float(year_data["stranded_per_plant"][p]) * max(0.0, float(share[retire_idx]) - prev_retire)
        # CCS retrofit CAPEX (model-consistent: charged on installed-stock increments,
        # i.e. the max historical share; coeff already includes the learning factor)
        ccs_capex = 0.0
        for j, k in enumerate(capex_pathway_indices):
            coeff = (
                float(stock_coeff[p, j]) if stock_coeff is not None and j < np.asarray(stock_coeff).shape[1]
                else float(year_data["ccs_retrofit_capex_matrix"][p, k])
            )
            if coeff <= 0:
                continue
            if retrofit_installed is not None:
                inst = float(retrofit_installed[p, j])
                prev_inst = float(prev_retrofit_installed[p, j]) if prev_retrofit_installed is not None else 0.0
            else:
                inst = float(share[k])
                prev_inst = float(prev_share_values[p, k]) if prev_share_values is not None else 0.0
            ccs_capex += coeff * max(0.0, inst - prev_inst)

        rows.append({
            "year": year,
            "plant_id": plants.iloc[p]["plant_id"],
            "province_name": plants.iloc[p]["province_name"],
            "capacity_mw": cap,
            "baseline_net_cost_cny": baseline_net,
            "carbon_cost_cny": carbon_cost,
            "coal_savings_cny": -coal_savings,
            "incremental_om_cny": incr_om,
            "energy_penalty_cny": energy_pen,
            "ccs_om_cny": ccs_om,
            "stranded_capex_cny": stranded,
            "ccs_retrofit_capex_cny": ccs_capex,
            "total_plant_cost_cny": baseline_net + carbon_cost - coal_savings + incr_om + energy_pen + ccs_om + stranded + ccs_capex,
        })
    return pd.DataFrame(rows)


def _build_industry_detail_table(
    prepared: PreparedInputs,
    year: int,
    industry_year_data: dict | None,
    share_values: np.ndarray | None,
    prev_share_values: np.ndarray | None = None,
    h2_flow_kg: np.ndarray | None = None,
    year_data: dict[str, object] | None = None,
) -> pd.DataFrame:
    """One row per industrial hub per year: routes chosen, abatement, capture, water, cost.

    Costs follow the model's own split: `cost_annual_cny` is the O&M share of the levelised
    cost plus the hydrogen actually bought on the hub's links this year, `cost_capital_cny` the
    one-time capital charge on the route-share increment (whole share in the first year).

    Args:
        prepared: Prepared inputs; `prepared.industry` carries the hub frame.
        year: Planning year.
        industry_year_data: The year's industrial coefficient block, or None when industry off.
        share_values: Solved route shares, shape (hub_count, len(INDUSTRY_ROUTES)).
        prev_share_values: Previous year's shares (None in the first year).
        h2_flow_kg: Solved hydrogen flow per link, kg.
        year_data: The year's matrices, for the hydrogen link costs and incidence.

    Returns:
        Empty frame with the right columns when industry is off, so downstream readers get a
        frame either way.
    """
    columns = [
        "year", "hub_id", "sector", "target_group", "province", "longitude", "latitude", "basin_code",
        "output_index", "production_kt_per_year", "baseline_co2_mt", "process_co2_mt",
        "share_unabated", "share_ccs", "share_h2",
        "reduction_mt", "residual_mt", "captured_mt", "h2_kg",
        "water_m3", "water_base_m3", "water_capture_increment_m3",
        "cost_cny", "cost_capital_cny", "cost_annual_cny", "cost_h2_purchase_cny",
        "h2_price_paid_cny_per_kg", "h2_price_national_mean_cny_per_kg",
    ]
    if prepared.industry is None or industry_year_data is None or share_values is None:
        return pd.DataFrame(columns=columns)
    hubs = prepared.industry.hubs
    reduction = industry_year_data["reduction_mt"]
    baseline = industry_year_data["baseline_emissions_mt"]
    captured = industry_year_data["captured_mt"]
    water = industry_year_data["water_m3"]
    opex = industry_year_data["opex_cny"]
    capex = industry_year_data["capex_cny"]
    output_scale = industry_year_data.get("output_scale", np.ones(len(hubs)))
    h2_price_mean = float(industry_year_data["h2_price_cny_per_kg"])
    n_hubs = len(hubs)
    # Hydrogen bought per hub: link flows x link costs, folded onto hubs with the incidence.
    h2_kg_by_hub = np.zeros(n_hubs)
    h2_cost_by_hub = np.zeros(n_hubs)
    if (
        h2_flow_kg is not None and len(h2_flow_kg) and year_data is not None
        and year_data.get("industry_h2_hub_membership") is not None
    ):
        incidence = year_data["industry_h2_hub_membership"]
        flows = np.asarray(h2_flow_kg, dtype=np.float64)
        unit_cost = np.asarray(year_data["industry_h2_link_cost_cny_per_kg"], dtype=np.float64) / AMMONIA_FLOW_SCALE
        h2_kg_by_hub = np.asarray(incidence @ flows).ravel()
        h2_cost_by_hub = np.asarray(incidence @ (flows * unit_cost)).ravel()
    rows: list[dict[str, object]] = []
    for hub_idx, hub in enumerate(hubs.itertuples(index=False)):
        share = share_values[hub_idx]
        prev = prev_share_values[hub_idx] if prev_share_values is not None else np.zeros_like(share)
        annual_ccs = float(opex[hub_idx, _CCS] * share[_CCS])
        annual_h2 = max(0.0, float(opex[hub_idx, _H2] * share[_H2]) + float(h2_cost_by_hub[hub_idx]))
        capital = float(
            capex[hub_idx, _CCS] * max(0.0, share[_CCS] - prev[_CCS])
            + capex[hub_idx, _H2] * max(0.0, share[_H2] - prev[_H2])
        )
        base_water = float(water[hub_idx, _UNABATED])
        total_water = float(sum(water[hub_idx, r] * share[r] for r in range(len(INDUSTRY_ROUTES))))
        red = float(sum(reduction[hub_idx, r] * share[r] for r in range(len(INDUSTRY_ROUTES))))
        rows.append({
            "year": year,
            "hub_id": str(hub.hub_id),
            "sector": str(hub.sector),
            "target_group": str(getattr(hub, "target_group", "")),
            "province": str(hub.province),
            "longitude": float(hub.longitude),
            "latitude": float(hub.latitude),
            "basin_code": str(getattr(hub, "basin_code", "")),
            "output_index": float(output_scale[hub_idx]),
            "production_kt_per_year": float(hub.production_kt_per_year) * float(output_scale[hub_idx]),
            "baseline_co2_mt": float(baseline[hub_idx]),
            "process_co2_mt": float(hub.process_co2_mt_per_year) * float(output_scale[hub_idx]),
            "share_unabated": float(share[_UNABATED]),
            "share_ccs": float(share[_CCS]),
            "share_h2": float(share[_H2]),
            "reduction_mt": red,
            "residual_mt": float(baseline[hub_idx]) - red,
            "captured_mt": float(captured[hub_idx, _CCS] * share[_CCS]),
            "h2_kg": float(h2_kg_by_hub[hub_idx]),
            "water_m3": total_water,
            "water_base_m3": base_water,
            "water_capture_increment_m3": float(
                (water[hub_idx, _CCS] - base_water) * share[_CCS]
            ),
            "cost_cny": annual_ccs + annual_h2 + capital,
            "cost_capital_cny": capital,
            "cost_annual_cny": annual_ccs + annual_h2,
            "cost_h2_purchase_cny": float(h2_cost_by_hub[hub_idx]),
            "h2_price_paid_cny_per_kg": (
                float(h2_cost_by_hub[hub_idx] / h2_kg_by_hub[hub_idx]) if h2_kg_by_hub[hub_idx] > 1e-6 else float("nan")
            ),
            "h2_price_national_mean_cny_per_kg": h2_price_mean,
        })
    return pd.DataFrame(rows, columns=columns)
