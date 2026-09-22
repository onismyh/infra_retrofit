from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd

from ..constants import PLANNING_YEARS
from ..experiments.scenario import ScenarioRunContext
from ..paths import ProjectPaths
from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS

# Re-export shared types so external code importing from .model still works
from ._shared import PreparedInputs, SolveState, PATHWAY_INDEX, GUROBI_STATUS_NAMES  # noqa: F401
from .data_prep import prepare_inputs, _paths_df_from_assumptions  # noqa: F401
from .solver import _solve_joint_multi_period
from .results import (
    _build_cost_breakdown,
    _build_pathway_table,
    _build_province_table,
    _build_edge_table,
    _build_storage_table,
    _build_supply_table,
    _build_sanity_checks,
    _render_summary_markdown,
    _build_plant_detail_table,
    _build_biomass_flow_table,
    _build_ammonia_flow_table,
    _build_water_flow_table,
    _build_slack_detail_table,
    _build_co2_flow_direction_table,
    _build_plant_cost_table,
    _build_industry_detail_table,
)

_SCENARIO_FIELD_NAMES = {f.name for f in fields(OptimizationScenario)}
_TUPLE_FIELD_NAMES = {f.name for f in fields(OptimizationScenario) if isinstance(f.default, tuple)}
# Fields owned by the run context or by explicit alias handling below; passing them
# directly in scenario parameters is rejected (use the aliases instead).
_MANAGED_FIELD_NAMES = {
    "experiment_id", "description", "notes",
    "planning_years", "forced_pathways", "min_forced_path_share",
}
_OVERRIDABLE_FIELD_NAMES = _SCENARIO_FIELD_NAMES - _MANAGED_FIELD_NAMES


def _map_context_to_scenario(context: ScenarioRunContext) -> OptimizationScenario:
    params = dict(context.scenario.parameters)
    planning_years = (int(params.pop("year")),) if "year" in params else tuple(PLANNING_YEARS)
    force_all_pathways = bool(params.pop("force_all_pathways", False))
    min_forced_path_share = float(params.pop("min_path_share", 0.0) or 0.0)
    forced_pathways: tuple[str, ...] = ()
    if force_all_pathways:
        forced_pathways = tuple(PATHWAYS)
        params.pop("forced_pathways", None)  # force_all takes precedence
    elif "forced_pathways" in params:
        forced_pathways = tuple(
            str(item).strip().lower()
            for item in params.pop("forced_pathways") or ()
            if str(item).strip()
        )
    unknown = sorted(key for key in params if key not in _OVERRIDABLE_FIELD_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown scenario parameter(s) {unknown} in scenario "
            f"{context.scenario.scenario_id!r} of experiment {context.experiment.experiment_id!r}. "
            f"Allowed OptimizationScenario fields: {sorted(_OVERRIDABLE_FIELD_NAMES)}; "
            f"aliases: 'year', 'force_all_pathways', 'min_path_share'."
        )
    overrides: dict[str, object] = {}
    for key, value in params.items():
        if key in _TUPLE_FIELD_NAMES and isinstance(value, (list, tuple)):
            value = tuple(value)
        overrides[key] = value
    return OptimizationScenario(
        experiment_id=context.experiment.experiment_id,
        description=context.scenario.label,
        planning_years=planning_years,
        forced_pathways=forced_pathways,
        min_forced_path_share=min_forced_path_share,
        notes=context.scenario.notes,
        **overrides,
    )


def _update_state(state: SolveState, new_cap_mtpa: np.ndarray, storage_use_mtpa: np.ndarray, interval_years: int) -> SolveState:
    next_state = state.clone()
    next_state.edge_added_stock_mtpa = next_state.edge_added_stock_mtpa + new_cap_mtpa
    next_state.remaining_storage_mt = np.maximum(0.0, next_state.remaining_storage_mt - storage_use_mtpa * interval_years)
    return next_state


def run_context_model(paths: ProjectPaths, context: ScenarioRunContext) -> dict[str, object]:
    assumptions = OptimizationAssumptions()
    scenario = _map_context_to_scenario(context)
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years) or list(PLANNING_YEARS))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    joint_solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state.clone())
    solver_quality = joint_solution.get("solver_quality", {})

    pathway_tables: list[pd.DataFrame] = []
    province_tables: list[pd.DataFrame] = []
    edge_tables: list[pd.DataFrame] = []
    storage_tables: list[pd.DataFrame] = []
    supply_tables: list[pd.DataFrame] = []
    sanity_tables: list[pd.DataFrame] = []
    cost_tables: list[pd.DataFrame] = []
    plant_detail_tables: list[pd.DataFrame] = []
    biomass_flow_tables: list[pd.DataFrame] = []
    ammonia_flow_tables: list[pd.DataFrame] = []
    water_flow_tables: list[pd.DataFrame] = []
    slack_detail_tables: list[pd.DataFrame] = []
    co2_direction_tables: list[pd.DataFrame] = []
    plant_cost_tables: list[pd.DataFrame] = []
    industry_detail_tables: list[pd.DataFrame] = []
    overview_rows: list[dict[str, object]] = []
    prev_share_values: np.ndarray | None = None
    prev_retrofit_installed: np.ndarray | None = None

    for year_index, year in enumerate(years):
        interval_years = scenario.interval_years(years, year_index, assumptions)
        state_before = state.clone()
        year_solution = joint_solution["year_solutions"][year]
        year_data = year_solution["year_data"]

        pathways = _build_pathway_table(
            prepared, scenario, year, year_solution["share"],
            year_solution["captured_mt_by_plant"],
            year_solution["blend_level_b"], year_solution["blend_level_a"],
        )
        province_table = _build_province_table(pathways)
        pathway_tables.append(pathways)
        province_tables.append(province_table)
        edge_tables.append(
            _build_edge_table(
                prepared,
                year,
                year_solution["edge_flow_mtpa"],
                year_solution["build_edge"],
                year_solution["new_cap_mtpa"],
                state_before,
                assumptions,
            )
        )
        storage_tables.append(_build_storage_table(prepared, year, year_solution["storage_use_mtpa"], state_before, interval_years))
        supply_tables.append(_build_supply_table(prepared, year, year_data, year_solution["biomass_flow_gj"], year_solution["ammonia_flow_kg"], year_solution["water_flow_m3"], year_solution["slacks"].get("water_basin_use_m3")))
        sanity_tables.append(_build_sanity_checks(year, year_solution["slacks"], pathways, province_table))
        cost_tables.append(_build_cost_breakdown(year, year_solution["cost_breakdown_cny"]))
        plant_detail_tables.append(_build_plant_detail_table(
            prepared, scenario, year, year_solution["share"],
            year_solution["captured_mt_by_plant"],
            year_solution["biomass_use_gj"], year_solution["ammonia_use_kg"],
            year_solution["water_use_m3"],
            year_solution["blend_level_b"], year_solution["blend_level_a"],
            year_solution.get("air_share"),
        ))
        industry_detail_tables.append(_build_industry_detail_table(
            prepared, year, year_data.get("industry"), year_solution.get("industry_share")
        ))
        biomass_flow_tables.append(_build_biomass_flow_table(prepared, year, year_solution["biomass_flow_gj"]))
        ammonia_flow_tables.append(_build_ammonia_flow_table(year_data, year, year_solution["ammonia_flow_kg"], prepared.plants))
        water_flow_tables.append(_build_water_flow_table(year_data, year, year_solution["water_flow_m3"], prepared.plants))
        slack_detail_tables.append(_build_slack_detail_table(prepared, year, year_data, year_solution["slacks"]))
        co2_direction_tables.append(_build_co2_flow_direction_table(prepared, year, year_solution["co2_flow_fwd"], year_solution["co2_flow_bwd"]))
        plant_cost_tables.append(_build_plant_cost_table(
            prepared, scenario, assumptions, year, year_data, year_solution["share"],
            year_solution["captured_mt_by_plant"],
            year_solution["biomass_use_gj"], year_solution["water_use_m3"],
            year_solution["blend_level_b"], year_solution["blend_level_a"],
            prev_share_values=prev_share_values,
            retrofit_installed=year_solution["retrofit_installed"],
            prev_retrofit_installed=prev_retrofit_installed,
            capex_pathway_indices=joint_solution.get("capex_pathway_indices", ()),
        ))
        overview_row = {
            "year": year,
            "status": year_solution["status"],
            "objective_cny": year_solution["objective_cny"],
            "target_shortfall_mt": year_solution["slacks"]["target_shortfall_mt"],
            "global_objective_cny": float(joint_solution["objective_cny"]),
            "global_objective_bound_cny": solver_quality.get("objective_bound_cny"),
            "mip_gap": solver_quality.get("mip_gap"),
            "solver_runtime_seconds": solver_quality.get("runtime_seconds"),
            "solver_node_count": solver_quality.get("node_count"),
            "solver_solution_count": solver_quality.get("solution_count"),
        }
        overview_rows.append(overview_row)

        if scenario.carry_state_between_years:
            state = _update_state(state_before, year_solution["new_cap_mtpa"], year_solution["storage_use_mtpa"], interval_years)
        else:
            state = SolveState(edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64), remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy())
        prev_share_values = year_solution["share"]
        prev_retrofit_installed = year_solution["retrofit_installed"]

    pathways_df = pd.concat(pathway_tables, ignore_index=True, sort=False)
    province_df = pd.concat(province_tables, ignore_index=True, sort=False)
    edge_df = pd.concat(edge_tables, ignore_index=True, sort=False)
    storage_df = pd.concat(storage_tables, ignore_index=True, sort=False)
    supply_df = pd.concat(supply_tables, ignore_index=True, sort=False)
    sanity_df = pd.concat(sanity_tables, ignore_index=True, sort=False)
    costs_df = pd.concat(cost_tables, ignore_index=True, sort=False)
    plant_detail_df = pd.concat(plant_detail_tables, ignore_index=True, sort=False)
    biomass_flow_df = pd.concat(biomass_flow_tables, ignore_index=True, sort=False)
    ammonia_flow_df = pd.concat(ammonia_flow_tables, ignore_index=True, sort=False)
    water_flow_df = pd.concat(water_flow_tables, ignore_index=True, sort=False)
    slack_detail_df = pd.concat(slack_detail_tables, ignore_index=True, sort=False)
    co2_direction_df = pd.concat(co2_direction_tables, ignore_index=True, sort=False)
    plant_cost_df = pd.concat(plant_cost_tables, ignore_index=True, sort=False)
    industry_detail_df = pd.concat(industry_detail_tables, ignore_index=True, sort=False)
    overview_df = pd.DataFrame(overview_rows)
    parameter_df = _paths_df_from_assumptions(assumptions, scenario)
    summary_md = _render_summary_markdown(context, scenario, costs_df, pathways_df, sanity_df)

    return {
        "status": "ok",
        "experiment_id": context.experiment.experiment_id,
        "scenario_id": context.scenario.scenario_id,
        "description": scenario.description,
        "artifacts": {
            "overview.csv": overview_df,
            "parameters_snapshot.csv": parameter_df,
            "pathway_shares.csv": pathways_df,
            "province_pathways.csv": province_df,
            "plant_detail.csv": plant_detail_df,
            "industry_detail.csv": industry_detail_df,
            "network_edges.csv": edge_df,
            "storage_utilization.csv": storage_df,
            "resource_use.csv": supply_df,
            "biomass_flows.csv": biomass_flow_df,
            "ammonia_flows.csv": ammonia_flow_df,
            "water_flows.csv": water_flow_df,
            "co2_flow_direction.csv": co2_direction_df,
            "slack_detail.csv": slack_detail_df,
            "plant_cost.csv": plant_cost_df,
            "cost_breakdown.csv": costs_df,
            "sanity_checks.csv": sanity_df,
            "summary.md": summary_md,
        },
    }
