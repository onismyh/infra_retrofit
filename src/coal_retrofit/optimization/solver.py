from __future__ import annotations

import os
from pathlib import Path as _Path
import logging

# Numerical scaling: Gurobi solves in billion-CNY to reduce coefficient range.
# All cost expressions are divided by _COST_SCALE before entering the objective.
# Results are multiplied back in the extraction phase.
_COST_SCALE = 1e9

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None

from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import (
    PreparedInputs,
    SolveState,
    PATHWAY_INDEX,
    _new_gurobi_model,
    _var_value,
    _expr_value,
    _model_obj_value,
    _var_scalar_value,
    _extract_solver_status,
    _year_objective_weight,
    _discount_factor,
)
from .data_prep import _build_year_matrices
from .constraints import (
    _add_vector_equality,
    _add_vector_upper_bound,
    _build_air_retrofit_capex,
    _build_blend_upgrade_capex,
    _add_plant_path_constraints,
    _add_forced_pathway_activation_constraints,
)


def _optional_model_attr(model, attr_name: str) -> float | int | str | None:
    try:
        return getattr(model, attr_name)
    except Exception:
        return None


def _scale_optional_cost(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * _COST_SCALE
    except (TypeError, ValueError):
        return None


def _solver_quality(model, status: str) -> dict[str, float | int | str | None]:
    objective = _model_obj_value(model, default=float("nan"))
    sol_count = _optional_model_attr(model, "SolCount")
    try:
        sol_count_out = int(sol_count) if sol_count is not None else None
    except (TypeError, ValueError):
        sol_count_out = None
    return {
        "status": status,
        "objective_cny": objective * _COST_SCALE,
        "objective_bound_cny": _scale_optional_cost(_optional_model_attr(model, "ObjBound")),
        "mip_gap": _optional_model_attr(model, "MIPGap"),
        "runtime_seconds": _optional_model_attr(model, "Runtime"),
        "node_count": _optional_model_attr(model, "NodeCount"),
        "solution_count": sol_count_out,
        # === Run provenance =================================================================
        # Added after an audit found that published contrasts differenced two BUILDS. The
        # seed replicates of the degeneracy floor carried 632442 columns and fingerprint
        # 0xbe7b31c2 while the run they were differenced against carried 632446 and
        # 0xb8630838: inputs/water_*.csv were rewritten between the two batches, so four
        # water-supply-link variables existed in one model and not the other. Nothing in the
        # result JSON recorded that, so the difference was reported as solver degeneracy.
        # `fingerprint` is Gurobi's hash of the built model and is the single strongest
        # check: two runs meant to differ only by seed MUST agree on it.
        #
        # `threads` is the PARAMETER, and 0 means auto -- which does NOT pin anything.
        # Gurobi is deterministic only for a fixed (model, params, thread count), and the
        # audit observed 32 / 8 / 9 actual threads across runs that were all solved with the
        # parameter left at 0. A comparison is only valid when threads is > 0 and equal.
        **_run_provenance(model),
    }


def _input_digest() -> dict:
    """SHA-256 prefixes of the input tables whose VALUES define the model's right-hand side.

    Column names and array shapes are already covered by the model fingerprint and the variable
    and constraint counts. What none of those see is a change to the NUMBERS, which is exactly
    what a data correction is. Cheap (a few MB), stable, and comparable across machines.
    """
    import hashlib
    root = _Path(__file__).resolve().parents[3]
    out = {}
    for key, rel in (("water_availability", "inputs/water_availability.csv"),
                     ("water_nodes", "inputs/water_nodes.csv"),
                     ("water_links", "inputs/water_supply_links.csv"),
                     ("plants", "inputs/plants.csv")):
        path = root / rel
        try:
            out[f"digest_{key}"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        except OSError:
            out[f"digest_{key}"] = None
    return out


def _run_provenance(model) -> dict[str, object]:
    """Identify the model and the solve, so two results can be checked for comparability."""
    import os as _os

    # Threads is a PARAMETER, not a model attribute, so it is read off Params.
    try:
        threads_out = int(model.Params.Threads)
    except Exception:
        threads_out = None
    fingerprint = _optional_model_attr(model, "Fingerprint")
    try:
        fingerprint_out = hex(int(fingerprint) & 0xFFFFFFFF) if fingerprint is not None else None
    except (TypeError, ValueError):
        fingerprint_out = None
    seed_env = _os.environ.get("COAL_RETROFIT_GUROBI_SEED")
    return {
        "fingerprint": fingerprint_out,
        "num_vars": _optional_model_attr(model, "NumVars"),
        "num_constrs": _optional_model_attr(model, "NumConstrs"),
        "num_nonzeros": _optional_model_attr(model, "NumNZs"),
        "threads_param": threads_out,
        "threads_pinned": bool(threads_out),
        "seed": int(seed_env) if seed_env else 0,
        # MIPFocus is part of the (model, params, threads) tuple Gurobi is deterministic on,
        # so two runs differenced against each other must agree on it. Stamped, not assumed.
        "mip_focus": int(_os.environ.get("COAL_RETROFIT_MIPFOCUS") or 0),
        # THE INPUT VINTAGE, HASHED. Model fingerprint, dimensions, threads and seed together
        # still do not identify a run, because they are all invariant to the VALUES in the
        # input tables. The dry-season correction of 2026-08-18 changed every entry of
        # `dry_season_water_m3_per_year` (median +20.9%) and not one column name, one variable
        # count or one constraint count -- so a corrected run and a superseded run are
        # indistinguishable by every check this study had, and were placed on the same axes in
        # three main figures. The proof they are different models is a dominance violation:
        #     WA_cwatm_126_dry_wd085          (retirement cap 0.15)  incumbent 14.14216e12
        #     WA_cwatm_126_dry_wd085_capfree  (retirement cap 0.50)  BOUND     14.55719e12
        # a valid dual bound on a RELAXATION cannot exceed a feasible primal of its own
        # restriction. Hashing the water inputs makes the vintage a first-class field.
        **_input_digest(),
        "host_cpu_count": _os.cpu_count(),
    }


def _solve_joint_multi_period(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    years: tuple[int, ...],
    state: SolveState,
) -> dict[str, object]:
    if scenario.solve_mode != "joint":
        raise ValueError(f"Unsupported solve_mode {scenario.solve_mode!r}: only 'joint' is implemented.")
    plant_count = len(prepared.plants)
    storage_count = len(prepared.storages)
    edge_count = len(prepared.network.edges)
    n_nodes = len(prepared.network.nodes)
    biomass_node_count = len(prepared.biomass)
    model = _new_gurobi_model(
        "joint_multi_period",
        threads=scenario.solver_threads,
        time_limit=scenario.solver_time_limit,
    )
    # scenario.mip_gap is the single source of truth for the optimality gap.
    model.Params.MIPGap = scenario.mip_gap
    logger.info("MIPGap set to %.4f", scenario.mip_gap)
    # MIPFocus is read from the environment so a campaign can set it once for every run in the
    # snapshot without editing scenario definitions. It exists because the corrected dry-season
    # budget (see builders/water.py) loosened the water constraint enough that the root
    # heuristic stopped finding good incumbents: on the same model the gap sat at 2.84% after
    # 950 s with the bound already converged to 13839.3 and the incumbent creeping down by
    # ~1 unit per 100 s, i.e. hours to close 1%. The binding limitation is INCUMBENT QUALITY,
    # not the bound, which is exactly what MIPFocus=1 targets. It changes the (model, params,
    # threads) tuple, so it must be identical across every run that is differenced -- which is
    # why it is set per campaign and stamped into the provenance record below.
    _focus = os.environ.get("COAL_RETROFIT_MIPFOCUS")
    if _focus:
        model.Params.MIPFocus = int(_focus)
        logger.info("MIPFocus set to %s", _focus)
    year_payloads: list[dict[str, object]] = []

    retirement_years = prepared.plants["retirement_year"].astype(int).to_numpy()

    # Build node index once — shared across years
    node_idx_dict = {
        str(row.node_id): i
        for i, row in enumerate(prepared.network.nodes.itertuples(index=False))
    }
    plant_ids = prepared.plants["plant_id"].astype(str).tolist()
    storage_ids = prepared.storages["storage_hub_id"].astype(str).tolist()
    plant_n_indices: set[int] = set()
    for pid in plant_ids:
        if pid in prepared.network.plant_node_ids:
            plant_n_indices.add(node_idx_dict[prepared.network.plant_node_ids[pid]])
    storage_n_indices: set[int] = set()
    for sid in storage_ids:
        if sid in prepared.network.storage_node_ids:
            storage_n_indices.add(node_idx_dict[prepared.network.storage_node_ids[sid]])
    source_sink_indices = plant_n_indices | storage_n_indices
    pipeline_indices = [i for i in range(n_nodes) if i not in source_sink_indices]
    B = prepared.network.incidence  # (n_nodes, n_edges)

    for year_index, year in enumerate(years):
        interval_years = scenario.interval_years(years, year_index, assumptions)
        year_data = _build_year_matrices(prepared, scenario, assumptions, year, state)
        ammonia_node_count = len(year_data["ammonia_nodes"])
        water_node_count = len(year_data["water_nodes"])
        biomass_link_count = len(prepared.biomass_links)
        ammonia_link_count = len(year_data["ammonia_links"])
        water_link_count = len(year_data["water_links"])
        year_suffix = str(year)
        share = model.addMVar((plant_count, len(PATHWAYS)), lb=0.0, name=f"share_{year_suffix}")
        # Rebuild decision: binary variable for expired plants choosing site rebuild vs retirement
        rebuild = model.addMVar(plant_count, vtype=GRB.BINARY, name=f"rebuild_{year_suffix}")
        co2_flow_fwd = model.addMVar(edge_count, lb=0.0, name=f"co2_flow_fwd_{year_suffix}")
        co2_flow_bwd = model.addMVar(edge_count, lb=0.0, name=f"co2_flow_bwd_{year_suffix}")
        co2_node_outflow = model.addMVar(n_nodes, lb=-GRB.INFINITY, name=f"co2_node_outflow_{year_suffix}")
        build_edge = model.addMVar(edge_count, vtype=GRB.BINARY, name=f"build_edge_{year_suffix}")
        # add_cap[e, t] = 1 iff capacity is actually added on edge e in period t.
        # build_edge is the latched "has been built" flag tied to add_cap below.
        add_cap = model.addMVar(edge_count, vtype=GRB.BINARY, name=f"add_cap_{year_suffix}")
        new_cap_mtpa = model.addMVar(edge_count, lb=0.0, name=f"new_cap_mtpa_{year_suffix}")
        biomass_flow_gj = model.addMVar(biomass_link_count, lb=0.0, name=f"biomass_flow_gj_{year_suffix}")
        ammonia_flow_kg = model.addMVar(ammonia_link_count, lb=0.0, name=f"ammonia_flow_kg_{year_suffix}")
        water_flow_m3 = model.addMVar(water_link_count, lb=0.0, name=f"water_flow_m3_{year_suffix}")
        target_shortfall_mt = model.addVar(lb=0.0, name=f"target_shortfall_mt_{year_suffix}")
        biomass_slack_gj = model.addMVar(biomass_node_count, lb=0.0, name=f"biomass_slack_gj_{year_suffix}")
        ammonia_slack_kg = model.addMVar(ammonia_node_count, lb=0.0, name=f"ammonia_slack_kg_{year_suffix}")
        water_slack_m3 = model.addMVar(water_node_count, lb=0.0, name=f"water_slack_m3_{year_suffix}")
        injectivity_slack_mtpa = model.addMVar(storage_count, lb=0.0, name=f"injectivity_slack_mtpa_{year_suffix}")
        storage_slack_mt = model.addMVar(storage_count, lb=0.0, name=f"storage_slack_mt_{year_suffix}")
        edge_slack_mtpa = model.addMVar(edge_count, lb=0.0, name=f"edge_slack_mtpa_{year_suffix}")
        edge_flow_mtpa = model.addMVar(edge_count, lb=0.0, name=f"edge_flow_mtpa_{year_suffix}")
        storage_use_mtpa = model.addMVar(storage_count, lb=0.0, name=f"storage_use_mtpa_{year_suffix}")

        (
            captured_mt_by_plant, biomass_use_gj, ammonia_use_kg, water_use_m3, total_reduction_mt,
            select_b, select_a, blend_level_b, blend_level_a,
            total_bio_penalty, plant_reduction_exprs, air_share, air_installed,
        ) = _add_plant_path_constraints(
            model,
            share,
            year_data,
            plant_count,
            scenario,
            assumptions,
            year_suffix=year_suffix,
        )
        model.addConstrs((share[plant_idx, :].sum() == 1.0 for plant_idx in range(plant_count)), name=f"share_sum_{year_suffix}")

        # Retrofit installed-stock tracking for one-time CCS/BECCS retrofit CAPEX:
        # installed[p, j, t] = max over tau<=t of share[p, k_j, tau] (upper bound below,
        # cross-period monotonicity later). CAPEX is charged on stock increments, so a
        # temporary share dip never re-triggers the cost.
        if year_index == 0:
            capex_pathway_indices = [
                k for k in range(len(PATHWAYS))
                if float(np.max(np.asarray(year_data["ccs_retrofit_capex_matrix"])[:, k])) > 0.0
            ]
        retrofit_installed = model.addMVar(
            (plant_count, len(capex_pathway_indices)), lb=0.0, name=f"retrofit_installed_{year_suffix}"
        )
        for j, k in enumerate(capex_pathway_indices):
            model.addConstrs(
                (retrofit_installed[p, j] >= share[p, k] for p in range(plant_count)),
                name=f"retrofit_installed_lb_{k}_{year_suffix}",
            )

        # Expired plants: retire OR rebuild (site rebuild at 70% new-build cost)
        retire_idx = PATHWAY_INDEX["retire"]
        for plant_idx in range(plant_count):
            if year >= retirement_years[plant_idx]:
                # Expired plant: if rebuild=0 → must retire; if rebuild=1 → can choose any pathway
                # share[p, retire] >= 1 - rebuild[p]  →  when rebuild=0: retire>=1; when rebuild=1: retire>=0
                model.addConstr(
                    share[plant_idx, retire_idx] >= 1.0 - rebuild[plant_idx],
                    name=f"expire_retire_or_rebuild_{plant_idx}_{year_suffix}",
                )
            else:
                # Non-expired plant: cannot rebuild (rebuild=0)
                model.addConstr(rebuild[plant_idx] == 0, name=f"no_rebuild_{plant_idx}_{year_suffix}")

        # CO₂ network flow — node balance
        model.addConstr(
            co2_node_outflow == B @ co2_flow_fwd - B @ co2_flow_bwd,
            name=f"co2_flow_define_{year_suffix}",
        )
        for p_idx, plant_id in enumerate(plant_ids):
            if plant_id in prepared.network.plant_node_ids:
                mapped_node = prepared.network.plant_node_ids[plant_id]
                n_idx = node_idx_dict.get(mapped_node)
                if n_idx is None:
                    raise ValueError(f"plant_node_ids maps plant {plant_id!r} → {mapped_node!r} which is not in network nodes")
                model.addConstr(
                    co2_node_outflow[n_idx] == captured_mt_by_plant[p_idx],
                    name=f"co2_plant_inject_{p_idx}_{year_suffix}",
                )
        for s_idx, storage_id in enumerate(storage_ids):
            if storage_id in prepared.network.storage_node_ids:
                mapped_node = prepared.network.storage_node_ids[storage_id]
                n_idx = node_idx_dict.get(mapped_node)
                if n_idx is None:
                    raise ValueError(f"storage_node_ids maps hub {storage_id!r} → {mapped_node!r} which is not in network nodes")
                model.addConstr(
                    co2_node_outflow[n_idx] == -storage_use_mtpa[s_idx],
                    name=f"co2_storage_absorb_{s_idx}_{year_suffix}",
                )
        model.addConstrs(
            (co2_node_outflow[i] == 0.0 for i in pipeline_indices),
            name=f"co2_pipeline_balance_{year_suffix}",
        )

        # edge_flow_mtpa = fwd + bwd (for reporting and capacity constraints)
        model.addConstrs(
            (edge_flow_mtpa[e] == co2_flow_fwd[e] + co2_flow_bwd[e] for e in range(edge_count)),
            name=f"edge_flow_define_{year_suffix}",
        )

        biomass_plant_expr = year_data["biomass_link_hub_membership"] @ biomass_flow_gj
        biomass_node_expr = year_data["biomass_link_node_membership"] @ biomass_flow_gj
        ammonia_plant_expr = year_data["ammonia_link_hub_membership"] @ ammonia_flow_kg
        ammonia_node_expr = year_data["ammonia_link_node_membership"] @ ammonia_flow_kg
        water_plant_expr = year_data["water_link_hub_membership"] @ water_flow_m3
        water_node_expr = year_data["water_link_node_membership"] @ water_flow_m3

        _add_vector_equality(model, biomass_plant_expr, biomass_use_gj, plant_count, f"biomass_plant_balance_{year_suffix}")
        _add_vector_equality(model, ammonia_plant_expr, ammonia_use_kg, plant_count, f"ammonia_plant_balance_{year_suffix}")
        _add_vector_equality(model, water_plant_expr, water_use_m3, plant_count, f"water_plant_balance_{year_suffix}")
        _add_vector_upper_bound(
            model,
            biomass_node_expr,
            year_data["biomass_available"] + biomass_slack_gj,
            biomass_node_count,
            f"biomass_node_limit_{year_suffix}",
        )
        _add_vector_upper_bound(
            model,
            ammonia_node_expr,
            year_data["ammonia_available_kg"] + ammonia_slack_kg,
            ammonia_node_count,
            f"ammonia_node_limit_{year_suffix}",
        )
        # Water node limit — only add when water constraints are active (not no_water mode).
        # This is the PHYSICAL half: the environmental-flow rule, acting on consumption, which
        # is the quantity a depletion rule is written about.
        if year_data["water_available_m3"] is not None:
            _add_vector_upper_bound(
                model,
                water_node_expr,
                year_data["water_available_m3"] + water_slack_m3,
                water_node_count,
                f"water_node_limit_{year_suffix}",
            )
        # Basin cap — the INSTITUTIONAL half, active only under water_budget='official_quota'.
        # It acts on WITHDRAWAL because that is what 用水总量控制指标 meters (the 水资源公报
        # counts once-through condenser flow inside 工业用水), and it is written per basin
        # because the cap is a basin budget, not a per-intake limit. The two halves are the
        # de-aliased successors of `WATER_EXTRACTABLE_FRACTION x (1 - existing_withdrawal_share)`,
        # whose product was the only thing the solver used to see.
        basin_membership = year_data.get("water_basin_membership")
        water_basin_slack_m3 = None
        water_basin_use_m3 = None
        if basin_membership is not None:
            withdrawal = year_data["withdrawal_intensity"]
            air_withdrawal = year_data["air_withdrawal_intensity"]
            allow_air = bool(year_data["allow_air_cooling_retrofit"])
            generation = year_data["generation_by_pathway"]
            flow_scale = float(year_data.get("water_flow_scale", 1.0))
            plant_withdrawal = [
                gp.quicksum(
                    float(generation[plant_idx, path_idx])
                    * (
                        float(withdrawal[plant_idx, path_idx]) * share[plant_idx, path_idx]
                        - (
                            float(withdrawal[plant_idx, path_idx]
                                  - air_withdrawal[plant_idx, path_idx])
                            * air_share[plant_idx, path_idx]
                            if allow_air
                            else 0.0
                        )
                    )
                    / flow_scale
                    for path_idx in range(len(PATHWAYS))
                )
                for plant_idx in range(plant_count)
            ]
            basin_count = basin_membership.shape[0]
            water_basin_slack_m3 = model.addMVar(
                basin_count, lb=0.0, name=f"water_basin_slack_m3_{year_suffix}"
            )
            # Named variable for the basin's own withdrawal rather than an anonymous expression.
            # Nine extra columns, and it makes the quantity the whole official-quota basis turns
            # on extractable and reportable instead of something results.py has to rebuild from
            # `share` and `air_share`.
            water_basin_use_m3 = model.addMVar(
                basin_count, lb=0.0, name=f"water_basin_use_m3_{year_suffix}"
            )
            basin_available = year_data["water_basin_available_m3"] / flow_scale
            for basin_idx in range(basin_count):
                members = np.flatnonzero(basin_membership[basin_idx])
                code = year_data["water_basin_codes"][basin_idx]
                model.addConstr(
                    water_basin_use_m3[basin_idx]
                    == gp.quicksum(plant_withdrawal[int(p)] for p in members),
                    name=f"water_basin_use_{code}_{year_suffix}",
                )
                model.addConstr(
                    water_basin_use_m3[basin_idx]
                    <= float(basin_available[basin_idx]) + water_basin_slack_m3[basin_idx],
                    name=f"water_basin_limit_{code}_{year_suffix}",
                )
        model.addConstrs(
            (
                storage_use_mtpa[storage_idx]
                <= float(prepared.storages["injectivity_mtpa"].iloc[storage_idx]) + injectivity_slack_mtpa[storage_idx]
                for storage_idx in range(storage_count)
            ),
            name=f"injectivity_limit_{year_suffix}",
        )
        model.addConstr(
            total_reduction_mt + target_shortfall_mt
            >= scenario.target_for_year(year) * float(year_data["emissions_mt"].sum()),
            name=f"emission_target_{year_suffix}",
        )

        for pathway, pathway_idx in PATHWAY_INDEX.items():
            if not scenario.path_enabled(pathway):
                model.addConstrs(
                    (share[plant_idx, pathway_idx] == 0.0 for plant_idx in range(plant_count)),
                    name=f"disable_{pathway}_{year_suffix}",
                )
        _add_forced_pathway_activation_constraints(
            model, share, scenario, year_data, plant_count,
            name_suffix=f"_{year_suffix}",
            retired_mask=np.array([year >= retirement_years[p] for p in range(plant_count)]),
        )

        year_payloads.append(
            {
                "year": year,
                "interval_years": interval_years,
                "year_data": year_data,
                "share": share,
                "co2_flow_fwd": co2_flow_fwd,
                "co2_flow_bwd": co2_flow_bwd,
                "build_edge": build_edge,
                "add_cap": add_cap,
                "rebuild": rebuild,
                "retrofit_installed": retrofit_installed,
                "new_cap_mtpa": new_cap_mtpa,
                "biomass_flow_gj": biomass_flow_gj,
                "ammonia_flow_kg": ammonia_flow_kg,
                "water_flow_m3": water_flow_m3,
                "target_shortfall_mt": target_shortfall_mt,
                "biomass_slack_gj": biomass_slack_gj,
                "ammonia_slack_kg": ammonia_slack_kg,
                "water_slack_m3": water_slack_m3,
                "water_basin_slack_m3": water_basin_slack_m3,
                "water_basin_use_m3": water_basin_use_m3,
                "injectivity_slack_mtpa": injectivity_slack_mtpa,
                "storage_slack_mt": storage_slack_mt,
                "edge_slack_mtpa": edge_slack_mtpa,
                "edge_flow_mtpa": edge_flow_mtpa,
                "storage_use_mtpa": storage_use_mtpa,
                "captured_mt_by_plant": captured_mt_by_plant,
                "biomass_use_gj": biomass_use_gj,
                "ammonia_use_kg": ammonia_use_kg,
                "water_use_m3": water_use_m3,
                "air_share": air_share,
                "air_installed": air_installed,
                "select_b": select_b,
                "select_a": select_a,
                "blend_level_b": blend_level_b,
                "blend_level_a": blend_level_a,
                "plant_reduction_exprs": plant_reduction_exprs,
                "total_bio_penalty": total_bio_penalty,
            }
        )

    # Blend level monotonicity: each plant's selected level can only increase across years.
    # CDF formulation: Σ_{l'≤l} select[p,l',t+1] ≤ Σ_{l'≤l} select[p,l',t]  ∀ p, l, t
    # Blend equipment is irreversible — monotonicity holds regardless of carry_state_between_years
    if len(year_payloads) > 1:
        n_opts_b = len(scenario.biomass_blend_levels) + 1  # includes level-0 (no blend)
        n_opts_a = len(scenario.ammonia_blend_levels) + 1
        for yi in range(1, len(year_payloads)):
            sel_b_curr = year_payloads[yi]["select_b"]
            sel_b_prev = year_payloads[yi - 1]["select_b"]
            sel_a_curr = year_payloads[yi]["select_a"]
            sel_a_prev = year_payloads[yi - 1]["select_a"]
            year_sfx = str(year_payloads[yi]["year"])
            for p in range(plant_count):
                cdf_b_curr = gp.LinExpr()
                cdf_b_prev = gp.LinExpr()
                for l in range(n_opts_b - 1):
                    cdf_b_curr += sel_b_curr[p, l]
                    cdf_b_prev += sel_b_prev[p, l]
                    model.addConstr(cdf_b_curr <= cdf_b_prev, name=f"mono_b_{p}_{l}_{year_sfx}")
                cdf_a_curr = gp.LinExpr()
                cdf_a_prev = gp.LinExpr()
                for l in range(n_opts_a - 1):
                    cdf_a_curr += sel_a_curr[p, l]
                    cdf_a_prev += sel_a_prev[p, l]
                    model.addConstr(cdf_a_curr <= cdf_a_prev, name=f"mono_a_{p}_{l}_{year_sfx}")

    # Retirement monotonicity: once a plant (partially) retires, it cannot restart.
    # share[p, retire, t+1] >= share[p, retire, t]  for all plants p and consecutive years t
    retire_idx = PATHWAY_INDEX["retire"]
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            share_curr = year_payloads[yi]["share"]
            share_prev = year_payloads[yi - 1]["share"]
            yr_sfx = str(year_payloads[yi]["year"])
            model.addConstrs(
                (share_curr[p, retire_idx] >= share_prev[p, retire_idx] for p in range(plant_count)),
                name=f"retire_mono_{yr_sfx}",
            )

    # --- Inter-period pipeline constraints ---
    # 1) Build irreversibility: once built, stays built
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            be_curr = year_payloads[yi]["build_edge"]
            be_prev = year_payloads[yi - 1]["build_edge"]
            yr_sfx = str(year_payloads[yi]["year"])
            model.addConstrs(
                (be_curr[e] >= be_prev[e] for e in range(edge_count)),
                name=f"build_irreversible_{yr_sfx}",
            )

    # 2) build_edge is exactly the latched "has been built" flag:
    #    build_edge[e, t] = 1 iff capacity was added in some period <= t.
    for yi, payload in enumerate(year_payloads):
        yr_sfx = str(payload["year"])
        model.addConstrs(
            (
                payload["build_edge"][edge_idx]
                <= gp.quicksum(year_payloads[pi]["add_cap"][edge_idx] for pi in range(yi + 1))
                for edge_idx in range(edge_count)
            ),
            name=f"build_flag_tie_{yr_sfx}",
        )

    # Rebuild irreversibility: once a plant is rebuilt on site, it stays rebuilt.
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            rb_curr = year_payloads[yi]["rebuild"]
            rb_prev = year_payloads[yi - 1]["rebuild"]
            yr_sfx = str(year_payloads[yi]["year"])
            model.addConstrs(
                (rb_curr[p] >= rb_prev[p] for p in range(plant_count)),
                name=f"rebuild_irreversible_{yr_sfx}",
            )

    # Retrofit stock monotonicity: installed CCS/BECCS retrofit capacity is irreversible.
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            ri_curr = year_payloads[yi]["retrofit_installed"]
            ri_prev = year_payloads[yi - 1]["retrofit_installed"]
            yr_sfx = str(year_payloads[yi]["year"])
            for j in range(ri_curr.shape[1]):
                model.addConstrs(
                    (ri_curr[p, j] >= ri_prev[p, j] for p in range(plant_count)),
                    name=f"retrofit_stock_mono_{j}_{yr_sfx}",
                )

    first_year_data = year_payloads[0]["year_data"]
    edge_base_stock = np.asarray(first_year_data["edge_base_stock_mtpa"], dtype=np.float64)
    edge_max_new_total = np.asarray(first_year_data["edge_max_new_mtpa"], dtype=np.float64)
    edge_min_build = np.asarray(first_year_data["edge_min_build_mtpa"], dtype=np.float64)
    edge_buildable = (edge_max_new_total > 1e-9).astype(float)

    for payload in year_payloads:
        year_suffix = str(payload["year"])
        build_edge = payload["build_edge"]
        add_cap = payload["add_cap"]
        new_cap_mtpa = payload["new_cap_mtpa"]
        edge_flow_mtpa = payload["edge_flow_mtpa"]
        edge_slack_mtpa = payload["edge_slack_mtpa"]
        storage_use_mtpa = payload["storage_use_mtpa"]
        storage_slack_mt = payload["storage_slack_mt"]
        interval_years = int(payload["interval_years"])

        model.addConstrs((build_edge[edge_idx] <= edge_buildable[edge_idx] for edge_idx in range(edge_count)), name=f"edge_buildable_{year_suffix}")
        # Min/max new-capacity limits are tied to add_cap (per-period "adding capacity now"),
        # NOT to the latched build_edge flag — otherwise the min-build rule would force
        # repeated >= min_build additions in every period after the edge is first built.
        model.addConstrs(
            (new_cap_mtpa[edge_idx] <= edge_max_new_total[edge_idx] * add_cap[edge_idx] for edge_idx in range(edge_count)),
            name=f"edge_new_cap_limit_{year_suffix}",
        )
        model.addConstrs(
            (new_cap_mtpa[edge_idx] >= edge_min_build[edge_idx] * add_cap[edge_idx] for edge_idx in range(edge_count)),
            name=f"edge_min_build_{year_suffix}",
        )
        model.addConstrs(
            (add_cap[edge_idx] <= build_edge[edge_idx] for edge_idx in range(edge_count)),
            name=f"edge_add_implies_build_{year_suffix}",
        )

        year_position = years.index(int(payload["year"]))
        current_year = int(payload["year"])
        lifetime = assumptions.pipeline_lifetime_years
        if scenario.carry_state_between_years:
            # Only count capacity from past years that is still within pipeline lifetime
            alive_indices = [
                past_idx for past_idx in range(year_position + 1)
                if current_year - int(year_payloads[past_idx]["year"]) < lifetime
            ]
            model.addConstrs(
                (
                    gp.quicksum(year_payloads[pi]["new_cap_mtpa"][edge_idx] for pi in range(year_position + 1))
                    <= edge_max_new_total[edge_idx]
                    for edge_idx in range(edge_count)
                ),
                name=f"edge_total_new_cap_limit_{year_suffix}",
            )
            model.addConstrs(
                (
                    edge_flow_mtpa[edge_idx]
                    <= edge_base_stock[edge_idx]
                    + gp.quicksum(year_payloads[pi]["new_cap_mtpa"][edge_idx] for pi in alive_indices)
                    + edge_slack_mtpa[edge_idx]
                    for edge_idx in range(edge_count)
                ),
                name=f"edge_capacity_limit_{year_suffix}",
            )
            model.addConstrs(
                (
                    gp.quicksum(
                        year_payloads[past_idx]["storage_use_mtpa"][storage_idx] * int(year_payloads[past_idx]["interval_years"])
                        for past_idx in range(year_position + 1)
                    )
                    <= float(state.remaining_storage_mt[storage_idx]) + storage_slack_mt[storage_idx]
                    for storage_idx in range(storage_count)
                ),
                name=f"storage_capacity_limit_{year_suffix}",
            )
        else:
            model.addConstrs(
                (
                    edge_flow_mtpa[edge_idx]
                    <= edge_base_stock[edge_idx] + new_cap_mtpa[edge_idx] + edge_slack_mtpa[edge_idx]
                    for edge_idx in range(edge_count)
                ),
                name=f"edge_capacity_limit_{year_suffix}",
            )
            model.addConstrs(
                (
                    storage_use_mtpa[storage_idx] * interval_years
                    <= float(state.remaining_storage_mt[storage_idx]) + storage_slack_mt[storage_idx]
                    for storage_idx in range(storage_count)
                ),
                name=f"storage_capacity_limit_{year_suffix}",
            )

        year_data = payload["year_data"]
        interval_weight = _year_objective_weight(interval_years, scenario.discount_rate)
        # NB: use the payload's own year, not the loop variable from model construction.
        df = _discount_factor(current_year, scenario.discount_base_year, scenario.discount_rate)
        yr_sfx = str(payload["year"])
        retire_idx = PATHWAY_INDEX["retire"]
        plant_reduction_exprs = payload["plant_reduction_exprs"]
        total_bio_penalty = payload["total_bio_penalty"]

        # === Baseline net operating cost: (coal + O&M - electricity) per pathway ===
        # Matrix row per plant: unabated at baseline generation, retrofit pathways with
        # the CF boost (and rebuilt-plant heat rate), retire column zero.
        baseline_net = gp.quicksum(
            float(year_data["baseline_net_matrix"][p, k]) * payload["share"][p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )

        # === Carbon cost: carbon_price × (E_p - reduction_p) × 1e6 ===
        carbon_price_t = float(year_data["carbon_price"])
        carbon_cost = carbon_price_t * 1e6 * gp.quicksum(
            float(year_data["emissions_mt"][p]) - plant_reduction_exprs[p]
            for p in range(plant_count)
        ) if carbon_price_t > 0 else 0.0

        # === Coal savings from biomass substitution (negative cost, per-plant coal price) ===
        coal_savings_vec = year_data["coal_savings_per_gj"]
        coal_savings = gp.quicksum(
            float(coal_savings_vec[p]) * payload["biomass_use_gj"][p] for p in range(plant_count)
        )

        # === Incremental O&M (pathway-specific, above baseline) ===
        incremental_om = gp.quicksum(
            float(year_data["fixed_cost_matrix"][p, k]) * payload["share"][p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )

        # === Energy penalty (CCS fixed + biomass blend-level-dependent) ===
        energy_penalty_cost = gp.quicksum(
            float(year_data["energy_penalty_matrix"][p, k]) * payload["share"][p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )
        # Backpressure penalty of dry cooling, on the converted share only.
        if year_data.get("air_penalty_cost_matrix") is not None and bool(
            year_data.get("allow_air_cooling_retrofit", False)
        ):
            energy_penalty_cost = energy_penalty_cost + gp.quicksum(
                float(year_data["air_penalty_cost_matrix"][p, k]) * payload["air_share"][p, k]
                for p in range(plant_count) for k in range(len(PATHWAYS))
            )
        ccs_om_cost = gp.quicksum(
            float(year_data["ccs_om_matrix"][p, k]) * payload["share"][p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )

        # === Resource procurement ===
        biomass_cost = gp.quicksum(
            float(year_data["biomass_link_cost_cny_per_gj"][link_idx]) * payload["biomass_flow_gj"][link_idx]
            for link_idx in range(len(prepared.biomass_links))
        )
        ammonia_cost = gp.quicksum(
            float(year_data["ammonia_link_cost_cny_per_kg"][link_idx]) * payload["ammonia_flow_kg"][link_idx]
            for link_idx in range(len(year_data["ammonia_links"]))
        )
        # Water supply cost (active in grid_supply mode; zero in no_water/base_water)
        water_link_cost = year_data.get("water_link_cost_cny_per_m3", np.zeros(0))
        water_cost = gp.quicksum(
            float(water_link_cost[link_idx]) * payload["water_flow_m3"][link_idx]
            for link_idx in range(len(year_data["water_links"]))
        ) if len(water_link_cost) > 0 else 0.0

        # === CO2 transport & storage ===
        # Per-edge coefficient (CNY per Mt over the edge) already carries length, the unit
        # O&M rate and the offshore multiplier; see data_prep._build_year_matrices.
        edge_opex_coeff = year_data["edge_route_opex_coeff"]
        transport_opex = gp.quicksum(
            float(edge_opex_coeff[e])
            * (payload["co2_flow_fwd"][e] + payload["co2_flow_bwd"][e])
            for e in range(edge_count)
        )
        storage_cost_lookup = prepared.storages["storage_cost_cny_per_t"].astype(float).to_numpy()
        storage_cost = gp.quicksum(
            float(storage_cost_lookup[s]) * 1_000_000.0 * payload["storage_use_mtpa"][s]
            for s in range(storage_count)
        )

        # === Pipeline CAPEX ===
        pipe_capex = gp.quicksum(
            float(year_data["edge_capex_coeff"][edge_idx]) * payload["new_cap_mtpa"][edge_idx]
            for edge_idx in range(edge_count)
        )

        # === Slack penalties (resource slacks in scaled units, multiply by scale factor) ===
        _amm_scale = float(year_data.get("ammonia_flow_scale", 1.0))
        _wat_scale = float(year_data.get("water_flow_scale", 1.0))
        _bio_scale = float(year_data.get("biomass_flow_scale", 1.0))
        slack_cost = (
            payload["target_shortfall_mt"] * assumptions.slack_penalty_cny_per_unit
            + payload["biomass_slack_gj"].sum() * 2_000.0 * _bio_scale
            + payload["ammonia_slack_kg"].sum() * 1_000.0 * _amm_scale
            + payload["water_slack_m3"].sum() * 1_000.0 * _wat_scale
            # Same unit penalty as the node slack: violating the allocation cap and
            # violating the environmental-flow limit must cost the same, or the solver
            # would rank one institution above the other for a purely numerical reason.
            + (payload["water_basin_slack_m3"].sum() * 1_000.0 * _wat_scale
               if payload.get("water_basin_slack_m3") is not None else 0.0)
            + payload["injectivity_slack_mtpa"].sum() * assumptions.slack_penalty_cny_per_unit
            + payload["storage_slack_mt"].sum() * assumptions.slack_penalty_cny_per_unit
            + payload["edge_slack_mtpa"].sum() * assumptions.slack_penalty_cny_per_unit
        )

        # === One-time CAPEX (stranded asset, CCS retrofit, blend upgrade) ===
        capacity_mw = year_data["capacity_mw"]
        if year_position == 0:
            # Stranded asset: full share_retire × stranded_per_plant
            stranded_capex = gp.quicksum(
                float(year_data["stranded_per_plant"][p]) * payload["share"][p, retire_idx]
                for p in range(plant_count)
                if float(year_data["stranded_per_plant"][p]) > 0
            )
            # CCS retrofit CAPEX on the installed stock (equals share in the first period)
            ccs_retrofit_capex = gp.quicksum(
                float(year_data["ccs_retrofit_capex_matrix"][p, k_j]) * payload["retrofit_installed"][p, j]
                for j, k_j in enumerate(capex_pathway_indices) for p in range(plant_count)
            )
            blend_upgrade_capex = _build_blend_upgrade_capex(
                model, capacity_mw, payload["blend_level_b"], payload["blend_level_a"],
                assumptions, plant_count, sfx=f"_{yr_sfx}",
            )
            air_retrofit_capex = _build_air_retrofit_capex(
                model, year_data, payload, plant_count, yr_sfx, prev_payload=None
            )
        else:
            prev_payload = year_payloads[year_position - 1]
            # Stranded asset: incremental delta_pos for newly retired share
            stranded_terms = []
            for p in range(plant_count):
                coeff = float(year_data["stranded_per_plant"][p])
                if coeff <= 0:
                    continue
                delta_ret = model.addVar(lb=0.0, name=f"stranded_delta_{p}_{yr_sfx}")
                model.addConstr(
                    delta_ret >= payload["share"][p, retire_idx] - prev_payload["share"][p, retire_idx],
                    name=f"stranded_delta_lb_{p}_{yr_sfx}",
                )
                stranded_terms.append(coeff * delta_ret)
            stranded_capex = gp.quicksum(stranded_terms) if stranded_terms else 0.0
            # CCS retrofit CAPEX: charged on the increment of the installed stock
            # (max historical share), not the period-over-period share delta — a share
            # dip followed by a rebound does NOT re-trigger the sunk retrofit cost.
            prev_installed = prev_payload["retrofit_installed"]
            ccs_retrofit_capex = gp.quicksum(
                float(year_data["ccs_retrofit_capex_matrix"][p, k_j])
                * (payload["retrofit_installed"][p, j] - prev_installed[p, j])
                for j, k_j in enumerate(capex_pathway_indices) for p in range(plant_count)
            )
            blend_upgrade_capex = _build_blend_upgrade_capex(
                model, capacity_mw, payload["blend_level_b"], payload["blend_level_a"],
                assumptions, plant_count,
                prev_blend_level_b=prev_payload["blend_level_b"],
                prev_blend_level_a=prev_payload["blend_level_a"],
                sfx=f"_{yr_sfx}",
            )
            air_retrofit_capex = _build_air_retrofit_capex(
                model, year_data, payload, plant_count, yr_sfx, prev_payload=prev_payload
            )

        # === Rebuild CAPEX for expired plants choosing site rebuild ===
        # Cost = capacity_MW × new_build_cost × rebuild_fraction × 1000 (kW→MW)
        # One-time cost: charged only when rebuild is first activated (delta vs previous
        # period). rebuild is latched (irreversible), so the delta is 1 only in the
        # activation period — charging payload["rebuild"] directly would re-charge the
        # full CAPEX in every subsequent period the rebuilt plant keeps operating.
        rebuild_cost_per_mw = assumptions.stranded_asset_base_cny_per_kw * scenario.rebuild_capex_fraction * 1000.0
        if year_position == 0:
            rebuild_capex = gp.quicksum(
                float(capacity_mw[p]) * rebuild_cost_per_mw * payload["rebuild"][p]
                for p in range(plant_count)
                if current_year >= retirement_years[p]
            )
        else:
            rebuild_capex_terms = []
            for p in range(plant_count):
                if current_year < retirement_years[p]:
                    continue
                delta_rebuild = model.addVar(lb=0.0, name=f"rebuild_delta_{p}_{yr_sfx}")
                model.addConstr(
                    delta_rebuild >= payload["rebuild"][p] - prev_payload["rebuild"][p],
                    name=f"rebuild_delta_lb_{p}_{yr_sfx}",
                )
                rebuild_capex_terms.append(float(capacity_mw[p]) * rebuild_cost_per_mw * delta_rebuild)
            rebuild_capex = gp.quicksum(rebuild_capex_terms) if rebuild_capex_terms else 0.0

        # === Retirement rate cap (excludes forced retirements from design life) ===
        # Normalized: divide both sides by total_gen so coefficients are generation shares [0, ~0.005]
        # and RHS is just the rate parameter (0.15). Avoids 4e7 matrix coefficients.
        if scenario.max_new_retirement_share_per_period > 0:
            generation = year_data["generation"]
            total_gen = float(generation.sum())
            voluntary_plants = [p for p in range(plant_count) if int(payload["year"]) < retirement_years[p]]
            if voluntary_plants and total_gen > 0:
                if year_position == 0:
                    model.addConstr(
                        gp.quicksum(float(generation[p]) / total_gen * payload["share"][p, retire_idx] for p in voluntary_plants)
                        <= scenario.max_new_retirement_share_per_period,
                        name=f"max_retire_rate_{yr_sfx}",
                    )
                else:
                    model.addConstr(
                        gp.quicksum(
                            float(generation[p]) / total_gen * (payload["share"][p, retire_idx] - prev_payload["share"][p, retire_idx])
                            for p in voluntary_plants
                        ) <= scenario.max_new_retirement_share_per_period,
                        name=f"max_retire_rate_{yr_sfx}",
                    )

        payload["cost_exprs"] = {
            "baseline_net_cost":   df * interval_weight * baseline_net / _COST_SCALE,
            "carbon_cost":         df * interval_weight * carbon_cost / _COST_SCALE if carbon_price_t > 0 else 0.0,
            "coal_savings_credit": df * interval_weight * (-coal_savings) / _COST_SCALE,
            "energy_penalty_cost": df * interval_weight * (energy_penalty_cost + total_bio_penalty) / _COST_SCALE,
            "ccs_om_cost":         df * interval_weight * ccs_om_cost / _COST_SCALE,
            "incremental_om":      df * interval_weight * incremental_om / _COST_SCALE,
            "biomass_cost":        df * interval_weight * biomass_cost / _COST_SCALE,
            "ammonia_cost":        df * interval_weight * ammonia_cost / _COST_SCALE,
            "water_cost":          df * interval_weight * water_cost / _COST_SCALE,
            "transport_opex":      df * interval_weight * transport_opex / _COST_SCALE,
            "storage_cost":        df * interval_weight * storage_cost / _COST_SCALE,
            "stranded_capex":      df * stranded_capex / _COST_SCALE,
            "ccs_retrofit_capex":  df * ccs_retrofit_capex / _COST_SCALE,
            "pipe_capex":          df * pipe_capex / _COST_SCALE,
            "blend_upgrade_capex": df * blend_upgrade_capex / _COST_SCALE,
            "air_retrofit_capex": df * air_retrofit_capex / _COST_SCALE,
            "rebuild_capex":      df * rebuild_capex / _COST_SCALE,
            "slack_penalty":       df * interval_weight * slack_cost / _COST_SCALE,
        }
        payload["objective_expr"] = gp.quicksum(list(payload["cost_exprs"].values()))

    model.setObjective(gp.quicksum(payload["objective_expr"] for payload in year_payloads), GRB.MINIMIZE)
    model.optimize()
    status = _extract_solver_status(model)
    solver_quality = _solver_quality(model, status)
    has_solution = bool(solver_quality.get("solution_count") or 0)
    acceptable_status = status in ("optimal", "suboptimal", "solution_limit") or (status == "time_limit" and has_solution)

    if not acceptable_status:
        logger.error("Solver returned status '%s' — results will be zero-filled.", status)
        # Return empty year_solutions so caller can detect failure
        return {
            "status": status,
            "objective_cny": float("nan"),
            "solver_quality": solver_quality,
            "capex_pathway_indices": tuple(capex_pathway_indices),
            "year_solutions": {
                int(p["year"]): {
                    "status": status,
                    "objective_cny": 0.0,
                    "share": np.zeros((plant_count, len(PATHWAYS))),
                    "build_edge": np.zeros(edge_count),
                    "rebuild": np.zeros(plant_count),
                    "new_cap_mtpa": np.zeros(edge_count),
                    "edge_flow_mtpa": np.zeros(edge_count),
                    "storage_use_mtpa": np.zeros(storage_count),
                    "water_use_m3": np.zeros(plant_count),
                    "water_flow_m3": np.zeros(len(p["year_data"]["water_links"])),
                    "biomass_use_gj": np.zeros(plant_count),
                    "biomass_flow_gj": np.zeros(len(prepared.biomass_links)),
                    "ammonia_use_kg": np.zeros(plant_count),
                    "ammonia_flow_kg": np.zeros(len(p["year_data"]["ammonia_links"])),
                    "captured_mt_by_plant": np.zeros(plant_count),
                    "air_share": np.zeros((plant_count, len(PATHWAYS))),
                    "air_installed": np.zeros(plant_count),
                    "blend_level_b": np.zeros(plant_count),
                    "blend_level_a": np.zeros(plant_count),
                    "retrofit_installed": np.zeros((plant_count, len(capex_pathway_indices))),
                    "co2_flow_fwd": np.zeros(edge_count),
                    "co2_flow_bwd": np.zeros(edge_count),
                    "cost_breakdown_cny": {k: 0.0 for k in list(year_payloads[0]["cost_exprs"].keys())},
                    "slacks": {
                        "target_shortfall_mt": 0.0,
                        "biomass_slack_gj": np.zeros(len(prepared.biomass)),
                        "ammonia_slack_kg": np.zeros(len(p["year_data"]["ammonia_nodes"])),
                        "water_slack_m3": np.zeros(len(p["year_data"]["water_nodes"])),
                        "water_basin_slack_m3": np.zeros(
                            len(p["year_data"].get("water_basin_codes") or [])),
                        "water_basin_use_m3": np.zeros(
                            len(p["year_data"].get("water_basin_codes") or [])),
                        "water_basin_codes": list(p["year_data"].get("water_basin_codes") or []),
                        "injectivity_slack_mtpa": np.zeros(storage_count),
                        "storage_slack_mt": np.zeros(storage_count),
                        "edge_slack_mtpa": np.zeros(edge_count),
                    },
                    "year_data": p["year_data"],
                }
                for p in year_payloads
            },
        }

    year_solutions: dict[int, dict[str, object]] = {}

    for payload in year_payloads:
        year = int(payload["year"])
        year_data = payload["year_data"]
        biomass_node_count = len(prepared.biomass)
        ammonia_node_count = len(year_data["ammonia_nodes"])
        water_node_count = len(year_data["water_nodes"])
        # Unscale resource flows back to original units (GJ, kg, m³) for output
        _amm_s = float(year_data.get("ammonia_flow_scale", 1.0))
        _wat_s = float(year_data.get("water_flow_scale", 1.0))
        _bio_s = float(year_data.get("biomass_flow_scale", 1.0))
        year_solutions[year] = {
            "status": status,
            "objective_cny": _expr_value(payload["objective_expr"]) * _COST_SCALE,
            "share": _var_value(payload["share"], (plant_count, len(PATHWAYS))),
            "build_edge": _var_value(payload["build_edge"], edge_count),
            "rebuild": _var_value(payload["rebuild"], plant_count),
            "new_cap_mtpa": _var_value(payload["new_cap_mtpa"], edge_count),
            "edge_flow_mtpa": _var_value(payload["edge_flow_mtpa"], edge_count),
            "co2_flow_fwd": _var_value(payload["co2_flow_fwd"], edge_count),
            "co2_flow_bwd": _var_value(payload["co2_flow_bwd"], edge_count),
            "storage_use_mtpa": _var_value(payload["storage_use_mtpa"], storage_count),
            "water_use_m3": _var_value(payload["water_use_m3"], plant_count) * _wat_s,
            "water_flow_m3": _var_value(payload["water_flow_m3"], len(year_data["water_links"])) * _wat_s,
            "biomass_use_gj": _var_value(payload["biomass_use_gj"], plant_count) * _bio_s,
            "biomass_flow_gj": _var_value(payload["biomass_flow_gj"], len(prepared.biomass_links)) * _bio_s,
            "ammonia_use_kg": _var_value(payload["ammonia_use_kg"], plant_count) * _amm_s,
            "ammonia_flow_kg": _var_value(payload["ammonia_flow_kg"], len(year_data["ammonia_links"])) * _amm_s,
            "captured_mt_by_plant": _var_value(payload["captured_mt_by_plant"], plant_count),
            "air_share": _var_value(payload["air_share"], (plant_count, len(PATHWAYS))),
            "air_installed": _var_value(payload["air_installed"], plant_count),
            "blend_level_b": _var_value(payload["blend_level_b"], plant_count),
            "blend_level_a": _var_value(payload["blend_level_a"], plant_count),
            "retrofit_installed": _var_value(payload["retrofit_installed"], (plant_count, len(capex_pathway_indices))),
            "cost_breakdown_cny": {category: _expr_value(expr) * _COST_SCALE for category, expr in payload["cost_exprs"].items()},
            "slacks": {
                "target_shortfall_mt": _var_scalar_value(payload["target_shortfall_mt"]),
                "biomass_slack_gj": _var_value(payload["biomass_slack_gj"], biomass_node_count) * _bio_s,
                "ammonia_slack_kg": _var_value(payload["ammonia_slack_kg"], ammonia_node_count) * _amm_s,
                "water_slack_m3": _var_value(payload["water_slack_m3"], water_node_count) * _wat_s,
                # Which basin cap bound, and by how much. Zero-length when the official-quota
                # budget is off, so downstream readers see an array either way.
                "water_basin_slack_m3": (
                    _var_value(payload["water_basin_slack_m3"],
                               len(year_data["water_basin_codes"])) * _wat_s
                    if payload.get("water_basin_slack_m3") is not None else np.zeros(0)),
                "water_basin_codes": list(year_data.get("water_basin_codes") or []),
                "water_basin_use_m3": (
                    _var_value(payload["water_basin_use_m3"],
                               len(year_data["water_basin_codes"])) * _wat_s
                    if payload.get("water_basin_use_m3") is not None else np.zeros(0)),
                "injectivity_slack_mtpa": _var_value(payload["injectivity_slack_mtpa"], storage_count),
                "storage_slack_mt": _var_value(payload["storage_slack_mt"], storage_count),
                "edge_slack_mtpa": _var_value(payload["edge_slack_mtpa"], edge_count),
            },
            "year_data": year_data,
        }

    return {
        "status": status,
        "objective_cny": _model_obj_value(model) * _COST_SCALE,
        "solver_quality": solver_quality,
        "capex_pathway_indices": tuple(capex_pathway_indices),
        "year_solutions": year_solutions,
    }
