from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None

from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import PATHWAY_INDEX

# Import flow scaling factors for numerical stability
from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE, BIOMASS_FLOW_SCALE


def _add_vector_equality(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] == rhs[idx] for idx in range(length)), name=name)


def _add_vector_upper_bound(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] <= rhs[idx] for idx in range(length)), name=name)


def _add_mccormick_product(model, share_var, binary_var, name: str):
    """Linearize z = share_var * binary_var where binary in {0,1}, share in [0,1]."""
    z = model.addVar(lb=0.0, ub=1.0, name=name)
    model.addConstr(z <= share_var,                      name=f"{name}_u1")
    model.addConstr(z <= binary_var,                     name=f"{name}_u2")
    model.addConstr(z >= share_var - (1.0 - binary_var), name=f"{name}_lo")
    return z


def _build_air_retrofit_capex(
    model,
    year_data: dict[str, object],
    payload: dict[str, object],
    plant_count: int,
    yr_sfx: str,
    prev_payload: dict[str, object] | None = None,
):
    """One-time capex for converting wet condensers to dry cooling.

    Charged on the increment of the INSTALLED STOCK, not of the period's converted share:
    like the CCS retrofit, an air-cooled condenser once built stays built, so a hub whose
    dry-cooled share dips in one period and recovers in the next must not pay twice. The
    stock is only bounded below (>= this period's converted share, >= last period's stock),
    which is enough because the objective is minimising and the coefficient positive.
    """
    if not bool(year_data.get("allow_air_cooling_retrofit", False)):
        return 0.0
    capex_per_plant = year_data.get("air_retrofit_capex_per_plant")
    if capex_per_plant is None:
        return 0.0
    installed = payload["air_installed"]
    terms = []
    for plant_idx in range(plant_count):
        coeff = float(capex_per_plant[plant_idx])
        if coeff <= 0:
            continue
        if prev_payload is None:
            terms.append(coeff * installed[plant_idx])
        else:
            previous = prev_payload["air_installed"][plant_idx]
            model.addConstr(
                installed[plant_idx] >= previous, name=f"air_installed_mono_{plant_idx}_{yr_sfx}"
            )
            terms.append(coeff * (installed[plant_idx] - previous))
    return gp.quicksum(terms) if terms else 0.0


def _build_blend_upgrade_capex(
    model,
    capacity_mw,
    blend_level_b,
    blend_level_a,
    assumptions: "OptimizationAssumptions",
    plant_count: int,
    sfx: str = "",
    prev_blend_level_b=None,
    prev_blend_level_a=None,
):
    """Blend upgrade capex expression. If prev_* is None, assumes starting from level 0."""
    cap = np.asarray(capacity_mw, dtype=np.float64)
    coeff_b = cap * assumptions.biomass_upgrade_capex_cny_per_mw_per_level
    coeff_a = cap * assumptions.ammonia_upgrade_capex_cny_per_mw_per_level
    if prev_blend_level_b is None:
        return coeff_b @ blend_level_b + coeff_a @ blend_level_a
    # Use max(0, delta) via auxiliary variables to prevent negative CAPEX
    # when retired plants reset blend level to 0
    delta_b_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_b_pos{sfx}")
    delta_a_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_a_pos{sfx}")
    model.addConstr(delta_b_pos >= blend_level_b - prev_blend_level_b, name=f"blend_delta_b_lb{sfx}")
    model.addConstr(delta_a_pos >= blend_level_a - prev_blend_level_a, name=f"blend_delta_a_lb{sfx}")
    return coeff_b @ delta_b_pos + coeff_a @ delta_a_pos


def _add_blend_level_constraints(
    model,
    share,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_data: dict[str, object],
    sfx: str,
) -> tuple:
    """Add per-plant blend-level binary variables, linearization zetas, and usage constraints.

    Returns
    -------
    select_b : MVar (plant_count, L_b+1)  — level 0 = no biomass
    select_a : MVar (plant_count, L_a+1)  — level 0 = no ammonia
    blend_level_b : MVar (plant_count,)   — numeric level index (Σ l·select_b)
    blend_level_a : MVar (plant_count,)
    biomass_use_gj : MVar (plant_count,)
    ammonia_use_kg : MVar (plant_count,)
    bio_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_b[l] · zeta_bio[p,l]  per plant
    beccs_blend_red_exprs : list[LinExpr]  — E_retrofit · Σ β_b[l]·zeta_beccs[p,l] per plant
        (blend-displacement part only; the η·share capture part is added by the caller)
    amm_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_a[l] · zeta_amm[p,l] per plant
    bio_penalty_exprs : list[LinExpr] — blend-level-dependent energy penalty cost per plant
    bio_penalty_emissions_exprs : list[LinExpr] — extra vented CO2 from the penalty fuel (Mt)
    """
    blend_b = np.asarray(scenario.biomass_blend_levels, dtype=np.float64)
    blend_a = np.asarray(scenario.ammonia_blend_levels, dtype=np.float64)
    L_b = len(blend_b)
    L_a = len(blend_a)
    lhv = float(assumptions.nh3_lhv_gj_per_kg)

    # select_b/select_a: column 0 = "no blend", column l+1 = "blend level l"
    select_b = model.addMVar((plant_count, L_b + 1), vtype=GRB.BINARY, name=f"sel_b{sfx}")
    select_a = model.addMVar((plant_count, L_a + 1), vtype=GRB.BINARY, name=f"sel_a{sfx}")
    blend_level_b = model.addMVar(plant_count, lb=0.0, ub=float(L_b), name=f"blv_b{sfx}")
    blend_level_a = model.addMVar(plant_count, lb=0.0, ub=float(L_a), name=f"blv_a{sfx}")
    biomass_use_gj = model.addMVar(plant_count, lb=0.0, name=f"biomass_use_gj{sfx}")
    ammonia_use_kg = model.addMVar(plant_count, lb=0.0, name=f"ammonia_use_kg{sfx}")

    bio_red_exprs: list[object] = []
    beccs_blend_red_exprs: list[object] = []
    amm_red_exprs: list[object] = []
    bio_penalty_exprs: list[object] = []  # blend-level-dependent energy penalty per plant
    bio_penalty_emissions_exprs: list[object] = []  # extra CO2 from the penalty fuel (Mt)

    bio_pen_coeff_data = year_data.get("biomass_penalty_coeff_per_level", 0.0)
    bio_pen_em_coeff_data = year_data.get("biomass_penalty_emissions_coeff_per_level", 0.0)
    # On BECCS the co-firing penalty fuel burns in the same boiler as the captured flue gas,
    # so only the uncaptured share is vented (Fan et al. 2023 SI eq. S42) -- and the captured
    # share is a real tonne that has to be piped and stored.
    beccs_pen_em_coeff_data = year_data.get(
        "beccs_penalty_emissions_coeff_per_level", bio_pen_em_coeff_data
    )
    beccs_pen_cap_coeff_data = year_data.get("beccs_penalty_captured_coeff_per_level", 0.0)
    beccs_penalty_captured_exprs: list[object] = []

    for p in range(plant_count):
        # Retrofit operations carry the efficiency ratio (rebuilt plants) and the
        # retrofit CF boost; fuel use additionally scales with the plant heat rate.
        E_rt = float(year_data["emissions_retrofit_mt"][p])
        hr_p = float(year_data["heat_rate_eff"][p])
        G_bp = year_data["generation_by_pathway"][p, :]
        G_bio = float(G_bp[PATHWAY_INDEX["biomass"]])
        G_beccs = float(G_bp[PATHWAY_INDEX["beccs"]])
        G_amm = float(G_bp[PATHWAY_INDEX["ammonia"]])
        bio_pen_coeff = float(bio_pen_coeff_data[p]) if hasattr(bio_pen_coeff_data, '__getitem__') and not isinstance(bio_pen_coeff_data, (int, float)) else float(bio_pen_coeff_data)
        bio_pen_em_coeff = float(bio_pen_em_coeff_data[p]) if hasattr(bio_pen_em_coeff_data, '__getitem__') and not isinstance(bio_pen_em_coeff_data, (int, float)) else float(bio_pen_em_coeff_data)
        beccs_pen_em_coeff = float(beccs_pen_em_coeff_data[p]) if hasattr(beccs_pen_em_coeff_data, '__getitem__') and not isinstance(beccs_pen_em_coeff_data, (int, float)) else float(beccs_pen_em_coeff_data)
        beccs_pen_cap_coeff = float(beccs_pen_cap_coeff_data[p]) if hasattr(beccs_pen_cap_coeff_data, '__getitem__') and not isinstance(beccs_pen_cap_coeff_data, (int, float)) else float(beccs_pen_cap_coeff_data)
        s_bio = share[p, PATHWAY_INDEX["biomass"]]
        s_beccs = share[p, PATHWAY_INDEX["beccs"]]
        s_amm = share[p, PATHWAY_INDEX["ammonia"]]

        # --- Exactly one blend level selected per pathway type ---
        model.addConstr(select_b[p, :].sum() == 1.0, name=f"sel_b_{p}{sfx}")
        model.addConstr(select_a[p, :].sum() == 1.0, name=f"sel_a_{p}{sfx}")

        # --- If level 0 (no blend), the relevant shares must be 0 ---
        model.addConstr(s_bio + s_beccs <= 1.0 - select_b[p, 0], name=f"nb_b_{p}{sfx}")
        model.addConstr(s_amm <= 1.0 - select_a[p, 0], name=f"nb_a_{p}{sfx}")

        # --- Numeric blend level (for upgrade cost computation) ---
        model.addConstr(
            blend_level_b[p] == gp.quicksum(l * select_b[p, l] for l in range(L_b + 1)),
            name=f"blv_b_{p}{sfx}",
        )
        model.addConstr(
            blend_level_a[p] == gp.quicksum(l * select_a[p, l] for l in range(L_a + 1)),
            name=f"blv_a_{p}{sfx}",
        )

        # --- Linearize: zeta = select_b[p, l+1] × share (binary × continuous in [0,1]) ---
        bio_use_expr = gp.LinExpr()
        bio_red = gp.LinExpr()
        beccs_blend_red = gp.LinExpr()
        bio_penalty = gp.LinExpr()  # blend-level-dependent energy penalty (cost)
        bio_penalty_emissions = gp.LinExpr()  # same penalty fuel, as vented CO2 (Mt)
        beccs_penalty_captured = gp.LinExpr()  # the captured share of the BECCS penalty fuel (Mt)

        for l, beta_b in enumerate(blend_b):
            bin_b = select_b[p, l + 1]

            z_bio = _add_mccormick_product(model, s_bio, bin_b, f"zb_{p}_{l}{sfx}")

            z_beccs = _add_mccormick_product(model, s_beccs, bin_b, f"zbc_{p}_{l}{sfx}")

            bio_use_expr += hr_p * beta_b * (G_bio * z_bio + G_beccs * z_beccs) / BIOMASS_FLOW_SCALE
            bio_red  += E_rt * beta_b * z_bio
            beccs_blend_red += E_rt * beta_b * z_beccs
            # Energy penalty: β_b × coeff × (G_bio·z_bio + G_beccs·z_beccs)
            bio_penalty += beta_b * bio_pen_coeff * (G_bio * z_bio + G_beccs * z_beccs)
            bio_penalty_emissions += beta_b * (
                bio_pen_em_coeff * G_bio * z_bio + beccs_pen_em_coeff * G_beccs * z_beccs
            )
            beccs_penalty_captured += beta_b * beccs_pen_cap_coeff * G_beccs * z_beccs

        model.addConstr(biomass_use_gj[p] == bio_use_expr, name=f"bu_{p}{sfx}")

        amm_use_expr = gp.LinExpr()
        amm_red = gp.LinExpr()

        for l, beta_a in enumerate(blend_a):
            bin_a = select_a[p, l + 1]

            z_amm = _add_mccormick_product(model, s_amm, bin_a, f"za_{p}_{l}{sfx}")

            amm_use_expr += G_amm * hr_p / lhv * beta_a * z_amm / AMMONIA_FLOW_SCALE
            amm_red  += E_rt * beta_a * z_amm

        model.addConstr(ammonia_use_kg[p] == amm_use_expr, name=f"au_{p}{sfx}")

        bio_red_exprs.append(bio_red)
        beccs_blend_red_exprs.append(beccs_blend_red)
        amm_red_exprs.append(amm_red)
        bio_penalty_exprs.append(bio_penalty)
        bio_penalty_emissions_exprs.append(bio_penalty_emissions)
        beccs_penalty_captured_exprs.append(beccs_penalty_captured)

    return (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs,
    )


def _add_plant_path_constraints(
    model,
    share,
    year_data: dict[str, object],
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_suffix: str = "",
):
    pathway_count = len(PATHWAYS)
    sfx = f"_{year_suffix}" if year_suffix else ""
    captured_mt_by_plant = model.addMVar(plant_count, lb=0.0, name=f"captured_mt_by_plant{sfx}")
    water_use_m3 = model.addMVar(plant_count, lb=0.0, name=f"water_use_m3{sfx}")

    # Wet-to-dry cooling conversion. `air_share[p, k]` is the part of hub p's generation that
    # is both on pathway k and dry-cooled, so 0 <= air_share <= share and the hub's converted
    # fraction is sum_k air_share[p, k]. Writing it per pathway keeps every term linear: the
    # alternative, a single air fraction multiplying the pathway shares, is bilinear and
    # would need either binaries or a McCormick envelope. It also represents a real degree of
    # freedom, since a hub aggregates ~10 units and the operator chooses which of them get
    # both the capture island and the air-cooled condenser.
    allow_air = bool(year_data.get("allow_air_cooling_retrofit", False)) and year_data.get(
        "air_water_intensity"
    ) is not None
    air_share = model.addMVar((plant_count, pathway_count), lb=0.0, ub=1.0, name=f"air_share{sfx}")
    # Installed stock, so capex is charged on the high-water mark and not re-charged when a
    # converted hub's dry-cooled share dips in one period and recovers in the next. Same
    # device as `retrofit_installed` for CCS; without it, an air-cooled condenser could be
    # paid for twice.
    air_installed = model.addMVar(plant_count, lb=0.0, ub=1.0, name=f"air_installed{sfx}")
    if not allow_air:
        model.addConstr(air_share == 0.0, name=f"air_share_off{sfx}")
        model.addConstr(air_installed == 0.0, name=f"air_installed_off{sfx}")
    else:
        model.addConstrs(
            (
                air_installed[plant_idx] >= gp.quicksum(
                    air_share[plant_idx, path_idx] for path_idx in range(pathway_count)
                )
                for plant_idx in range(plant_count)
            ),
            name=f"air_installed_ge_share{sfx}",
        )

    (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs,
    ) = _add_blend_level_constraints(model, share, plant_count, scenario, assumptions, year_data, sfx)

    eta = float(scenario.capture_rate)
    plant_reduction_exprs: list[object] = []
    ccs_penalty_captured = year_data.get("ccs_penalty_captured_matrix")
    air_penalty_captured = year_data.get("air_penalty_captured_matrix")

    for plant_idx in range(plant_count):
        E_p = float(year_data["emissions_mt"][plant_idx])               # baseline
        E_op = float(year_data["emissions_operating_mt"][plant_idx])    # efficiency-adjusted
        E_rt = float(year_data["emissions_retrofit_mt"][plant_idx])     # efficiency × CF boost
        G_bp = year_data["generation_by_pathway"][plant_idx, :]
        s_un = share[plant_idx, PATHWAY_INDEX["unabated"]]
        s_ccs = share[plant_idx, PATHWAY_INDEX["ccs"]]
        s_bio = share[plant_idx, PATHWAY_INDEX["biomass"]]
        s_beccs = share[plant_idx, PATHWAY_INDEX["beccs"]]
        s_amm = share[plant_idx, PATHWAY_INDEX["ammonia"]]

        # Water use (scaled by WATER_FLOW_SCALE for numerical stability); per-pathway
        # generation (retrofit boost, retire = 0) times pathway water intensity, with the
        # dry-cooled part of each pathway priced at the air-cooled intensity instead.
        air_intensity = year_data["air_water_intensity"] if allow_air else None
        water_expr = gp.quicksum(
            float(G_bp[path_idx])
            * (
                float(year_data["water_intensity"][plant_idx, path_idx]) * share[plant_idx, path_idx]
                - (
                    float(
                        year_data["water_intensity"][plant_idx, path_idx]
                        - air_intensity[plant_idx, path_idx]
                    )
                    * air_share[plant_idx, path_idx]
                    if air_intensity is not None
                    else 0.0
                )
            )
            / WATER_FLOW_SCALE
            for path_idx in range(pathway_count)
        )
        model.addConstr(water_use_m3[plant_idx] == water_expr, name=f"water_use_{plant_idx}{sfx}")
        if allow_air:
            for path_idx in range(pathway_count):
                model.addConstr(
                    air_share[plant_idx, path_idx] <= share[plant_idx, path_idx],
                    name=f"air_share_le_share_{plant_idx}_{path_idx}{sfx}",
                )

        # Physical captured CO2 differs from BECCS net reduction. Beyond eta x the boiler's
        # own flue gas, the capture train also takes eta of every penalty fuel burnt in the
        # same boiler (CCS energy penalty, BECCS co-firing penalty, dry-cooling backpressure);
        # those tonnes were vented at (1-eta) in the residual but never counted here, so they
        # were abated on paper without being transported or stored.
        captured_expr = E_rt * eta * (s_ccs + s_beccs) + beccs_penalty_captured_exprs[plant_idx]
        if ccs_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(ccs_penalty_captured[plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(ccs_penalty_captured[plant_idx, path_idx]) != 0.0
            )
        if allow_air and air_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(air_penalty_captured[plant_idx, path_idx]) * air_share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(air_penalty_captured[plant_idx, path_idx]) != 0.0
            )
        model.addConstr(captured_mt_by_plant[plant_idx] == captured_expr, name=f"captured_balance_{plant_idx}{sfx}")

        # Actual (residual) emissions under each pathway, then reduction vs baseline.
        # With unit CF boost and efficiency ratio this reduces exactly to the classic
        # per-pathway reduction fractions times baseline emissions.
        # The efficiency-penalty fuel (CCS fixed + blend-level-dependent) is burned and
        # vented like any other coal, so its emissions are part of the residual.
        residual_expr = (
            E_op * s_un
            + E_rt * (1.0 - eta) * s_ccs
            + (E_rt * s_bio - bio_red_exprs[plant_idx])
            + (E_rt * (1.0 - eta) * s_beccs - beccs_blend_red_exprs[plant_idx])
            + (E_rt * s_amm - amm_red_exprs[plant_idx])
            + gp.quicksum(
                float(year_data["ccs_penalty_emissions_matrix"][plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
            )
            + bio_penalty_emissions_exprs[plant_idx]
            # Backpressure penalty of dry cooling: burnt and vented like any other coal.
            + (
                gp.quicksum(
                    float(year_data["air_penalty_emissions_matrix"][plant_idx, path_idx])
                    * air_share[plant_idx, path_idx]
                    for path_idx in range(pathway_count)
                )
                if allow_air
                else 0.0
            )
        )
        plant_red = E_p - residual_expr
        plant_reduction_exprs.append(plant_red)

    total_reduction_mt = gp.quicksum(plant_reduction_exprs)
    total_bio_penalty = gp.quicksum(bio_penalty_exprs)
    return (
        captured_mt_by_plant, biomass_use_gj, ammonia_use_kg, water_use_m3, total_reduction_mt,
        select_b, select_a, blend_level_b, blend_level_a,
        total_bio_penalty, plant_reduction_exprs, air_share, air_installed,
    )


def _add_forced_pathway_activation_constraints(
    model,
    share,
    scenario: OptimizationScenario,
    year_data: dict[str, object],
    hub_count: int,
    name_suffix: str = "",
    retired_mask: np.ndarray | None = None,
) -> None:
    min_share = float(scenario.min_forced_path_share)
    if min_share <= 0.0 or not scenario.forced_pathways:
        return
    generation = np.asarray(year_data["generation"], dtype=np.float64)
    if retired_mask is not None:
        active_gen = generation * (~retired_mask).astype(float)
    else:
        active_gen = generation
    total_active_generation = float(active_gen.sum())
    if total_active_generation <= 0.0:
        return
    forced_pathways = []
    seen: set[str] = set()
    for pathway in scenario.forced_pathways:
        pathway_name = str(pathway).strip().lower()
        if pathway_name in PATHWAY_INDEX and pathway_name not in seen and scenario.path_enabled(pathway_name):
            forced_pathways.append(pathway_name)
            seen.add(pathway_name)
    for pathway_name in forced_pathways:
        pathway_idx = PATHWAY_INDEX[pathway_name]
        pathway_generation = gp.quicksum(
            float(active_gen[hub_idx]) * share[hub_idx, pathway_idx]
            for hub_idx in range(hub_count)
            if active_gen[hub_idx] > 0.0
        )
        model.addConstr(
            pathway_generation >= min_share * total_active_generation,
            name=f"force_{pathway_name}{name_suffix}",
        )
