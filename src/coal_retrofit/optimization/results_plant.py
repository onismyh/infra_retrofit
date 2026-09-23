"""煤电厂侧结果表：路径份额（逐厂 x 路径）、省级汇总、逐厂明细、逐厂成本分解。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs
from .emissions import blend_level_to_ratio, reduction_fraction
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from .year_types import YearData


def _build_pathway_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    captured_mt_by_plant: np.ndarray | None = None,
    blend_level_b: np.ndarray | None = None,
    blend_level_a: np.ndarray | None = None,
    year_data: YearData | None = None,
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
        np.asarray(year_data.generation, dtype=np.float64) if year_data is not None
        else prepared.plants["annual_generation_mwh"].astype(float).to_numpy()
    )
    em_year = (
        np.asarray(year_data.emissions_mt, dtype=np.float64) if year_data is not None
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
    year_data: YearData | None = None,
    plant_reduction_mt: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-plant high-resolution detail: pathway shares, resource use, blend levels, storage proximity."""
    plants = prepared.plants
    gen_year = (
        np.asarray(year_data.generation, dtype=np.float64) if year_data is not None
        else plants["annual_generation_mwh"].astype(float).to_numpy()
    )
    em_year = (
        np.asarray(year_data.emissions_mt, dtype=np.float64) if year_data is not None
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


def _build_plant_cost_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    year_data: YearData,
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
    emissions = np.asarray(year_data.emissions_mt, dtype=np.float64)
    capacity_mw = plants["total_capacity_mw"].astype(float).to_numpy()
    carbon_price = float(year_data.carbon_price)
    retire_idx = PATHWAY_INDEX["retire"]
    # Capture-island / BECCS-increment stock coefficients (see solver); fall back to the
    # per-pathway matrix for results written before the two-stock form existed.
    stock_coeff = year_data.retrofit_stock_capex

    rows: list[dict[str, object]] = []
    for p in range(n):
        share = share_values[p]
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
            float(year_data.baseline_net_matrix[p, k]) * float(share[k])
            for k in range(len(PATHWAYS))
        )
        # Carbon cost: price × residual emissions (approximate reporting: retrofit
        # pathways use the efficiency × CF-boost emission basis, retire avoids baseline)
        e_rt = float(year_data.emissions_retrofit_mt[p])
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
        _bio_scale = float(year_data.biomass_flow_scale)
        _cspg = year_data.coal_savings_per_gj
        _cspg_val = float(_cspg[p]) if hasattr(_cspg, '__getitem__') and not isinstance(_cspg, (int, float)) else float(_cspg)
        coal_savings = (_cspg_val / _bio_scale) * float(biomass_use_gj[p])
        # Incremental O&M
        incr_om = sum(float(year_data.fixed_cost_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # Energy penalty
        energy_pen = sum(float(year_data.energy_penalty_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # CCS O&M
        ccs_om = sum(float(year_data.ccs_om_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # Stranded asset (model-consistent: charged on newly retired share increment)
        prev_retire = float(prev_share_values[p, retire_idx]) if prev_share_values is not None else 0.0
        stranded = float(year_data.stranded_per_plant[p]) * max(0.0, float(share[retire_idx]) - prev_retire)
        # CCS retrofit CAPEX (model-consistent: charged on installed-stock increments,
        # i.e. the max historical share; coeff already includes the learning factor)
        ccs_capex = 0.0
        for j, k in enumerate(capex_pathway_indices):
            coeff = (
                float(stock_coeff[p, j]) if stock_coeff is not None and j < np.asarray(stock_coeff).shape[1]
                else float(year_data.ccs_retrofit_capex_matrix[p, k])
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
