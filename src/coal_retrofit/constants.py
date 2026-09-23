from __future__ import annotations


TARGET_GEO_CRS = "EPSG:4326"
PLANT_YEAR_BASIS = "2025"
STATIC_LAYER_YEAR_BASIS = "static_source_layer"
NETWORK_EDGE_CLASS_MAIN = "existing_main_corridor"
NETWORK_EDGE_CLASS_TRIANGULATION = "triangulation_candidate"
NETWORK_COORD_DECIMALS = 6
NETWORK_TRIANGULATION_CRS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
NETWORK_TRIANGULATION_MAX_EDGE_KM = 500.0
NETWORK_TRIANGULATION_MAX_STRETCH = 1.2
NETWORK_EDGE_CLASS_DIRECT = "runtime_direct_fallback"
# Detour factor on straight-line candidate edges. Measured on this model's OWN corridor layer:
# the 62 existing oil and gas trunk segments run 24 338 km along 21 418 km of straight line, a
# length-weighted factor of 1.136 (median 1.036, p90 1.306). Pipeline techno-economics normally
# borrows 1.2-1.4 from international routes; China's built trunk network is straighter than that,
# so the country's own infrastructure is the better anchor and it is internally consistent with
# the corridor edges in the same table. Existing-corridor edges already carry their routed
# polyline length and are NOT scaled by this.
NETWORK_DETOUR_FACTOR = 1.136
# Every plant gets direct candidate arcs to its k nearest sinks, so the candidate network can
# never leave a source unable to reach storage. SimCCS guarantees the same invariant: every
# source and sink must be joined by at least one feasible corridor even when the cost surface is
# weighted to discourage crossings. Mirrors OptimizationAssumptions.top_k_storage_pairs, which
# declared this rule and was never read.
NETWORK_DIRECT_SINK_TOP_K = 5

BIOMASS_GJ_PER_TONNE = 15.0
# Purchase cost anchor: Wang & Cai 2024 (Nat Commun, SI Table 3) 10.3 USD/MWh × 7.0 / 3.6 ≈ 20.0.
# LOW/HIGH are the scenario spread around the base, not conversions.
BIOMASS_COST_BASE = 20.0
BIOMASS_COST_LOW = 18.0
BIOMASS_COST_HIGH = 26.0
BIOMASS_MATCH_BUFFER_KM = 150.0
BIOMASS_NODE_AGGREGATION_DEGREES = 0.25

NH3_H2_RATIO = 0.176
NH3_HB_POWER_KWH_PER_KG = 0.74
NH3_STORAGE_ADDER_USD_PER_KG = 0.017
NH3_TRANSPORT_ADDER_USD_PER_KG = 0.0
# Electrolysis routes admitted into the ammonia supply curve. China's installed electrolyser
# fleet is >90% alkaline, and the PEM layers in data/H2 carry costs 2-3x the AE layers in the
# near term (2025 production-weighted mean: AE 4.36 vs PEM 10.87 USD/kg H2). Including both
# also double-counted the same wind/solar resource. Restricting to AE keeps the curve on the
# route China actually builds; revisit when the joint wind-solar H2/NH3 dataset lands.
NH3_ELECTROLYSIS_TECHS = ("AE",)
# Haber-Bosch synthesis loop + air separation unit + balance of plant, per tonne of annual
# NH3 capacity. The electrolyser and its renewable supply are already inside LCOH, so only
# the synthesis island is added here: greenfield green-ammonia CAPEX is 1300-2000 USD/(t·yr)
# with the electrolyser at 40-50% of direct cost, leaving 650-1100 for the rest (midpoint).
NH3_HB_CAPEX_USD_PER_TONNE_YEAR = 875.0
NH3_HB_CAPEX_DISCOUNT_RATE = 0.08
NH3_HB_CAPEX_LIFETIME_YEARS = 20
AMMONIA_NODE_AGGREGATION_DEGREES = 0.5

# Water-grid coarsening, applied once when the inputs are built
# (`builders.water.write_water_inputs` -> `coarsen_water_inputs`), never at solve time.
WATER_COARSE_GRID_DEGREES = 2.0      # coarsen from 0.5° to 2.0°

WATER_MATCH_BUFFER_KM = 200.0  # 200km water supply radius
# BIBLIOGRAPHY KEY: richter2012. A presumptive environmental-flow standard: protecting 80%
# of daily flows maintains ecological integrity, so 20% is notionally extractable. Applied
# here as a single global factor to every basin, which is a simplification the Methods must
# state -- Richter's standard is a screening-level default, not a basin-specific allocation,
# and the Hai and the Yangtze would not carry the same value under a basin-specific study.
WATER_EXTRACTABLE_FRACTION = 0.20  # 20% of natural discharge extractable for industrial cooling (80% env flow)

# Solver unit scaling: reduce coefficient range for numerical stability
# Flow variables use scaled units; costs are adjusted accordingly
BIOMASS_FLOW_SCALE = 1e6   # solver flow vars in TJ instead of GJ (÷1e6)
AMMONIA_FLOW_SCALE = 1e6   # solver flow vars in kt instead of kg (÷1e6)
WATER_FLOW_SCALE = 1e6     # solver flow vars in Mm³ instead of m³ (÷1e6)

# BIBLIOGRAPHY KEY: ndrc2015no9.
COOLING_WATER_INTENSITY_M3_PER_MWH = {
    "once-through": 0.35,   # consumption basis, NDRC 2015 No.9: 0.29-0.41
    "recirculating": 1.85,
    "air": 0.37,
}

# --- Water withdrawal and consumption by combustion technology and cooling system --------
# BIBLIOGRAPHY KEYS: wang2023waterintensity (primary), macknick2011 (cross-check).
# These were cited here and in a figure caption but were missing from references.bib until
# v5, which is a poor state for the single most load-bearing input table in the study: every
# water number in every figure is built from it.
#
# Wang F., Wang P., Xu M. (2023) Water 15:1167, Table 1 — China-specific, and the only
# source found that gives withdrawal AND consumption, with AND without capture, split by
# both combustion technology and cooling system. Units m3/MWh.
#
# Cross-checked against Macknick et al. (2011) NREL/TP-6A20-50900 for the US fleet
# (reference/water/macknick_osti1009674.pdf, 1 m3 = 264.172 gal):
#     tower subcritical      withdrawal 2.01 (US) vs 2.31 (CN);  consumption 1.78 vs 2.01
#     tower supercritical+CCS           4.25      vs 4.14;                   3.20 vs 3.06
#     once-through subcritical        102.54      vs 116.48
# Two independent sources, different countries, agreeing to 5-14% on withdrawal.
#
# Note how far apart the two bases are for once-through: ~100 m3/MWh withdrawn against
# ~1 m3/MWh consumed, because the condenser flow is returned to the river. China's own
# 取水定额 excludes that returned flow entirely, which is why the quota for once-through
# (0.19-0.72 m3/MWh) is the LOWEST of the three cooling systems rather than the highest.
WATER_INTENSITY_BY_TECH_M3_PER_MWH: dict[tuple[str, str], dict[str, float]] = {
    # (combustion class, cooling class): withdrawal / consumption, base and with capture
    ("subcritical", "once-through"):   {"withdrawal": 116.48, "consumption": 1.240,
                                        "withdrawal_ccs": 199.11, "consumption_ccs": 1.770},
    ("supercritical", "once-through"): {"withdrawal": 88.90, "consumption": 0.690,
                                        "withdrawal_ccs": 161.49, "consumption_ccs": 0.850},
    ("ultra-supercritical", "once-through"): {"withdrawal": 82.80, "consumption": 0.228,
                                              "withdrawal_ccs": 143.20, "consumption_ccs": 0.344},
    ("subcritical", "recirculating"):   {"withdrawal": 2.31, "consumption": 2.01,
                                         "withdrawal_ccs": 4.51, "consumption_ccs": 3.65},
    ("supercritical", "recirculating"): {"withdrawal": 2.19, "consumption": 1.61,
                                         "withdrawal_ccs": 4.14, "consumption_ccs": 3.06},
    ("ultra-supercritical", "recirculating"): {"withdrawal": 1.58, "consumption": 1.26,
                                               "withdrawal_ccs": 3.44, "consumption_ccs": 2.53},
    ("subcritical", "air"):   {"withdrawal": 0.23, "consumption": 0.20,
                               "withdrawal_ccs": 0.45, "consumption_ccs": 0.36},
    ("supercritical", "air"): {"withdrawal": 0.21, "consumption": 0.16,
                               "withdrawal_ccs": 0.41, "consumption_ccs": 0.31},
    ("ultra-supercritical", "air"): {"withdrawal": 0.15, "consumption": 0.12,
                                     "withdrawal_ccs": 0.34, "consumption_ccs": 0.25},
}

# --- Chinese abstraction quota: the basis on which water is actually charged ------------
# BIBLIOGRAPHY KEY: mwr_quota_2019.
# 《水利部关于印发钢铁等十八项工业用水定额的通知》, 燃煤发电 单位发电量取水量 基准值,
# m3/MWh, by cooling system and unit size. Reproduced in 武汉产业能效指南 (2025) pp.88-89.
#
# This is a THIRD basis, distinct from the two in WATER_INTENSITY_BY_TECH_M3_PER_MWH:
#   consumption  what the basin actually loses           -> the availability constraint
#   withdrawal   everything diverted, incl. returned flow -> reported, never constrained
#   quota        what China meters and charges            -> the water tariff
# The quota deliberately excludes once-through condenser flow, which is why its once-through
# values (0.35-0.72) are the LOWEST of the three cooling systems rather than ~100x the highest.
CHINA_WATER_QUOTA_M3_PER_MWH: dict[tuple[str, str], float] = {
    ("recirculating", "<300"):    3.20,
    ("recirculating", "300"):     2.70,
    ("recirculating", "600"):     2.35,
    ("recirculating", ">=1000"):  2.00,
    ("once-through", "<300"):     0.72,
    ("once-through", "300"):      0.49,
    ("once-through", "600"):      0.42,
    ("once-through", ">=1000"):   0.35,
    ("air", "<300"):              0.80,
    ("air", "300"):               0.57,
    ("air", "600"):               0.49,
    ("air", ">=1000"):            0.42,
}


def quota_capacity_band(capacity_mw: float) -> str:
    """Map a unit's nameplate capacity to the quota table's size band."""
    if capacity_mw >= 1000.0:
        return ">=1000"
    if capacity_mw >= 600.0:
        return "600"
    if capacity_mw >= 300.0:
        return "300"
    return "<300"


# --- Official water resources by level-1 water-resource region, 10^8 m3/yr ---------------
# Third National Water Resources Survey and Evaluation, multi-year mean 1956-2016, via
# Wang G. et al. (2025) Advances in Water Science 36(6) Table 1
# (reference/water/wangGQ2025_adv_water_sci_36_948.pdf). Internally consistent with that
# paper's aggregates: north six 5221, south four 23078, national 28299.
#
# Used to bias-correct modelled runoff basin by basin: global hydrological models are
# calibrated against basin discharge, and at 0.5 deg they carry large regional biases even
# when the national total looks right (WaterGAP2-2e reproduces China's total to 1.2% while
# overestimating the Hai basin by 2.43x). Correction keeps each member's rate of change and
# replaces only the absolute level.
#
# Code A merges Songhua and Liao because data/ChinaBasins/basin_l1.gpkg does.
OFFICIAL_BASIN_WATER_1E8_M3: dict[str, float] = {
    "A": 1469.2 + 483.4,   # 东北诸河区 (松花江 + 辽河)
    "C": 327.6,            # 海河区
    "D": 702.8,            # 黄河区
    "E": 928.3,            # 淮河区
    "F": 9871.2,           # 长江区
    "G": 2694.5,           # 东南诸河区
    "H": 4758.6,           # 珠江区
    "J": 5753.8,           # 西南诸河区
    "K": 1310.1,           # 西北诸河区
}
# Window used to estimate model bias against the official baseline. The historical ISIMIP3b
# runs stop at 2014; the official series ends 2016.
BIAS_BASELINE_WINDOW = (1956, 2014)

# GEM combustion labels -> the three classes in the table above. CFB units are subcritical
# in steam conditions; IGCC (0.2 GW nationally) has no row in the source and is mapped to
# supercritical. `.../CCS` suffixes describe the retrofit state, not the steam cycle.
COMBUSTION_CLASS_MAP = {
    "subcritical": "subcritical",
    "supercritical": "supercritical",
    "ultra-supercritical": "ultra-supercritical",
    "cfb": "subcritical",
    "igcc": "supercritical",
}

PLANNING_YEARS = [2030, 2040, 2050, 2060]
