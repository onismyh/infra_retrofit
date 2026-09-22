"""Industrial point-source parameters: water intake quotas, hydrogen routes, asset lives.

Every number here carries its source. Two rules inherited from `constants.py`:

1. **Water basis.** The basin constraint acts on CONSUMPTION, not withdrawal
   (`optimization/data_prep.py`). For industry the two are much closer than for
   once-through power cooling: Chinese industrial plants run closed water circuits with
   >95% recycling, so the GB "unit-product water intake" (单位产品取水量) is make-up water
   replacing evaporative loss, i.e. very nearly consumptive. This is an ASSUMPTION and the
   Methods must state it; `INDUSTRY_CONSUMPTION_SHARE` is the knob that carries it.
2. **China-matched sourcing.** All quotas come from the GB/T 18916《取水定额》series — the
   same family as the coal-power quota already in `CHINA_WATER_QUOTA_M3_PER_MWH`
   (GB/T 18916.1, via 水利部《钢铁等十八项工业用水定额》水节约〔2019〕373号). Mixing a
   Chinese power quota with, say, a European industrial intensity would break the
   like-for-like comparison the water constraint depends on.

Quota levels: the GB standards give 通用值 (existing plants, day-to-day management) and
先进值 (new/rebuilt plants, used for water-permit approval). Stock plants therefore take
通用值; a plant that retrofits takes 先进值.
"""
from __future__ import annotations

from typing import Final

# --- Sector keys -------------------------------------------------------------------------
SECTOR_STEEL_BF: Final = "steel_bf_bof"
SECTOR_STEEL_EAF: Final = "steel_eaf"
SECTOR_CEMENT: Final = "cement"
SECTOR_AMMONIA: Final = "ammonia"
SECTOR_METHANOL: Final = "methanol"
SECTOR_REFINERY: Final = "refinery"
SECTOR_COAL_CHEM: Final = "coal_chemical"

# Refinery and modern coal chemicals are OUT OF SCOPE for now (author's call, 2026-09-06).
# The decisive reason is data: neither GB/T 18916.3-2022 (石油炼制) nor the coal-chemical
# quota tables were obtainable, so their water intensity could only have been guessed — and
# water is this study's binding constraint. Their loaders and keys are kept so that supplying
# the two quotas is all it takes to switch them back on.
INDUSTRY_SECTORS: Final[tuple[str, ...]] = (
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
    SECTOR_CEMENT,
    SECTOR_AMMONIA,
    SECTOR_METHANOL,
)

SECTORS_OUT_OF_SCOPE: Final[frozenset[str]] = frozenset({SECTOR_REFINERY, SECTOR_COAL_CHEM})

# Sector -> target group. The sector caps (scripts/build_sector_targets.py) are read off China
# TIMES at the level TIMES books CO2: iron & steel, building materials, chemicals. Coal power
# is the fourth group, "power", and is not in this map because it is not an industrial sector.
SECTOR_TARGET_GROUP: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "steel",
    SECTOR_STEEL_EAF: "steel",
    SECTOR_CEMENT: "cement",
    SECTOR_AMMONIA: "chemicals",
    SECTOR_METHANOL: "chemicals",
}
POWER_TARGET_GROUP: Final[str] = "power"

SECTOR_LABELS_ZH: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "钢铁（高炉-转炉）",
    SECTOR_STEEL_EAF: "钢铁（电炉）",
    SECTOR_CEMENT: "水泥",
    SECTOR_AMMONIA: "合成氨",
    SECTOR_METHANOL: "甲醇",
    SECTOR_REFINERY: "炼化",
    SECTOR_COAL_CHEM: "现代煤化工",
}


# --- Water intake quota, m3 per tonne of product ------------------------------------------
# BIBLIOGRAPHY KEY: gb18916_series.
#
# (通用值, 先进值) per (sector, feedstock/route). `None` means NOT YET SOURCED — the builder
# raises rather than substituting a guess, because a fabricated water intensity would flow
# straight into the binding constraint of this study.
#
# 钢铁 GB/T 18916.2-2022《取水定额 第2部分：钢铁联合企业》, m3/t 粗钢:
#     含焦化+含冷轧 4.8/3.1 ; 含焦化+不含冷轧 4.5/2.4
#     不含焦化+含冷轧 4.2/2.2 ; 不含焦化+不含冷轧 3.6/2.1
#   Chinese integrated mills overwhelmingly carry both coking and cold rolling, so the
#   "含焦化+含冷轧" row is the stock default.
# 电炉 has no whole-plant row in GB/T 18916.2; it is built from that standard's PROCESS rows
#   (电炉炼钢 1.74/0.55 + 棒材轧钢 0.70/0.34) because a short-process mill is exactly those
#   two steps. Composed, not quoted — flagged as such.
WATER_INTAKE_QUOTA_M3_PER_T: Final[dict[tuple[str, str], tuple[float, float] | None]] = {
    (SECTOR_STEEL_BF, "default"): (4.8, 3.1),
    (SECTOR_STEEL_EAF, "default"): (2.44, 0.89),   # 1.74+0.70 / 0.55+0.34, composed
    # 水泥 GB/T 18916.62-2022《取水定额 第62部分：水泥》, m3/t 熟料
    (SECTOR_CEMENT, "default"): (0.510, 0.225),
    # 合成氨 GB/T 18916.8《取水定额 第8部分：合成氨》, m3/t 氨, BY FEEDSTOCK — which lines up
    # one-to-one with the `Feedstock` column of the ammonia point-source table.
    (SECTOR_AMMONIA, "Coal"): (12.0, 8.0),          # 烟煤
    (SECTOR_AMMONIA, "Gas"): (9.0, 5.5),            # 天然气
    (SECTOR_AMMONIA, "Oil"): (12.0, 8.0),           # no oil row in GB; coal row used, 1 plant
    (SECTOR_AMMONIA, "Anthracite"): (11.0, 7.5),    # 无烟块煤（型煤）
    # Fallback for blank/unmapped feedstock: the coal route, which is 160 of 198 plants.
    (SECTOR_AMMONIA, "default"): (12.0, 8.0),
    # 甲醇 DB13/T 5448.7-2021《工业取水定额 第7部分：煤化工行业》(河北), m3/t 甲醇
    # Provincial standard, used because no national 甲醇 part of GB/T 18916 was located.
    (SECTOR_METHANOL, "Coal"): (15.0, 11.0),
    (SECTOR_METHANOL, "Coke oven gas"): (9.03, 6.34),
    (SECTOR_METHANOL, "Gas"): (9.03, 6.34),         # gas-based assigned the coke-oven-gas row
    # 矿热炉尾气 / 电石炉尾气 are by-product off-gas routes, the same class as coke oven gas.
    (SECTOR_METHANOL, "矿热炉尾气"): (9.03, 6.34),
    (SECTOR_METHANOL, "电石炉尾气"): (9.03, 6.34),
    (SECTOR_METHANOL, "default"): (15.0, 11.0),      # fallback: coal route, 148 of 221 plants
    # NOT SOURCED. GB/T 18916.3-2022《取水定额 第3部分：石油炼制》exists and is the right
    # standard, but its numeric table was not obtained; likewise the 煤制烯烃/煤制油/煤制乙二醇
    # rows (GB/T 18916 has a 第36部分：煤制乙二醇). Do not guess these.
    (SECTOR_REFINERY, "default"): None,
    (SECTOR_COAL_CHEM, "default"): None,
}

# Fraction of unit-product water intake that leaves the basin (evaporation, product moisture,
# sludge). 1.0 = intake is fully consumptive, the assumption argued in the module docstring.
# Exposed so `SA_industry_consumption_share` can test it.
INDUSTRY_CONSUMPTION_SHARE: Final[float] = 1.0


# --- Hydrogen routes ----------------------------------------------------------------------
# Which sectors can substitute green hydrogen at all, and for WHAT.
#
# 水泥 = False. Checked against the literature as asked. A hydrogen route physically exists —
# H2 can be co-fired in the kiln burner — but it can only displace FUEL combustion; the
# calcination of CaCO3 is a chemical decomposition that emits CO2 no matter what heats it.
# Reported ceilings: a 100% hydrogen-fuelled clinker process cuts CO2 intensity by 27.6%
# (Energy Conversion & Management 2023, "Decarbonisation pathways of the cement production
# process via hydrogen and oxy-combustion"), and >60% of cement-sector emissions are
# calcination. The point-source table's `H2_DMD` for cement (1 396 万吨 H2/yr nationally) is
# a fuel-heat-equivalent figure; treating it as an abatement basis would overstate cement's
# hydrogen-reachable abatement roughly two-fold. Cement is therefore CCS-only here, and this
# flag is the switch for a sensitivity run rather than a hard-coded exclusion.
SECTOR_HAS_H2_ROUTE: Final[dict[str, bool]] = {
    SECTOR_STEEL_BF: True,      # H2-DRI displaces the blast furnace's coke reduction
    SECTOR_STEEL_EAF: False,    # already scrap+electricity; no fossil reductant to displace
    SECTOR_CEMENT: False,       # see above
    SECTOR_AMMONIA: True,       # green H2 replaces grey H2 feedstock, stoichiometric
    SECTOR_METHANOL: True,      # green H2 adjusts the H/C ratio of coal-to-methanol
    SECTOR_REFINERY: True,      # green H2 replaces on-site SMR/by-product H2
    SECTOR_COAL_CHEM: True,     # green H2 replaces syngas-shift H2
}

# What the `H2_DMD` column of each point-source table MEANS. All are "hydrogen required if
# this plant's H2-substitutable demand were fully met by hydrogen", but the substitutable
# quantity differs by sector, so the blend level must be interpreted against the right base.
H2_DEMAND_BASIS: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "full_h2_dri",          # 0.081 t H2 / t crude steel
    SECTOR_CEMENT: "fuel_heat_equivalent",   # not used: no H2 route
    SECTOR_AMMONIA: "stoichiometric_feed",   # 0.178 t H2 / t NH3
    SECTOR_METHANOL: "stoichiometric_feed",
    SECTOR_REFINERY: "current_h2_use",       # ~1.5% of crude throughput by mass
    SECTOR_COAL_CHEM: "stoichiometric_feed",
}


# --- Asset life, for retirement ------------------------------------------------------------
# Two regimes, because the sectors differ in what is known about them:
#
#   PARTIALLY OBSERVED (cement 99.6%, steel BOF 55% / EAF 35%) -> gaps are filled from that
#     sector's OWN observed year distribution, by deterministic quantile draw. The observed
#     half is real information about the sector's age profile and a flat ladder would throw
#     it away, biasing the stock older or younger than it is.
#   NOT OBSERVED AT ALL (ammonia, methanol) -> ages spread evenly over [0, life], so a
#     constant 1/life of capacity reaches end of life each year. Maximum-entropy given no
#     age data; it is not a claim about the real age profile and must not be read as one.
#
# Both fills are deterministic, never sampled, so two runs are bit-identical.
ASSET_LIFETIME_YEARS: Final[dict[str, int]] = {
    SECTOR_STEEL_BF: 30,
    SECTOR_STEEL_EAF: 30,
    SECTOR_CEMENT: 35,
    SECTOR_AMMONIA: 30,
    SECTOR_METHANOL: 30,
    SECTOR_REFINERY: 40,
    SECTOR_COAL_CHEM: 30,
}

# Sectors whose retirement schedule is a uniform-age assumption rather than observed years.
SECTORS_WITH_UNIFORM_RETIREMENT: Final[frozenset[str]] = frozenset(
    {SECTOR_AMMONIA, SECTOR_METHANOL}
)


# --- Sector hub counts ----------------------------------------------------------------------
# Coal power clusters 3 541 units into 350 hubs (~10 units/hub). Industry is clustered PER
# SECTOR, so that a hub is never a mixture of, say, a cement kiln and an ammonia plant: their
# water intensities, hydrogen bases and capture costs have nothing in common and a blended
# hub would be uninterpretable. Counts keep roughly the coal-power units-per-hub ratio, with
# a floor so small sectors are not collapsed to a handful of points.
DEFAULT_SECTOR_HUB_COUNTS: Final[dict[str, int]] = {
    SECTOR_STEEL_BF: 80,
    SECTOR_STEEL_EAF: 40,
    SECTOR_CEMENT: 150,
    SECTOR_AMMONIA: 60,
    SECTOR_METHANOL: 60,
    SECTOR_REFINERY: 50,
    SECTOR_COAL_CHEM: 30,
}


def water_quota(sector: str, feedstock: str, advanced: bool = False) -> float:
    """Unit-product water intake, m3/t, for a sector and feedstock.

    Args:
        sector: One of `INDUSTRY_SECTORS`.
        feedstock: Feedstock/route key, or "default" for sectors without one.
        advanced: True returns 先进值 (new/retrofitted plants), False returns 通用值.

    Returns:
        Water intake per tonne of product, m3/t.

    Raises:
        KeyError: Sector/feedstock pair is not in the quota table.
        ValueError: The pair is present but its value has not been sourced yet.
    """
    key = (sector, feedstock)
    if key not in WATER_INTAKE_QUOTA_M3_PER_T:
        key = (sector, "default")
    value = WATER_INTAKE_QUOTA_M3_PER_T[key]
    if value is None:
        raise ValueError(
            f"water intake quota for {key} has not been sourced; refusing to guess. "
            "Fill it from GB/T 18916 before running any scenario that touches this sector."
        )
    general, advanced_value = value
    return advanced_value if advanced else general


# =========================================================================================
# ABATEMENT OPTIONS
# =========================================================================================
# Three routes per hub, the set the author selected on 2026-09-07: `unabated`, `ccs`, `h2`.
# No retirement route -- industrial demand is exogenous here, so a hub that stopped producing
# would silently drop its output rather than move it elsewhere. Whether industrial output can
# fall at all is a demand question this model does not answer.
#
# COST BASIS. Everything below is per TONNE, matching Tang et al. (2023, iScience 26:106347),
# whose formulation this extends: Z_cap(i) = sum_g [ CAPEX_g(a_g_i) * CRF + OPEX_g(a_g_i) ]
# with a_g_i the annual capture from sector g at hub i. The coal side of this repo instead
# works per MW and per MWh; the two never mix, they only meet in the shared CO2 network, the
# shared basin water cap and the single joint emission target.
#
# WHAT IS AND IS NOT INCLUDED. The capture parameters below cover the CAPTURE island only --
# separation, compression and on-site handling. Transport and storage are NOT in them: those
# come from the pipeline network and the storage hubs, exactly as for coal. Dropping a
# literature "full-chain CCUS cost" in here would double-count them.
#
# COST BASIS (author's decision 2026-09-22, same for coal and industry): every abatement option
# is priced as ONE-TIME RETROFIT CAPEX + FIXED O&M (a share of capex per year) + the energy and
# consumables it actually uses, priced at the model's own coal and electricity prices, with an
# end-of-horizon SALVAGE VALUE on the undepreciated capex. Levelised per-tonne capture costs
# (ACCA21-style CNY/t CO2) are NOT used in the objective any more: a levelised cost embeds a
# capital-recovery assumption that cannot be consistent with a dynamic model that decides
# WHEN to build and how long the asset then runs inside the horizon. The ACCA21 ranges are
# kept only as a cross-check on what the parameters below imply
# (`levelised_capture_cost_cny_per_t`).

INDUSTRY_ROUTES: Final[tuple[str, ...]] = ("unabated", "ccs", "h2")

# --- CO2 capture: retrofit capex, CNY per tonne of ANNUAL capture capacity ----------------
# Reference year 2030 (the same `ccs_learning_factor` the coal retrofits get scales it), incl.
# compression to pipeline pressure. Chinese project filings, capture island only:
#   cement   中联水泥青州 20 万 t/a 全氧燃烧耦合碳捕集示范线 2.56 亿元 -> 1 280 元/(t·a)
#            (中国建材, 2023 开工); 中联 20 万 t/a 捕集提纯项目 1.98 亿元 -> 990;
#            海螺白马山 5 万 t/a 5 500 万元 -> 1 100 (2018 投运, 含食品级提纯).
#            Range 990-1 280 at 0.05-0.2 Mt/a; 1 150 taken.
#   steel    包钢 200 万 t CCUS 一期 50 万 t/a 6.14 亿元 -> 1 228 (全产业链一期, 含部分输送,
#            upper); 日照钢铁 18 万 t/a 1.35 亿元 -> 750 (2025 开工, 捕集+资源化);
#            宝武案例 (PKU CCUS 2020, Baowu Zhanjiang slip-stream, amine): capture plant
#            CNY 360 M + 7% owner's cost + 20 M working capital = 407 M for 0.5 Mt/a -> 814;
#            IEAGHG 2013/04 Table 6, Case 2A: capture plant US$(2010) 679 M for ~4.7 Mt/a
#            captured -> ~145 USD/(t·a) ~ 1 000 元/(t·a). Central 1 000.
#   steel_eaf ASSUMPTION: no source. Off-gas is dilute and intermittent, i.e. harder than a
#            cement kiln, so it takes the cement value. EAF is 1.7% of modelled industrial CO2.
#   ammonia / methanol   high-concentration (>95%) gasification off-gas: no absorption, only
#            dehydration + compression/liquefaction. 延长石油榆林煤化 30 万 t/a (2022) reports
#            105 元/t all-in capture cost; net of ~110 kWh/t compression electricity at
#            ~0.45 元/kWh that leaves ~55 元/t for capital + fixed O&M, i.e. capex ~400-450
#            元/(t·a) at CRF(6%, 20 a) + 5%/a. ASSUMPTION-DERIVED (no filing gives the
#            investment itself; 齐鲁石化 100 万 t/a 未公布投资额). 450 taken.
INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 1000.0,
    SECTOR_STEEL_EAF: 1150.0,
    SECTOR_CEMENT: 1150.0,
    SECTOR_AMMONIA: 450.0,
    SECTOR_METHANOL: 450.0,
}
# Fixed O&M of the capture island as a share of (learning-adjusted) capex per year. Same
# 5%/a the coal side uses (An et al. 2025 Nat Commun SI Table 7: fixed O&M / investment =
# 5.4%/a); the PKU/Baowu case gives 12 M/a on 407 M = 2.9%/a, IEAGHG 2013/04 steel-mill
# maintenance 142/3 928 = 3.6%/a of installed cost. 5% is the conservative end.
INDUSTRY_CCS_FIXED_OM_FRACTION: Final[float] = 0.05
# Capture energy per tonne CO2 captured. Priced in `industry_year_data` at the model's own
# coal price (steam, via a boiler) and the scenario electricity price -- not at a frozen
# literature price -- so the industrial and the coal-side energy penalties move together.
#   amine post-combustion (steel BF gas / hot-stove flue, cement kiln flue, EAF off-gas):
#     reboiler steam 2.8 GJ/t: IEAGHG 2013/04 MDEA/Pz 2.3 GJ/t (CSIRO review: 2.5-2.7
#     achievable), 国能锦界 2nd-generation solvent 2.35 GJ/t (中国 CCUS 进展报告 2025 p.18),
#     MEA 3.0-3.5 GJ/t; 2.8 is the centre of that spread for a 2030 retrofit.
#     electricity 130 kWh/t incl. compression: PKU/Baowu 142 kWh/t total output penalty
#     (steam + power, Table 9); Gardarsdottir et al. 2019 cement MEA ~130 kWh/t.
#   high-concentration streams (ammonia, methanol): no reboiler; compression + dehydration
#     only, ~110 kWh/t (0.1 -> 11 MPa dense phase; 延长 105 元/t all-in is consistent).
INDUSTRY_CCS_STEAM_GJ_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 2.8,
    SECTOR_STEEL_EAF: 2.8,
    SECTOR_CEMENT: 2.8,
    SECTOR_AMMONIA: 0.0,
    SECTOR_METHANOL: 0.0,
}
INDUSTRY_CCS_ELECTRICITY_KWH_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 130.0,
    SECTOR_STEEL_EAF: 130.0,
    SECTOR_CEMENT: 130.0,
    SECTOR_AMMONIA: 110.0,
    SECTOR_METHANOL: 110.0,
}
# Solvent make-up, waste amine disposal, water: PKU/Baowu 40 000 元/t amine, 6.5 元/t CO2
# water; ~15 元/t CO2 for amine systems, 5 for compression-only.
INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 15.0,
    SECTOR_STEEL_EAF: 15.0,
    SECTOR_CEMENT: 15.0,
    SECTOR_AMMONIA: 5.0,
    SECTOR_METHANOL: 5.0,
}
# Reboiler steam is raised in a coal boiler at the site: coal per GJ steam = 1 / efficiency.
# Its CO2 is VENTED (the retrofit captures the process stream, not the auxiliary boiler --
# the PKU/Baowu case counts it the same way, 743 g/kWh on the auxiliary plant), so the net
# reduction of the CCS route is captured minus this steam CO2. Before 2026-09-22 the model
# took the ACCA21 unit cost as energy-inclusive and vented nothing.
INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY: Final[float] = 0.88
# Economic life of the capture island. PKU/Baowu assume 25 a; the coal side's retrofit
# island is tied to units with 15-25 a left; 20 a for both sides so the salvage rule treats
# a capture island the same wherever it is built.
INDUSTRY_CAPTURE_LIFETIME_YEARS: Final[int] = 20

# ACCA21 / China Energy News levelised capture costs, CNY per tonne, CROSS-CHECK ONLY (not in
# the objective since 2026-09-22). Cement 305-730 (ACCA21; China Energy News 430-650), steel 348-560, coal
# power 300-450, high-concentration coal chemicals <100 (ACCA21 105-250). The parameters
# above imply, at 38 元/GJ coal, 0.40 元/kWh and 6%/20 a: cement ~345, steel ~325,
# high-concentration ~110 元/t -- at the low end of each range (steel 7% below ACCA21's
# 348), because the capex anchors are Chinese filings rather than European FOAK estimates
# and the energy is priced at the model's coal and power prices, not at literature ones.
INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T: Final[dict[str, tuple[float, float]]] = {
    SECTOR_STEEL_BF: (348.0, 560.0),
    SECTOR_STEEL_EAF: (305.0, 730.0),
    SECTOR_CEMENT: (305.0, 730.0),
    SECTOR_AMMONIA: (105.0, 250.0),
    SECTOR_METHANOL: (105.0, 250.0),
}

# --- Water penalty of industrial capture, m3 per tonne CO2 captured ------------------------
# Amine capture needs cooling and a water wash. 1.60-1.69 m3/t CO2 for Chinese industrial
# retrofits (Wang, Wen & Xu, Nat Commun 16:4251, 2025); 1.65 taken.
# This sits ON TOP of the sector's GB/T 18916 unit-product intake, and it is the term that
# puts industrial CCS into competition with coal-power cooling for the same basin quota.
INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2: Final[float] = 1.65

# --- Hydrogen substitution route ------------------------------------------------------------
# Fraction of a hub's CO2 that the H2 route can remove. Not 1.0 anywhere: every route leaves a
# residue the hydrogen cannot touch.
#   steel_bf_bof  H2-DRI + EAF replaces coke reduction; lime, electrodes and the EAF's own grid
#                 electricity remain (~0.2-0.3 of 1.8 t CO2 per t crude steel).
#   ammonia       green H2 replaces gasification + water-gas shift entirely; utilities and the
#                 air separation unit remain.
#   methanol      green H2 adjusts the H/C ratio, it does not remove the carbon feedstock, so
#                 the residual is larger than for ammonia.
# ASSUMPTION in all three cases -- these are route-chemistry judgements, not quoted values.
INDUSTRY_H2_ABATEMENT_FRACTION: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 0.85,
    SECTOR_AMMONIA: 0.95,
    SECTOR_METHANOL: 0.90,
}

# Net incremental cost of the H2 route, CNY per tonne of PRODUCT, at the reference hydrogen
# price beside it. "Net" = green H2 purchase + new-route CAPEX + O&M - avoided fossil
# feedstock/reductant. The model re-prices it at the hydrogen price of the year being solved:
#
#     premium(P) = premium_ref + h2_intensity_t_per_t * 1000 * (P - P_ref)   [CNY / t product]
#
# i.e. a first-order expansion around the literature point. Exact in the hydrogen-price
# dimension, which is the dominant term (~70% of green-methanol cost per NER 2024), and
# constant in everything else. `h2_intensity` is not a constant here: it comes from the
# point-source table's own h2_demand_kt_per_year / production_kt_per_year (81 / 180 / 190 kg
# H2 per tonne of steel / ammonia / methanol), so the slope is per hub.
#
# Anchors, all China:
#   steel     green premium ~225 USD/t crude steel at 5 USD/kg H2 (Transition Asia / Global
#             Efficiency Intelligence, Green Steel Economics 2024) -> 1575 CNY/t at the repo's
#             7.0 CNY/USD, reference price 35.0 CNY/kg.
#   ammonia   green-H2 ammonia ~2870 CNY/t NH3 against coal-based ammonia 2380-2560 CNY/t
#             (China Energy News 2023) -> premium ~400 CNY/t at the ~12 CNY/kg hydrogen implied
#             by that case's wind/solar assumption.
#   methanol  green-H2 methanol 4500-5500 CNY/t at hydrogen 15-18 CNY/kg against fossil
#             ~2000 CNY/t (NER 2024) -> premium ~3000 CNY/t at reference price 16.5 CNY/kg.
# The three anchors sit at three different reference hydrogen prices on purpose: each is quoted
# at the price its own source assumed, and the expansion moves each to the model's price.
# Averaging them onto a common reference first would discard information, not add it.
INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT: Final[dict[str, tuple[float, float]]] = {
    # sector: (premium_cny_per_t_product, reference_h2_price_cny_per_kg)
    SECTOR_STEEL_BF: (1575.0, 35.0),
    SECTOR_AMMONIA: (400.0, 12.0),
    SECTOR_METHANOL: (3000.0, 16.5),
}

# Retrofit capex of the H2 route, CNY per tonne of ANNUAL product capacity, booked once on
# the route-share increment and depreciated straight-line for the salvage rule.
#   steel_bf_bof  H2-DRI shaft + EAF replacing BF-BOF at an existing site (sinter, coke
#                 ovens, BF and BOF are abandoned; casting and rolling stay):
#                 shaft 宝钢湛江百万吨级氢基竖炉 总投资 18.9 亿元 for 1.0 Mt/a DRI
#                 (中国钢铁新闻网 2022-02-17; 2023-12 投产) -> 1 890 元/(t DRI·a), ×1.08 t
#                 DRI per t crude steel = 2 040; EAF 184 EUR/(t·a) (Vogl, Åhman & Nilsson
#                 2018, J Clean Prod 203:736, cost assumptions) ~ 1 430 元/(t·a) at 7.8
#                 元/EUR -- no Chinese per-tonne EAF filing was found (the "80 t 电炉 5 000
#                 万元" figures are furnace-only). Sum ~3 470; 3 500 taken.
#   ammonia       existing coal-based plant switched to purchased green H2: the Haber-Bosch
#                 loop and the air separation unit are RETAINED, the gasifier, shift and
#                 purification train are idled. New: H2 receiving/compression, N2 tie-in,
#                 controls. ASSUMPTION 500 元/(t·a) ~ 8% of a greenfield synthesis island
#                 (875 USD/(t·a), `constants.NH3_HB_CAPEX_USD_PER_TONNE_YEAR`). No Chinese
#                 retrofit filing found; Yara Pilbara (2022-24) publishes only the
#                 electrolyser side.
#   methanol      绿氢耦合煤制甲醇: hydrogen replaces the shift stage to fix the H/C ratio,
#                 the synthesis loop is retained. Same tie-in scope as ammonia; ASSUMPTION
#                 500 元/(t·a).
INDUSTRY_H2_ROUTE_CAPEX_CNY_PER_T_PRODUCT_YR: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 3500.0,
    SECTOR_AMMONIA: 500.0,
    SECTOR_METHANOL: 500.0,
}
# Fixed O&M of the new route's equipment, share of capex per year. IEAGHG 2013/04 Tables
# 6-7: steel-mill maintenance 142 M$/a on 3 928 M$ installed = 3.6%/a; 3.5% taken.
INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION: Final[float] = 0.035
# Economic life of the rebuilt route (DRI shaft, EAF, synthesis tie-in): 25 a (PKU/Baowu
# use 25 a for a capture retrofit; a DRI/EAF module is a longer-lived asset than that).
INDUSTRY_H2_LIFETIME_YEARS: Final[int] = 25

# The literature premium anchors above are LEVELISED (they contain the route's own capital
# recovery). With capex now explicit, the anchor is decomposed, not discarded:
#     premium_ref = k * P_ref + capex * (CRF(r, life) + fom) + opex_delta_nonH2
# so `opex_delta_nonH2` -- the non-hydrogen operating difference against the incumbent
# fossil route (electricity for the EAF/compressors, avoided coke or coal, avoided fossil-
# route O&M) -- is what the anchor implies once its hydrogen and its capital are taken out.
# It is negative for steel (avoided coke and BF opex exceed the EAF power bill), which the
# anchor's own arithmetic forces; the model's floor (fixed O&M + opex_delta + H2 purchase
# >= 0, capex annuity always paid) still holds.

# Site water for a hub on the H2 route: the sector's advanced value rather than its general
# value. Not a guess -- the advanced value is by definition the quota applied to new and
# rebuilt plants, and the H2 route is a rebuild. GB/T 18916 has no H2-DRI row, and inventing
# one was the alternative. Set False to hold site water at the general value instead.
INDUSTRY_H2_USES_ADVANCED_QUOTA: Final[bool] = True

# NOT MODELLED, and it has to stay visible: electrolysis raw-water demand, 10-22 L/kg H2
# (Arup, "Water for Hydrogen" 2022; Energy UK 2022 review), lands in the basin of the
# ELECTROLYSER, not of the industrial hub. The coal side's ammonia co-firing does not charge it
# either, so leaving it out is at least consistent between the two -- but at the 103 Mt/yr of
# hydrogen that full industrial substitution would need, it is 1.0-2.3e9 m3/yr, i.e. 10-23
# 亿 m3, which is 5-11x the entire enforced residual of basin K. However it is added later, it
# belongs to the supply node, and both hydrogen users must be charged at the same time.
INDUSTRY_ELECTROLYSIS_WATER_L_PER_KG_H2: Final[tuple[float, float]] = (10.0, 22.0)


def capital_recovery_factor(rate: float, life_years: int) -> float:
    """Capital recovery factor, the reciprocal of the annuity factor.

    Args:
        rate: Discount rate per year.
        life_years: Economic life in years (floored at 1).

    Returns:
        Annual payment per unit of capital that repays it over `life_years` at `rate`.
    """
    n = max(1, int(life_years))
    if rate <= 1e-9:
        return 1.0 / n
    return rate / (1.0 - (1.0 + rate) ** (-n))


def _require(table: dict[str, float], sector: str, what: str) -> float:
    if sector not in table:
        raise KeyError(
            f"no {what} sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(table)}"
        )
    return float(table[sector])


def capture_capex_cny_per_t_yr(sector: str) -> float:
    """Retrofit capex of the capture island, CNY per tonne of annual capture capacity.

    Args:
        sector: One of `INDUSTRY_SECTORS`.

    Returns:
        Capex at the 2030 reference year, before learning and the scenario multiplier.

    Raises:
        KeyError: Sector has no sourced capture capex.
    """
    return _require(INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR, sector, "capture capex")


def capture_variable_cost_cny_per_t(
    sector: str, coal_price_cny_per_gj: float, electricity_price_cny_per_mwh: float
) -> float:
    """Energy and consumables per tonne CO2 captured, at the given fuel and power prices.

    Args:
        sector: One of `INDUSTRY_SECTORS`.
        coal_price_cny_per_gj: Delivered coal price used to raise reboiler steam.
        electricity_price_cny_per_mwh: Electricity price of the year being priced.

    Returns:
        CNY per tonne CO2 captured: steam coal + electricity + consumables.
    """
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    kwh = _require(INDUSTRY_CCS_ELECTRICITY_KWH_PER_T_CO2, sector, "capture electricity")
    consumables = _require(INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2, sector, "capture consumables")
    steam_coal = steam_gj / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_price_cny_per_gj)
    return steam_coal + kwh / 1000.0 * float(electricity_price_cny_per_mwh) + consumables


def capture_steam_co2_t_per_t(sector: str, coal_emission_factor_t_per_gj: float) -> float:
    """Vented CO2 from raising the reboiler steam, tonnes per tonne CO2 captured."""
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    return steam_gj / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_emission_factor_t_per_gj)


def levelised_capture_cost_cny_per_t(
    sector: str,
    discount_rate: float,
    coal_price_cny_per_gj: float,
    electricity_price_cny_per_mwh: float,
    learning: float = 1.0,
) -> float:
    """What the capex + O&M + energy parameters imply as a levelised CNY per tonne captured.

    Reporting and cross-check only (against `INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T`);
    the objective never uses a levelised figure.
    """
    capex = capture_capex_cny_per_t_yr(sector) * float(learning)
    crf = capital_recovery_factor(discount_rate, INDUSTRY_CAPTURE_LIFETIME_YEARS)
    return capex * (crf + INDUSTRY_CCS_FIXED_OM_FRACTION) + capture_variable_cost_cny_per_t(
        sector, coal_price_cny_per_gj, electricity_price_cny_per_mwh
    )


def h2_route_capex_cny_per_t_yr(sector: str) -> float:
    """Retrofit capex of the H2 route, CNY per tonne of annual product capacity."""
    return _require(INDUSTRY_H2_ROUTE_CAPEX_CNY_PER_T_PRODUCT_YR, sector, "H2-route capex")


def h2_route_annual_capital_cny_per_t(sector: str, discount_rate: float) -> float:
    """Capex annuity plus fixed O&M of the H2 route, CNY per tonne of product per year.

    What the anchor decomposition takes out of `premium_ref` as capital. NB the solver's
    floor keeps only the annuity outside the `max` (fixed O&M is part of the annual term);
    see `h2_premium_cny_per_t`.
    """
    capex = h2_route_capex_cny_per_t_yr(sector)
    crf = capital_recovery_factor(discount_rate, INDUSTRY_H2_LIFETIME_YEARS)
    return capex * (crf + INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION)


def h2_route_opex_delta_cny_per_t(
    sector: str, h2_intensity_t_per_t: float, discount_rate: float, multiplier: float = 1.0
) -> float:
    """Non-hydrogen operating difference of the H2 route against the incumbent, CNY/t product.

    Backed out of the literature premium anchor once its hydrogen (at the anchor's own
    reference price) and the explicit capex annuity + fixed O&M are removed. Negative where
    the avoided fossil feedstock and incumbent O&M exceed the new route's power bill.

    `multiplier` is the scenario's `industry_h2_cost_multiplier`: it scales the route's OWN
    costs (the anchor premium and the capex together), never the hydrogen, so that at the
    anchor's reference price the levelised premium is exactly `multiplier x premium_ref`.
    Scaling the backed-out delta alone would invert the knob (the delta is negative).

    Args:
        sector: One of `INDUSTRY_SECTORS` whose `SECTOR_HAS_H2_ROUTE` entry is true.
        h2_intensity_t_per_t: Tonnes of H2 per tonne of product at this hub.
        discount_rate: Scenario discount rate, for the capex annuity.
        multiplier: Scenario cost multiplier on the route's own costs.

    Raises:
        KeyError: Sector has no sourced H2 premium anchor or capex.
    """
    if sector not in INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT:
        raise KeyError(
            f"no H2 premium anchor sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT)}"
        )
    premium_ref, price_ref = INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT[sector]
    hydrogen_at_ref = float(h2_intensity_t_per_t) * 1000.0 * float(price_ref)
    own_cost_at_ref = float(premium_ref) - h2_route_annual_capital_cny_per_t(sector, discount_rate)
    return float(multiplier) * own_cost_at_ref - hydrogen_at_ref


def h2_premium_cny_per_t(
    sector: str,
    h2_price_cny_per_kg: float,
    h2_intensity_t_per_t: float,
    discount_rate: float = 0.06,
    multiplier: float = 1.0,
) -> float:
    """Levelised net premium of the H2 route at a given H2 price, CNY per tonne of product.

    Reporting only (figure panels); the solver books capex once and buys hydrogen per link.
    Mirrors the solver's own arithmetic: capex annuity + max(fixed O&M + opex delta +
    hydrogen, 0). The floor is the one `add_industry_year` applies (annual cost incl. fixed
    O&M + hydrogen purchase >= 0) with the capex annuity outside it: switching cannot be
    cheaper than running the incumbent's sunk asset, however cheap hydrogen gets, and the
    new route's capital is always paid.

    Args:
        sector: One of `INDUSTRY_SECTORS` whose `SECTOR_HAS_H2_ROUTE` entry is true.
        h2_price_cny_per_kg: Delivered green hydrogen price in the year being priced.
        h2_intensity_t_per_t: Tonnes of H2 per tonne of product at this hub.
        discount_rate: Scenario discount rate, for the capex annuity.
        multiplier: Scenario `industry_h2_cost_multiplier` (see `h2_route_opex_delta_cny_per_t`).
    """
    capex = h2_route_capex_cny_per_t_yr(sector) * float(multiplier)
    annuity = capex * capital_recovery_factor(discount_rate, INDUSTRY_H2_LIFETIME_YEARS)
    fixed_om = capex * INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION
    opex_delta = h2_route_opex_delta_cny_per_t(sector, h2_intensity_t_per_t, discount_rate, multiplier)
    hydrogen = float(h2_intensity_t_per_t) * 1000.0 * float(h2_price_cny_per_kg)
    return annuity + max(fixed_om + opex_delta + hydrogen, 0.0)
