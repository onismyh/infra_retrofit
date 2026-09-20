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
# WHAT IS AND IS NOT INCLUDED. `INDUSTRY_CAPTURE_COST_CNY_PER_T` is the CAPTURE cost only --
# separation, compression and on-site handling. Transport and storage are NOT in it: those
# come from the pipeline network and the storage hubs, exactly as for coal. Dropping a
# literature "full-chain CCUS cost" in here would double-count them.

INDUSTRY_ROUTES: Final[tuple[str, ...]] = ("unabated", "ccs", "h2")

# --- CO2 capture cost, CNY per tonne CO2 captured -----------------------------------------
# BIBLIOGRAPHY KEY: acca21_2023_ccus, cpnn_2023_ccus_cost.
#
# Two Chinese sources, same ordering, overlapping ranges (cement > steel > power >
# high-concentration chemicals). The ordering is physical: capture cost scales with the
# inverse of flue-gas CO2 concentration, which runs ~14-33% at a cement kiln (calcination CO2
# included), ~20-27% in blast-furnace gas, and >95% in coal-gasification syngas after the
# water-gas shift.
#
#   ACCA21 China CCUS Annual Report (2023): cement 305-730; high-concentration chemicals 105-250
#   China Energy News 2023 (ACCA21-derived): cement 430-650; steel 348-560; coal power 300-450;
#                                            high-concentration coal chemicals below 100
#
# Central values below are midpoints of the narrower (China Energy News) ranges, which sit
# inside the ACCA21 ranges wherever the two overlap. Cross-check: the production-weighted mean
# over the five sectors is ~430 CNY/t; Tang et al. (2023) report a whole-chain 50.6 USD/t with
# transport 2.1 and storage 6.9, i.e. capture ~41.6 USD/t ~ 291 CNY/t -- lower, as it should
# be, because that average is 2050 (post-learning) and 59% of it is the cheaper power sector.
#
# ASSUMPTION (steel_eaf): no Chinese source gives an EAF capture cost. EAF off-gas is dilute
# and intermittent, i.e. harder than a cement kiln, so it takes the cement value rather than
# the steel one. EAF is 56 of 3 269 Mt (1.7%) of modelled industrial CO2, so the choice cannot
# drive any result; it is recorded rather than hidden.
INDUSTRY_CAPTURE_COST_CNY_PER_T: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 454.0,     # (348+560)/2
    SECTOR_STEEL_EAF: 540.0,    # cement value, see ASSUMPTION above
    SECTOR_CEMENT: 540.0,       # (430+650)/2
    SECTOR_AMMONIA: 177.5,      # (105+250)/2, ACCA21 high-concentration chemicals
    SECTOR_METHANOL: 177.5,     # same stream class: gasification syngas after the shift
}

# Share of the capture cost that is CAPITAL rather than annual O&M. Charged once on the
# installed-stock increment, the way the coal side charges `ccs_retrofit_capex`; the remainder
# is an annual cost. Tang et al. (2023) split Z_cap into CAPEX*CRF + OPEX but publish neither
# term separately, so the split is taken from the coal-CCS structure already in this repo:
# An et al. (2025, Nat Commun) SI Table 7 gives fixed O&M / investment = 5.4%/yr, which over a
# 20-year life at this model's discount rate puts roughly 55-60% of levelised cost in capital.
# ASSUMPTION, exposed for sensitivity through `scenario.industry_cost_multiplier`.
INDUSTRY_CAPTURE_CAPEX_SHARE: Final[float] = 0.58
# Capital recovery: the capital half above is a LEVELISED (annual) quantity, so it is
# de-annualised with this life before being booked as the one-time charge.
INDUSTRY_CAPTURE_LIFETIME_YEARS: Final[int] = 20

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

# Share of the H2-route premium that is CAPITAL. The H2 route is a plant rebuild (DRI shaft +
# EAF; an electrolyser-fed synthesis loop), not a bolt-on, so the capital share is higher than
# for capture. ASSUMPTION, exposed for sensitivity.
INDUSTRY_H2_CAPEX_SHARE: Final[float] = 0.35
INDUSTRY_H2_LIFETIME_YEARS: Final[int] = 25

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


def capture_cost_cny_per_t(sector: str) -> float:
    """CO2 capture cost for a sector, CNY per tonne captured (capture only, no T&S).

    Args:
        sector: One of `INDUSTRY_SECTORS`.

    Returns:
        Capture cost, CNY per tonne CO2.

    Raises:
        KeyError: Sector has no sourced capture cost.
    """
    if sector not in INDUSTRY_CAPTURE_COST_CNY_PER_T:
        raise KeyError(
            f"no capture cost sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(INDUSTRY_CAPTURE_COST_CNY_PER_T)}"
        )
    return float(INDUSTRY_CAPTURE_COST_CNY_PER_T[sector])


def h2_premium_cny_per_t(
    sector: str, h2_price_cny_per_kg: float, h2_intensity_t_per_t: float
) -> float:
    """Net incremental cost of the H2 route, CNY per tonne of product, at a given H2 price.

    Args:
        sector: One of `INDUSTRY_SECTORS` whose `SECTOR_HAS_H2_ROUTE` entry is true.
        h2_price_cny_per_kg: Delivered green hydrogen price in the year being priced.
        h2_intensity_t_per_t: Tonnes of H2 per tonne of product at this hub.

    Returns:
        Net premium over the incumbent fossil route, CNY per tonne of product. Can go negative
        if hydrogen falls far enough below the anchor's reference price.

    Raises:
        KeyError: Sector has no sourced H2 premium anchor.
    """
    if sector not in INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT:
        raise KeyError(
            f"no H2 premium anchor sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT)}"
        )
    premium_ref, price_ref = INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT[sector]
    premium = float(premium_ref) + float(h2_intensity_t_per_t) * 1000.0 * (
        float(h2_price_cny_per_kg) - float(price_ref)
    )
    # FLOOR at the capital half of the anchor. The route's own rebuild capital -- a DRI shaft
    # and an EAF, or an electrolyser-fed synthesis loop -- does not fall when hydrogen gets
    # cheaper, so the premium cannot fall below it however far the price drops. Without this
    # floor the expansion goes NEGATIVE for steel by 2060 (1575 + 81 * (12.4 - 35) = -255
    # CNY/t), i.e. the model would be paid to convert every blast furnace, and 1 437 Mt of
    # abatement would arrive free. That is an artefact of applying a GREENFIELD LCOS
    # comparison to existing plants whose incumbent capital is already sunk: switching cannot
    # be cheaper than running a paid-for asset. The floor is still generous -- it credits the
    # avoided fossil feedstock in full -- so H2 uptake remains an upper bound.
    return max(premium, float(premium_ref) * INDUSTRY_H2_CAPEX_SHARE)
