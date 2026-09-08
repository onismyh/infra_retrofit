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
