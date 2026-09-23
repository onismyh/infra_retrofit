from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import PLANNING_YEARS


PATHWAYS = ("unabated", "retire", "ccs", "biomass", "beccs", "ammonia")


@dataclass(frozen=True)
class OptimizationAssumptions:
    capacity_factor: float = 0.55
    # Fleet-average emission intensity. Kept at 0.82 t/MWh: with the efficiency-anchored
    # heat rate below it implies 95.7 kgCO2/GJ, within the IPCC bituminous range (~94.6-96.1).
    coal_emission_factor_t_per_mwh: float = 0.82
    # Derived from the efficiency anchor: 3.6 / 0.42 = 8.5714 GJ/MWh (was 9.0, which
    # implied 40% efficiency and was inconsistent with coal_plant_base_efficiency).
    heat_rate_gj_per_mwh: float = 8.5714
    nh3_lhv_gj_per_kg: float = 0.0186
    usd_to_cny: float = 7.0
    retire_cost_cny_per_mwh: float = 450.0
    # CCS/BECCS capture-island O&M is charged as ccs_om_fraction x retrofit CAPEX per year
    # (see ccs_om_fraction below), so no separate per-MWh adder: keeping both double-counted
    # the same cost. The BECCS per-MWh term retains only the biomass co-firing O&M, which is
    # the same 30 CNY/MWh the pure-biomass pathway carries (Wang & Cai 2024 SI Table 3, gamma2
    # 18.85 $/kW/yr ~ 132 CNY/kW/yr at fleet-average operating hours).
    ccs_fixed_cost_cny_per_mwh: float = 0.0
    biomass_fixed_cost_cny_per_mwh: float = 30.0
    beccs_fixed_cost_cny_per_mwh: float = 30.0
    ammonia_fixed_cost_cny_per_mwh: float = 80.0
    # Capture-island retrofit CAPEX at the learning reference year (2030), incl. compression,
    # 90% amine capture on an existing 300-1000 MW unit. Literature review 2026-09-10
    # (docs/工业联合减排实现说明.md §9.6): centre 3 500 CNY/kW, range 2 700-4 400.
    #   Yuan J-H et al. 2022, 气候变化研究进展 18(6) 764-776, Table 2: 3 318-3 925 CNY/kW by
    #     province, retrofit (primary citation);
    #   Lockwood 2018, IEA Clean Coal Centre for CIAB, Table 4: 4 121 CNY(2016)/kW, 1000 MW
    #     USC, 90%, capture+compression, 2025-2030 retrofit, self-described conservative (upper);
    #   An K et al. 2025, Nat Commun 16:2311, SI Table 7: 381.9 / 305.5 $/kW in 2025 / 2030
    #     (~2 670 / 2 140 CNY/kW), learning-curve lower end;
    #   国能锦界 4 Mt/a full-flue-gas CCUS, 19.9 亿元 on a 600 MW-class unit (2024 filing):
    #     ~3 000-3 300 CNY/kW, a real retrofit incl. pilot storage.
    # Was 4 000 (unsourced) before 2026-09-10; the BECCS figure keeps its +1 000 CNY/kW
    # biomass-conversion increment on top of the capture island (two-stock capex).
    ccs_retrofit_capex_cny_per_kw: float = 3500.0
    beccs_retrofit_capex_cny_per_kw: float = 4500.0
    biomass_efficiency_penalty_per_ratio: float = 0.0373  # 0.56% eff drop at 15% co-firing (Fan et al. 2023)
    coal_plant_base_efficiency: float = 0.42
    coal_fuel_cost_cny_per_gj: float = 38.2             # national mean, overridden by province lookup
    # Fixed O&M of the capture island as a share of (learning-adjusted) retrofit CAPEX per year.
    # An et al. 2025 (Nat Commun) SI Table 7 reports fixed O&M / investment = 20.7/381.9 = 5.4%
    # in every projection year for coal CCS retrofits; 5% is used here.
    ccs_om_fraction: float = 0.05
    # CCS energy penalty expressed as EXTRA FUEL per unit of output (dimensionless, = extra
    # GJ / baseline GJ). Replaces the old absolute efficiency-penalty point value: the model
    # holds generation fixed and buys extra coal, so the extra-fuel ratio is the quantity that
    # actually enters cost and emissions. Level anchored at 15% in 2030; the decline follows
    # An et al. 2025 SI Table 7 (coal energy penalty 22.2 / 15.6 / 13.3 / 11.1 % for
    # 2030/2040/2050/2060, i.e. normalised 1.000 / 0.703 / 0.599 / 0.500).
    ccs_energy_penalty_ratio_by_year: tuple[float, ...] = (0.1500, 0.1054, 0.0899, 0.0750)

    # Province-specific coal prices (An et al. 2025, Nat Commun, Supp Table 2)
    # Unit: CNY/GJ, converted from USD/GJ × 7 (USD/CNY)
    province_coal_cost_cny_per_gj: dict[str, float] = field(default_factory=lambda: {
        "Anhui": 42.8, "Beijing": 69.4, "Chongqing": 42.6,
        "Fujian": 40.0, "Gansu": 37.0, "Guangdong": 41.5,
        "Guangxi": 47.9, "Guizhou": 34.6, "Hainan": 33.6,
        "Hebei": 30.8, "Heilongjiang": 35.6, "Henan": 42.1,
        "Hubei": 46.2, "Hunan": 48.5, "Inner Mongolia": 17.9,
        "Jiangsu": 40.7, "Jiangxi": 48.2, "Jilin": 39.1,
        "Liaoning": 37.5, "Ningxia": 27.4, "Qinghai": 33.9,
        "Shaanxi": 31.0, "Shandong": 41.5, "Shanghai": 38.1,
        "Shanxi": 28.2, "Sichuan": 45.4, "Tianjin": 38.6,
        "Xinjiang": 16.8, "Yunnan": 29.6, "Zhejiang": 39.4,
    })

    def province_coal_cost(self, province_name: str) -> float:
        """Get coal fuel cost for a province (CNY/GJ). Falls back to national mean."""
        return self.province_coal_cost_cny_per_gj.get(province_name, self.coal_fuel_cost_cny_per_gj)
    # Technology learning curve (exogenous Wright's law).
    # LR=15% per doubling of cumulative capture capacity (capture-island learning rates in the
    # literature: 9.6-20.2% for China IGCC+CC, Li et al. 2012 Appl. Energy; ~10% full chain,
    # DNV/Gassnova 2020), with a doubling every 5.6 years -- consistent with scaling China's
    # CCS fleet from ~4 Mt/yr today to several hundred Mt/yr by 2060 (~6 doublings in 35 yr).
    # The resulting calendar decline is fitted (log-linear through 2030) to the coal-CCS
    # retrofit CAPEX trajectory of An et al. 2025 SI Table 7 (305.5 -> 142.5 $/kW, 2030->2060):
    #   model factor 1.000 / 0.748 / 0.560 / 0.419   vs   source 1.000 / 0.640 / 0.519 / 0.466
    # i.e. the model is slightly conservative mid-century and slightly optimistic at 2060.
    ccs_learning_rate: float = 0.15
    ccs_deployment_doubling_years: float = 5.6
    ccs_learning_reference_year: int = 2030
    # Net system cost framework
    baseline_om_cost_cny_per_mwh: float = 80.0       # non-fuel O&M for baseline coal operation
    stranded_asset_base_cny_per_kw: float = 3500.0    # new-build cost for stranded asset calculation
    stranded_asset_accounting_life: int = 20           # depreciation years
    # Storage cost 32 CNY/t vs An et al. 2025 SI Table 7: 5.0 (3.0-8.5) $/t = 35 (21-60) CNY/t.
    storage_cost_cny_per_t: float = 32.0
    eor_credit_cny_per_t: float = 12.0
    # --- Hub injection rate, derived from candidate-site density -------------------------
    # The source raster (Fan 5 km grid; see data/封存汇图层-Fan) stores, per 5x5 km cell, the
    # GEOLOGICAL injection rate the formation could accept (national mean 8.0, max 134 Mt/a
    # per cell) -- verified against the dataset's own provincial sheet, whose Max column
    # reproduces the per-cell raster maximum province by province. Summing cells over a basin
    # therefore has no engineering meaning (it totals 555 Gt/yr nationally).
    # Instead the buildable rate is derived as:
    #     hub injectivity = (cells in hub / storage_site_block_pixels) x site project rate
    # i.e. one storage project per 50x50 km block (=100 cells of 5 km; pressure-interference
    # spacing), each at real project scale (Qilu-Shengli 1 Mt/yr, Gorgon 4 Mt/yr design ->
    # 2 Mt/yr central). National total = 1 350 Mt/yr, inside the ACCA21/CAEP 2060 CCUS
    # deployment range of 1 000-1 800 Mt/yr. max_hub_injectivity_mtpa is now only a ceiling
    # for pathological hubs. All three are applied before injectivity_multiplier.
    storage_site_block_pixels: float = 100.0
    storage_site_project_rate_mtpa: float = 2.0
    max_hub_injectivity_mtpa: float = 200.0
    # Pipeline O&M. An et al. 2025 SI Table 7 gives all-in CO2 transport 0.026 (0.020-0.036)
    # $/(t·km) = 0.182 (0.14-0.25) CNY/(t·km). This model charges pipeline CAPEX separately
    # (~0.029 CNY/(t·km) annualised at full utilisation), so 0.15 opex + 0.029 capex = 0.179
    # CNY/(t·km) levelised reproduces the source's central value.
    route_opex_cny_per_t_km: float = 0.15
    # Offshore basins (East China Sea, Pearl River Mouth, Bohai, Beibu Gulf) carry subsea
    # pipeline and platform costs: transport CAPEX and OPEX are scaled by this factor on any
    # edge touching an offshore storage hub.
    offshore_transport_multiplier: float = 1.5
    pipe_capex_cny_per_mtpa_km: float = 400_000.0  # trunk-scale (>=20 Mtpa) factor-based estimate:
    # MIT Smith et al. 2021 (IJGGC) $52,892/(in·mi) → 20-in trunk ≈4.6M CNY/km ≈0.23e6 CNY/(Mtpa·km);
    # ADB China factor 57,124 USD/(km·in) → ≈0.4e6; small-scale actual (Qilu-Shengli, 1.7 Mtpa,
    # incl. stations) 3.1e6. Base rate represents the standard 20-Mtpa pipe; smaller pipes carry
    # branch/direct multipliers. (Previous value 18,000 was a unit error, ~100x too low.)
    # Free CO2 capacity credited to the 62 candidate edges that follow an existing oil/gas
    # trunk (`existing_corridor_flag`). Was 20 Mtpa (unsourced) before 2026-09-10; the
    # literature review (docs/工业联合减排实现说明.md §9.6) found the opposite: repurposed gas
    # lines carry small flows at reduced pressure (IEAGHG 2013/18: Longannet 1.8 Mt/yr, OCAP
    # de-rated 56 -> 21 bar) and dense-phase CO2 at 20 Mt/yr over >150 km is explicitly ruled
    # out for gas-pipe wall thickness (Smith et al. 2021 via Drax DR1907-6; Anvari et al.
    # 2025). The corridor edges here average 393 km (min 22, max 1 349), so no reuse credit
    # is defensible on the main line. A reuse sensitivity may set 2.0 (IEAGHG case scale).
    existing_corridor_capacity_mtpa: float = 0.0
    standard_pipe_capacity_mtpa: float = 20.0
    max_parallel_pipes: int = 2
    pipeline_lifetime_years: int = 30
    # Pipe DIAMETER TIERS. Before 2026-09-10 every edge had exactly one size, the 20-Mtpa trunk,
    # and a 1-Mt/yr branch paid for 20 Mtpa: 222 of the 344 edges built in IND_BASE_t95 sat at
    # exactly 20, and any flow too small to justify a trunk was cheaper to push through the
    # big-M "capacity slack" than to pipe -- so 11 edges carried CO2 with zero built capacity.
    # Three tiers with economies of scale close that gap. Per-km capex per tier scales the
    # 20-Mtpa trunk rate (400 000 x 20 = 8.0e6 CNY/km, see pipe_capex_cny_per_mtpa_km) by
    # (cap/20)^0.6, the usual diameter-cost exponent for CO2 pipelines (Knoope et al. 2013,
    # IJGGC 16:241, Table 4 fits 0.5-0.7). Cross-check: the 2-Mtpa tier at 2.0e6 CNY/km sits
    # below the 1.7-Mtpa Qilu-Shengli line's 3.1e6 CNY/km, which includes its compressor
    # stations, so the small tier is if anything cheap. Class multipliers (branch 1.35, direct
    # 2.8, corridor 0.97) still apply on top, as before.
    pipe_capacity_tiers_mtpa: tuple[float, ...] = (2.0, 5.0, 20.0)
    pipe_capex_cny_per_km_by_tier: tuple[float, ...] = (2.0e6, 3.5e6, 8.0e6)
    # Storage DEPLOYMENT RAMP. `injectivity_mtpa` is the 2060-scale buildable rate (calibrated
    # to the ACCA21 2060 range, see storage_site_project_rate_mtpa). Offering all of it in
    # 2030 let IND_BASE_t95 inject 1 265 Mt/yr in 2040 on carbon price alone, against ~4 Mt/yr
    # injected nationally today. The fraction available in each planning year follows the
    # midpoints of the ACCA21 (2021) CCUS roadmap: 2030 0.2-4.08 亿 t (mid 2.1), 2050 6-14.5
    # (10.2), 2060 10-18.2 (14.1); 2040 is interpolated between the 2035 and 2050 ranges
    # (~7.5). Divided by the 12.8 亿 t/yr national buildable rate and capped at 1.
    storage_deployment_fraction_by_year: tuple[float, ...] = (0.17, 0.58, 0.80, 1.00)
    # Green-ammonia SUPPLY RAMP, same device. The supply curve is a TECHNICAL POTENTIAL
    # (8 551 Mt NH3/yr in 2025 against ~1 600 Mt for 50% co-firing of the whole fleet), so
    # without a ramp the node limits never bind. All-ones by default because no sourced
    # build-out trajectory for Chinese green ammonia is in the repo yet; set it when one is.
    ammonia_supply_deployment_fraction_by_year: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # Industrial hydrogen delivered by tube trailer from the supply node. INTERIM VALUE: the
    # order of magnitude of Chinese long-tube-trailer costs (~3 CNY/kg per 100 km, 中国氢能联盟
    # 白皮书 2019 range 2-4), pending a sourced figure from the author. Same role as
    # `ammonia_transport_cost_cny_per_kg_km` on the coal side.
    h2_transport_cost_cny_per_kg_km: float = 0.03
    cooling_once_through_water_intensity_m3_per_mwh: float = 0.35  # consumption basis (耗水量):
    # 0.29-0.41 m³/MWh per NDRC et al. 2015 No.9 clean-production benchmarks (median ≈0.35).
    # (Previous 1.0 mixed up the withdrawal and consumption bases.)
    cooling_recirculating_water_intensity_m3_per_mwh: float = 1.85
    cooling_air_water_intensity_m3_per_mwh: float = 0.37
    ccs_water_multiplier: float = 1.82
    biomass_water_multiplier: float = 1.00
    beccs_water_multiplier: float = 1.82
    ammonia_water_multiplier: float = 1.01
    # Water supply cost (for grid_supply mode)
    # Share of a province's renewable water already committed to agriculture, households
    # and other industry, deducted before any is offered to power plants. 0 keeps the raw
    # physical availability; set per the water bulletin once the withdrawal data is loaded.
    # Share of a province's renewable water already committed to agriculture, households and
    # other industry, deducted before any is offered to power plants.
    #
    # The residual structure is Richter et al. (2012) River Res. Applic. 28(8):1312-1321 own
    # prescription, not an analogy: 20% of natural monthly mean flow "can be allocated for
    # consumptive use", assessed "when added to already-existing water uses" (his Table II is
    # headed "Cumulative allowable depletion"). In Smakhtin et al. (2004) Eq.(1) terms this
    # parameter IS the water stress indicator, and `utilizable x (1 - WSI)` is his residual.
    #
    # 0.85 is supported by measurement. The Yellow River 1987 "八七分水" allocation is a
    # CONSUMPTION quota (its rows are headed 年耗水量, surface water only), so it compares
    # like-for-like with the consumption basis this constraint acts on. Against that 370x10^8
    # m3 quota, YRCC's 2024 bulletin (307.09 surface consumption) gives a measured non-power
    # share of 0.807-0.830 (0.776-0.786 once power is netted out). 0.85 sits just above that
    # range, i.e. slightly conservative.
    #
    # DISCLOSE IN METHODS: the anchoring, not the value, is the approximation. That measured
    # share is relative to China's own allowance of 52.6% of runoff, while the code multiplies
    # it by Richter's 20% -- and Richter is 2.63x stricter than Chinese water law. The result is
    # a deliberate intermediate between two internally consistent conventions, and it is also
    # why four northern basins breach the limit: a fully law-compliant northern basin still
    # exceeds Richter by ~2x.
    #
    # Precedent for constraining power-sector basin water in an optimisation without any
    # environmental-flow share at all: Zhang, He, Johnston & Zhong (2021) J. Clean. Prod.
    # 329:129765 (SWITCH-China). Power is ~8% of China's withdrawal but only ~1% of its
    # consumption (Zhang et al. 2017, JCLP 161:1171-1179).
    existing_withdrawal_share: float = 0.0
    # Which water budget the availability constraint is built from.
    #
    #   "runoff"          v9 and earlier. available = qtot x 0.20 x (1 - existing_withdrawal_
    #                     share). The two factors are ALIASED (see `_water_available_by_node`),
    #                     so nothing distinguishes the environmental-flow standard from the
    #                     allocation rule, and for the north China basins the denominator --
    #                     local natural runoff -- is smaller than actual use, which is financed
    #                     by inter-basin transfer and groundwater.
    #   "official_quota"  v9.1 onward. The two rules become two separate constraints on two
    #                     different bases, each binding the quantity it is actually written
    #                     about:
    #                        node  <= qtot x 0.20          environmental flow, on CONSUMPTION
    #                                                      (a depletion rule)
    #                        basin <= 用水总量控制指标 - 非电既有取水
    #                                                      allocation, on WITHDRAWAL
    #                                                      (what the 公报 meters)
    #                     `existing_withdrawal_share` is then unused: the allocation rule is
    #                     read off 国办发〔2013〕2号 instead of being assumed.
    #                     Basin caps come from `inputs/water_basin_caps.csv`
    #                     (`scripts/build_water_basin_caps.py`).
    water_budget: str = "runoff"
    # Basin cap on/off, honoured only under water_budget='official_quota'. Off leaves the
    # environmental-flow node limit alone, which is the CONTROL arm of the v9.1 design:
    #
    #   BASE                 no water rule at all
    #   *_oq_envonly         environmental flow only          (this flag False)
    #   *_oq                 environmental flow + allocation  (this flag True)
    #
    # The ladder is what the de-aliasing bought. Under 'runoff' the two rules are one product
    # and no experiment can separate them; here BASE->envonly prices the environmental-flow
    # standard and envonly->oq prices the allocation rule, each on its own.
    apply_basin_cap: bool = True
    # Apply the basin bias-correction factors when serving water to the solver. The factors
    # (`.basin_bias_factors` in builders/water.py, estimated from each model's `historical`
    # run against third-survey basin totals) are baked into `available_water_m3_per_year` in
    # water_availability.csv; `local_runoff_m3_per_year` holds the same number uncorrected.
    # Setting this False serves raw modelled runoff so the constraint's sensitivity to the
    # correction can be quantified (the "correction on/off" robustness check). Kept as an
    # assumption, not a rebuild switch, so the pair of runs reads the same input file.
    apply_bias_correction: bool = True
    # --- Wet-to-dry cooling conversion -------------------------------------------------
    # The single largest lever on the water account, and the one Chinese operators actually
    # pull when a river runs short: converting a recirculating condenser to direct air
    # cooling drops consumption from ~1.3-2.0 to ~0.12-0.20 m3/MWh. With capture attached the
    # gap is wider still (2.5-3.7 vs 0.25-0.36), which is why cooling choice moves the
    # m3/tCO2 figure ~5x while the retrofit pathway moves it ~1.6x. Holding cooling fixed
    # forces every water response through early retirement and overstates the cost of water
    # scarcity.
    #
    # Capex spans a wide range in the Chinese literature (roughly 200-400 CNY/kW for a
    # condenser and ACC island replacement on an existing unit); 300 is the midpoint and
    # carries a sensitivity.
    #
    # THE EFFICIENCY PENALTY IS THE DOMINANT COST OF DRY COOLING, NOT THE CAPEX. In the solved
    # runs the energy penalty term is 11-23x the air retrofit capex, so this number matters far
    # more than the 300 CNY/kW above. It is in percentage points of net efficiency: dry cooling
    # raises turbine backpressure and adds fan auxiliary load, so more coal and more CO2 are
    # burnt per MWh delivered.
    #
    # 2.0 pp, NOT the 1.5 pp used before. The old value was taken as the midpoint of a "1-2 pp"
    # range, but three independent sources put the real distribution in the upper half of that
    # range or past it, and 1.5 sits below every one of them:
    #
    #   * Same-parameter Chinese units, measured coal rates. 600 MW subcritical: 288 g/kWh wet
    #     vs 301 air (gross); 200 MW: 315 vs 333. Auxiliary power rises 8.2% -> 8.8%, so on the
    #     NET (supplied) basis the 600 MW pair is 313.7 -> 330.0 g/kWh. At 7000 kcal/kg standard
    #     coal that is 39.2% -> 37.2% net efficiency = 1.93 pp (200 MW: 2.11 pp).
    #   * Zhang, Anadon, Mo, Zhao & Liu (2014), "Water-Carbon Trade-off in China's Coal Power
    #     Industry", Environ. Sci. Technol. 48(19), 11082. China's ~130 GW of air-cooled units
    #     emitted 24.3-31.9 Mt CO2 MORE than wet-cooled equivalents in 2012. Spread over that
    #     fleet at 4500-5000 operating hours and an 820 g CO2/kWh baseline, that back-solves to
    #     1.9-2.8 pp.
    #   * Qin et al. (2023), "Global assessment of the carbon-water tradeoff of dry cooling for
    #     thermal power generation", Nature Water 1(8), 682-693. Unit-level global assessment:
    #     the dry-cooling energy and CO2 penalty is 1-15% of power output, location- and
    #     climate-specific. 2.0 pp = 4.8% more fuel sits inside that; the old 1.5 pp = 3.6% was
    #     at its lower edge.
    #
    # KNOWN SIMPLIFICATION, and it biases dry cooling to look cheap. The penalty here is a
    # constant. Qin et al. find it worsens FASTER than ambient temperature rises, and in northern
    # China the dry season and the hot season coincide -- so the periods where this understates
    # the penalty are exactly the periods where water is scarcest. `SA_air_penalty_high` (2.8 pp)
    # is the crude stand-in for that until the penalty is made temperature-dependent, which is a
    # model change rather than a parameter change.
    allow_air_cooling_retrofit: bool = True
    air_retrofit_capex_cny_per_kw: float = 300.0
    air_retrofit_efficiency_penalty_pp: float = 0.020
    air_retrofit_lifetime_years: int = 20
    # --- End-of-horizon salvage (author's decision 2026-09-22) ---
    # Every one-time capex in the objective (coal CCS/BECCS retrofit island, blend upgrades,
    # air-cooling retrofit, pipelines, site rebuilds, industrial capture and H2 routes) is
    # depreciated straight-line over the economic life below; whatever is undepreciated at the
    # END of the horizon (last planning year + its interval, 2070 on the standard grid) is
    # credited back, discounted from that year. Without it a 2060 retrofit paid its whole
    # capex for ten years of use and the model under-invested in the last period; with it the
    # capex actually borne inside the horizon is the depreciation over the years the asset
    # serves, which is the only accounting consistent with charging capex once. The stranded-
    # asset write-off is NOT salvaged (it is a loss, not an asset). Lives: coal capture
    # island 20 a (same as `INDUSTRY_CAPTURE_LIFETIME_YEARS`); blend-burner upgrades 20 a;
    # air-cooling 20 a (`air_retrofit_lifetime_years`, above); pipelines 30 a
    # (`pipeline_lifetime_years`); a rebuilt site 30 a. Set `end_of_horizon_salvage=False`
    # to reproduce results solved before 2026-09-22.
    end_of_horizon_salvage: bool = True
    ccs_retrofit_lifetime_years: int = 20
    blend_upgrade_lifetime_years: int = 20
    rebuild_lifetime_years: int = 30
    water_extraction_cost_cny_per_m3: float = 4.0       # Industrial water extraction cost
    water_transport_cost_cny_per_m3_km: float = 0.05     # Water pipeline/truck transport
    direct_fallback_capex_multiplier: float = 2.8
    branch_capex_multiplier: float = 1.35
    # New pipe laid in an existing pipeline corridor: the saving is the right-of-way (and
    # permitting time, unquantified). NETL 2013 (DOE/NETL-2013/1614, reproduced in IEAGHG
    # 2013/18 Table 20): ROW = 51 200 + 1.28 L (577 D + 29 788) USD, i.e. 2-3% of a 24-40 inch
    # line's capex over 100 miles. 0.97 = that ROW share removed. Was 0.4 (unsourced); a
    # German topology paper is reported to use a 10% corridor discount but could not be
    # opened (ScienceDirect S2772656826001004) -- 0.9 only after the author verifies it.
    corridor_capex_multiplier: float = 0.97
    top_k_storage_pairs: int = 5
    slack_penalty_cny_per_unit: float = 5_000_000_000.0
    sparse_interval_years: int = 10
    # Dynamic resource routing parameters
    resource_match_radius_km: float = 200.0
    # National ceiling on biomass burned in the coal fleet, GJ per year (16 EJ; author's call,
    # 2026-09-10). The 0.25-degree node layer sums to ~30 EJ/yr of collectable residue, but
    # that is a technical potential shared with every other biomass user; this cap is the
    # fleet-wide share the study allows. Applied per planning year on top of the node limits.
    # 0 or negative disables it.
    biomass_national_cap_gj_per_year: float = 16.0e9
    # Hub-level rebuild and blend-level decisions as CONTINUOUS shares (author's call,
    # 2026-09-10). A hub aggregates ~10 units, so "part of the hub is rebuilt" and "part of
    # the hub is converted to blend level l" are real degrees of freedom; as one-hot binaries
    # they made the relaxation so weak that the sector-cap MIP stalled at 4-15% gap after 10 h
    # (bound flat at the root LP). Pipes stay integer. False restores the binary form.
    hub_decisions_continuous: bool = True
    # National ceiling on green ammonia burned in the coal fleet, Mt NH3 per planning year
    # (2030/2040/2050/2060). The node layer is an electrolysis potential (~8 800 Mt NH3 in
    # 2030) that never binds; the literature review of 2026-09-10 (see
    # docs/工业联合减排实现说明.md §9.6) puts the AMMONIA AVAILABLE TO POWER at:
    #   2050  47 Mt  -- Xiong et al. 2022, 储能科学与技术 11(12), 掺氨发电渗透率 30%
    #                  (DOI 10.19799/j.cnki.2095-4239.2022.0364), directly citable;
    #   2030   2 Mt  -- demonstration scale: national green-ammonia CAPACITY 4.5 Mt in 2030
    #                  (中国化工节能技术协会 via 中国能源报 2025-09-01), fertiliser first;
    #   2040  12 Mt  -- INTERPOLATED between Xiong's 2035 co-firing demand (5.4 Mt) and 2050;
    #   2060  55 Mt  -- Xiong 2060 total ammonia 120 Mt at >97% renewable, roughly half of it
    #                  energy use (RMI/CPCIF 2024); midpoint of the 50-60 Mt range.
    # 2030/2040/2060 are derived, not quoted -- flagged for the author. Empty tuple = off.
    ammonia_fleet_cap_mt_by_year: tuple[float, ...] = (2.0, 12.0, 47.0, 55.0)
    # National ceiling on GREEN HYDROGEN drawn from the shared electrolysis nodes by every
    # user (fleet ammonia in H2 terms via NH3_H2_RATIO + industrial hydrogen), Mt H2 per
    # planning year. 2030 and 2060 from 中国氢能联盟《中国氢能技术发展路线图研究》(2024-12):
    # renewable hydrogen 3.5-6.5 Mt in 2030 and 75-162 Mt in 2060 (upper / mid taken);
    # 2040 and 2050 from 水电水利规划设计总院 (澎湃 2024, "据预测", secondary): 69 / 91 Mt.
    # Empty tuple = off.
    green_h2_national_cap_mt_by_year: tuple[float, ...] = (6.5, 69.0, 91.0, 120.0)
    # Blend ratio upgrade capital cost (CNY per MW of plant capacity per blend level step)
    biomass_upgrade_capex_cny_per_mw_per_level: float = 500_000.0
    ammonia_upgrade_capex_cny_per_mw_per_level: float = 25_000.0  # ≈125 CNY/kW at top level (50% blend).
    # China full-conversion literature: 90 CNY/kW (CNERI 2025, 100% conversion) + storage;
    # MIT supply system ≈163 CNY/kW (Deng et al. 2024); burner-only 121.6 CNY/kW (Li & Li 2022,
    # unconfirmed). Median ≈125 CNY/kW mapped to level 5; linear 25 CNY/kW per level.
    # (Previous 800,000 was ~30x above the China cluster.)
    # Biomass delivered cost components (Wang et al. 2024, Nat Commun)
    # 收购价 base_cost_cny_per_gj 在供给曲线 CSV 里（`constants.BIOMASS_COST_BASE` = 20 元/GJ）。
    biomass_pretreatment_cost_cny_per_gj: float = 11.96     # γ₅: 6.15 $/MWh × 7.0 / 3.6 (Wang et al. 2024, Nat Commun)
    biomass_transport_fixed_cost_cny_per_gj: float = 13.13    # γ₆: 6.75 $/MWh × 7.0 / 3.6
    biomass_transport_variable_cost_cny_per_gj_km: float = 0.126  # γ₇: 0.065 $/(MWh·km) × 7.0 / 3.6
    # Ammonia transport cost (truck, Hydrogen Council & McKinsey 2022; IEA GHR 2023)
    # 0.12 USD/(t·km) = 0.00012 USD/(kg·km) × 7.0 = 0.00084 CNY/(kg·km)
    ammonia_transport_cost_cny_per_kg_km: float = 0.00084
    # Runtime grid coarsening (degrees; 0 = no coarsening, data already coarsened on disk)
    biomass_coarse_grid_degrees: float = 0.0
    ammonia_coarse_grid_degrees: float = 0.0
    water_coarse_grid_degrees: float = 0.0

    # Province-specific annual operating hours (replace uniform capacity_factor for generation)
    # Source: China Electricity Council statistics, average coal power utilization hours by province
    province_operating_hours: dict[str, float] = field(default_factory=lambda: {
        "Inner Mongolia": 5031.5, "Zhejiang": 5535.3, "Xinjiang": 5281.2,
        "Jiangsu": 5015.5, "Fujian": 5126.2, "Anhui": 5082.1,
        "Hainan": 5030.9, "Ningxia": 4970.9, "Shanghai": 4954.8,
        "Sichuan": 4996.4, "Chongqing": 4794.2, "Guangdong": 4831.7,
        "Guangxi": 4786.6, "Jiangxi": 4766.4, "Guizhou": 4747.1,
        "Shanxi": 4629.5, "Shaanxi": 4538.0, "Yunnan": 4482.2,
        "Hebei": 4455.1, "Shandong": 4361.1, "Tianjin": 4347.5,
        "Gansu": 4189.6, "Qinghai": 4097.0, "Hunan": 3906.9,
        "Hubei": 3749.6, "Jilin": 3621.4, "Henan": 3599.6,
        "Heilongjiang": 3478.5, "Liaoning": 3100.7, "Beijing": 779.2,
    })

    def province_cf(self, province_name: str) -> float:
        """Get capacity factor for a province from operating hours."""
        hours = self.province_operating_hours.get(province_name, self.capacity_factor * 8760.0)
        return hours / 8760.0

    def fixed_cost_cny_per_mwh(self, pathway: str) -> float:
        return {
            "unabated": 0.0,
            "retire": self.retire_cost_cny_per_mwh,
            "ccs": self.ccs_fixed_cost_cny_per_mwh,
            "biomass": self.biomass_fixed_cost_cny_per_mwh,
            "beccs": self.beccs_fixed_cost_cny_per_mwh,
            "ammonia": self.ammonia_fixed_cost_cny_per_mwh,
        }[pathway]

    def ccs_learning_factor(self, year: int) -> float:
        """Exogenous cost reduction factor from Wright's law learning curve."""
        n_doublings = max(0, year - self.ccs_learning_reference_year) / self.ccs_deployment_doubling_years
        return (1.0 - self.ccs_learning_rate) ** n_doublings

    def ccs_energy_penalty_ratio(self, year: int) -> float:
        """Extra fuel per unit of output for a CCS/BECCS plant in `year` (dimensionless).

        Values are tabulated against `constants.PLANNING_YEARS`; off-grid years take the
        nearest tabulated point, matching how carbon and electricity prices are looked up.
        """
        ratios = self.ccs_energy_penalty_ratio_by_year
        if not ratios:
            return 0.0
        mapping = dict(zip(PLANNING_YEARS, ratios, strict=False))
        if year in mapping:
            return float(mapping[year])
        nearest = min(mapping, key=lambda candidate: abs(candidate - year))
        return float(mapping[nearest])

    def _fraction_for_year(self, values: tuple[float, ...], year: int) -> float:
        if not values:
            return 1.0
        mapping = dict(zip(PLANNING_YEARS, values, strict=False))
        if year in mapping:
            return float(mapping[year])
        nearest = min(mapping, key=lambda candidate: abs(candidate - year))
        return float(mapping[nearest])

    def storage_deployment_fraction(self, year: int) -> float:
        """Share of the 2060-scale injection rate that is built and available in `year`."""
        return self._fraction_for_year(self.storage_deployment_fraction_by_year, year)

    def ammonia_supply_deployment_fraction(self, year: int) -> float:
        """Share of the green-ammonia technical potential that is built in `year`."""
        return self._fraction_for_year(self.ammonia_supply_deployment_fraction_by_year, year)

    def ammonia_fleet_cap_mt(self, year: int) -> float:
        """National green-ammonia ceiling for coal co-firing in `year`, Mt NH3 (0 = off)."""
        if not self.ammonia_fleet_cap_mt_by_year:
            return 0.0
        return self._fraction_for_year(self.ammonia_fleet_cap_mt_by_year, year)

    def green_h2_national_cap_mt(self, year: int) -> float:
        """National green-hydrogen ceiling on all node users in `year`, Mt H2 (0 = off)."""
        if not self.green_h2_national_cap_mt_by_year:
            return 0.0
        return self._fraction_for_year(self.green_h2_national_cap_mt_by_year, year)

    def cooling_baseline_water_intensity(self, cooling_label: str) -> float:
        label = str(cooling_label).strip().lower()
        if "once" in label:
            return self.cooling_once_through_water_intensity_m3_per_mwh
        if "air" in label or "dry" in label:
            return self.cooling_air_water_intensity_m3_per_mwh
        return self.cooling_recirculating_water_intensity_m3_per_mwh


@dataclass(frozen=True)
class OptimizationScenario:
    experiment_id: str
    description: str
    planning_years: tuple[int, ...] = field(default_factory=lambda: tuple(PLANNING_YEARS))
    capture_rate: float = 0.90
    biomass_blend_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    ammonia_blend_levels: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    # 部门碳目标来源：读 `inputs/sector_targets_<source>.csv`（scripts/build_sector_targets.py）。
    # 每组每个规划年一条上限：residual_g(y) <= cap_fraction_g(y) x baseline_g(2030) + shortfall_g(y)，
    # g 属于 {power, steel, cement, chemicals}；baseline_g(2030) 是本模型自身的 2030 冻结技术排放，
    # 即 TIMES 轨迹给形状、模型给水平。煤电整体为 power 组。
    sector_target_source: str = "times_cn60"
    # Coal fleet utilisation by planning year, national capacity-weighted hours. Empty keeps
    # the province statistics frozen across all four years (the pre-2026-09-10 behaviour, in
    # which the fleet generated 6 576 TWh in 2060 as in 2030). When set, every hub's province
    # hours are scaled by hours_y / fleet-average current hours (4 643 h), so provincial
    # differences are kept and only the level moves. The author's instruction (2026-09-10) was
    # a rough utilisation trajectory rather than a dispatch model; TIMES CN60 gives 3 594 h in
    # 2030 and 3 092 h in 2040 (coal is a residual after 2040 there), the later points are
    # round numbers for a fleet kept for flexibility.
    coal_operating_hours_by_year: tuple[float, ...] = ()
    # Exogenous industrial OUTPUT index by sector and year from
    # `inputs/industry_output_index_<source>.csv`; empty means output is held at its 2025
    # level in every year (the pre-2026-09-10 behaviour). Defaults to `sector_target_source`
    # when that is set and this is not, so a TIMES-anchored cap always comes with the TIMES
    # output path it was derived under.
    industry_output_index_source: str = ""
    storage_scope: str = "dsa_eor"
    injectivity_multiplier: float = 1.0
    biomass_supply_multiplier: float = 1.0
    biomass_cost_multiplier: float = 1.0
    ccs_cost_multiplier: float = 1.0
    # 工业捕集成本乘子：乘捕集 capex（固定运维随之）与每吨捕集的能耗、耗材成本
    # （`industry_matrices.industry_year_data`）。2026-09-22 起不再乘 ACCA21 平准化成本。
    industry_cost_multiplier: float = 1.0
    # Industrial H2-route premium. Separate from the multiplier above because the H2 anchor
    # carries a KNOWN downward bias -- it is a greenfield-vs-greenfield comparison applied to
    # existing plants whose incumbent capital is sunk -- so its uptake is an upper bound and
    # needs its own handle. See optimization/industry.py, KNOWN BIASES.
    industry_h2_cost_multiplier: float = 1.0
    ammonia_cost_multiplier: float = 1.0
    ammonia_transport_adder_usd_per_kg: float = 0.0
    water_mode: str = "no_water"
    # Climate member driving water availability, e.g. "cwatm|gfdl-esm4|ssp370".
    # Empty selects the first member of the family implied by water_mode.
    water_scenario_id: str = ""
    # There is no longer a basis switch: the availability constraint always acts on
    # consumption, the tariff always on the Chinese abstraction quota, and withdrawal is
    # reported but never constrained. See `data_prep._prepare_plants` for why constraining
    # withdrawal against an environmental-flow allowance is a category error.
    # "annual" uses the decadal mean; "dry" uses the lowest three consecutive months,
    # which is when thermal plants actually get curtailed.
    water_season: str = "annual"
    water_multiplier: float = 1.0
    # Parametric charge added to every m3 of water delivered to a plant, on top of the
    # extraction and conveyance costs already in water_supply_links.csv. Sweeping it traces
    # the water-carbon frontier: because the model is a MIP, Gurobi cannot return reliable
    # duals for the availability constraint, so the shadow price of water is recovered by
    # parametric pricing instead - the adder at each point IS the shadow price there.
    water_price_adder_cny_per_m3: float = 0.0
    forced_cooling_technology: str = ""
    ccs_water_multiplier_adjustment: float = 1.0
    beccs_water_multiplier_adjustment: float = 1.0
    dense_time_grid: bool = False
    carry_state_between_years: bool = True
    pathway_disable: tuple[str, ...] = ()
    forced_pathways: tuple[str, ...] = ()
    min_forced_path_share: float = 0.0
    corridor_prior_strength: float = 0.70
    solve_mode: str = "joint"           # only "joint" is implemented; other values raise
    mip_gap: float = 0.01               # MIP optimality gap (1% default; relax for exploratory runs)
    solver_threads: int = 0       # 0 = Gurobi auto-detect
    solver_time_limit: int = 36000  # seconds (default 10h)
    discount_rate: float = 0.06
    discount_base_year: int = 2025
    # Net system cost parameters
    carbon_price_cny_per_t_by_year: tuple[float, ...] = (120.0, 500.0, 880.0, 1260.0)
    electricity_price_cny_per_mwh_by_year: tuple[float, ...] = (400.0, 440.0, 490.0, 550.0)
    max_new_retirement_share_per_period: float = 0.15
    # Retrofit CF boost: retrofitted plants (CCS/biomass/BECCS/ammonia) can increase
    # generation by this factor vs baseline (e.g., 1.15 = +15% due to priority dispatch)
    retrofit_cf_boost: float = 1.15
    # Site rebuild parameters: expired plants can rebuild at 70% of new-build cost
    rebuild_capex_fraction: float = 0.70  # 70% of stranded_asset_base_cny_per_kw
    rebuild_efficiency: float = 0.45  # USC efficiency for rebuilt plant (vs 0.42 baseline)
    notes: str = ""

    def _interpolate_year_tuple(self, values: tuple[float, ...], year: int) -> float:
        mapping = dict(zip(self.planning_years, values, strict=False))
        if year in mapping:
            return float(mapping[year])
        if not mapping:
            return float(values[0]) if values else 0.0
        nearest = min(mapping, key=lambda y: abs(y - year))
        return float(mapping[nearest])

    def carbon_price_for_year(self, year: int) -> float:
        return self._interpolate_year_tuple(self.carbon_price_cny_per_t_by_year, year)

    @property
    def effective_output_index_source(self) -> str:
        chosen = str(self.industry_output_index_source).strip()
        return chosen or str(self.sector_target_source).strip()

    def operating_hours_scale(self, year: int, fleet_hours_now: float) -> float:
        """Multiplier on every hub's current province hours in `year` (1.0 when unset)."""
        if not self.coal_operating_hours_by_year:
            return 1.0
        if fleet_hours_now <= 0:
            raise ValueError("fleet_hours_now must be positive to scale operating hours")
        target = self._interpolate_year_tuple(self.coal_operating_hours_by_year, year)
        return float(target) / float(fleet_hours_now)

    def electricity_price_for_year(self, year: int) -> float:
        return self._interpolate_year_tuple(self.electricity_price_cny_per_mwh_by_year, year)

    def effective_years(self, available_years: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if not self.dense_time_grid:
            return tuple(self.planning_years)
        filtered = [year for year in sorted(set(available_years)) if year <= max(self.planning_years)]
        return tuple(filtered or self.planning_years)

    def interval_years(self, years: tuple[int, ...], index: int, assumptions: OptimizationAssumptions) -> int:
        if len(years) <= 1:
            return assumptions.sparse_interval_years
        if index < len(years) - 1:
            return max(1, years[index + 1] - years[index])
        return max(1, years[index] - years[index - 1])

    def path_enabled(self, pathway: str) -> bool:
        return pathway not in set(self.pathway_disable)
