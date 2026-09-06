# Sourcing `existing_withdrawal_share` (0.85)

Decision memo. Evidence-only; every number below carries its source. Anything I could not
read is marked NOT VERIFIED rather than inferred.

## 0. The decision

**Declare 0.85 as a sourced assumption with a strong defence, and fix its documentation rather
than its value. No re-run required.**

The finding that settles it (§9c): China's 八七分水 Yellow River allocation is a **consumption**
quota (rows headed 「年耗水量」, surface water only), so it compares like-for-like with measured
consumption. The Yellow River's measured non-power consumptive utilization of that quota is
**0.807-0.830** (0.776-0.786 power-netted, YRCC 2024 bulletin). **0.85 sits just above that
range — the value was never far wrong.**

What *is* wrong is the anchoring. 0.85 measures utilization of China's official **64%**-of-runoff
allowance, but the code multiplies it by Richter's **20%** allowance:

| anchoring | allowance | measured share | power gets (fraction of runoff) |
|---|---|---|---|
| (A) pure Richter | 0.20 x runoff | 2.185 | **0** — degenerate |
| (B) pure official quota | 370 (63.8% of runoff) | 0.830 | **10.9%** |
| (C) current code | 0.20 x runoff | 0.85 (a B-magnitude share) | **3.0%** |

The code lands between the two internally consistent answers. That is defensible as a deliberate
intermediate choice — but it must be *stated*, because Richter's 20% is **2.6x stricter than
China's own water law**, which is also the root cause of the four degenerate northern basins in §8.

Five things are now sourced and go into Methods (details in §10):

1. **The formula is Richter's own prescription** — he writes that "20% of the natural monthly mean
   flow can be allocated for consumptive use" and that withdrawals must be evaluated "when added
   to already-existing water uses"; his Table II is headed "Cumulative allowable depletion" (§9e).
2. **The parameter has a name**: Smakhtin et al. 2004 Eq.(1)'s WSI; `utilizable x (1 - WSI)` is his
   residual — our algebra exactly. 0.85 falls in his "heavily exploited / environmentally water
   stressed" band (§9e).
3. **The value is empirically anchored** to the YRCC measurement above (§9c).
4. **An optimization precedent exists**: Zhang, He, Johnston & Zhong 2021 JCLP 329:129765 caps
   power-sector basin water from base-year use with no environmental-flow share, and reports
   sensitivity rather than a point value (§9d).
5. **The residual-allocation national value is 0.559** from 表9 plus the bulletin's own 耗水率
   (§8) — keep it as a reported figure and sensitivity endpoint, not as the headline.

Drop the "~15% of national abstraction" justification: it rests on 《中国环境统计年鉴》, which is
not on disk and could not be verified, and it is now superseded by a directly measured number.

**Earlier draft recommendation "switch the headline to 0.56" is WITHDRAWN** — see §10 for why.

### Two allocation philosophies, both now with precedent

| | **Residual allocation** | **Status-quo / quota anchoring** |
|---|---|---|
| Rule | deduct other sectors' actual depletion; power gets the remainder | allowance is the official/observed quota; share is its measured utilization |
| Value | **0.559** national; >1.0 in 海河/黄河/淮河/西北 | **0.807-0.830** measured (Yellow River) |
| Precedent | Hoekstra et al. 2012; Jin et al. 2022; **Richter 2012 himself** (§2, §4, §9e) | **Zhang, He, Johnston & Zhong 2021** JCLP 329:129765 (§9d) |
| Failure mode | degenerate in 4 northern basins (764.7 GW zeroed) | quotas published for very few basins (§9) |

Both are internally coherent and both are now sourced. `data_prep.py:439` implements the residual
*form*; the 0.85 *value* comes from quota anchoring. The recommendation is to keep that hybrid but
declare it, with both consistent bounds reported — see §9c and §10.

## 1. The basis mismatch is not a mismatch — Richter is a depletion standard

`data_prep.py:137-142` records that withdrawal was rejected as a constraint basis on
empirical grounds. The source text supports that choice directly, which is worth stating in
the paper because it converts an apparent weakness into a correctness argument.
`plan/environmental_flow_sourcing.md` already reached the same conclusion and, unlike my
search attempts, **read the Richter PDF body verbatim**:

> "When a single threshold value or standard is needed, such as for corporate risk screening
> or water supply planning purposes, we suggest that protecting 80% of daily flows will
> maintain ecological integrity in most rivers."

Tiers from the same paper: <=10% daily flow alteration = high protection; 11-20% = moderate;
>20% causes "moderate to major changes in natural structure and ecosystem functions"; 90%
retention advised for endangered-species / high-biodiversity rivers. (Richter BD, Davis MM,
Apse C, Konrad C. *A presumptive standard for environmental flow protection.* River Res.
Applic. 2012;28(8):1312-1321, doi:10.1002/rra.1511.)

Two consequences:

1. Richter's standard is about **flow alteration / depletion**, so consumption is the
   *correct* basis for a 20% allowance and withdrawal is not. Hoekstra et al. 2012 (§2)
   operationalizes it on exactly that basis, which settles the question.
2. **0.20 is Richter's moderate tier, not his strict one.** We are at the permissive end of
   his two-tier scheme, which `plan/environmental_flow_sourcing.md` already recommends
   acknowledging while noting that 0.20 is still stricter than Smakhtin et al. 2004's
   20-50%-of-MAF convention.

## 2. Hoekstra et al. (2012) — our formula, already published

Hoekstra AY, Mekonnen MM, Chapagain AK, Mathews RE, Richter BD. *Global monthly water
scarcity: blue water footprints versus blue water availability.* PLoS ONE 2012;7(2):e32688.
Fetched and read (journals.plos.org).

- Availability: "Blue water availability is estimated by reducing total natural runoff by
  80% to account for presumed environmental flow requirements" — i.e. **0.20 x natural
  runoff**, described as "the volume of water that can be **consumed** without expected
  adverse ecological impacts". Cited to Richter et al. (their ref 22).
- Rationale quoted for the threshold: "depletion beyond 20% of a river's natural flow
  increases risks to ecological health and ecosystem services."
- Numerator is **consumption, explicitly not withdrawal**: they "measure water use in terms
  of consumptive use of ground- and surface water flows - i.e., the blue water footprint -
  rather than water withdrawals", justified because ~40% of agricultural and 90-95% of
  industrial/domestic withdrawals return to the system.
- Blue water scarcity = total blue water footprint / blue water availability, per month per
  basin. Classes: **<100% low; 100-150% moderate; 150-200% significant; >200% severe**;
  100% means the available blue water is entirely consumed and the EFR is violated.
- Their assumed consumed fractions: **5% of industrial and 10% of domestic withdrawals**
  (note: far lower than the China-specific factors in Section 3 — a real uncertainty range).

This is the single best citation for the model's water constraint as a whole. It supplies
the 0.20, the consumption basis, *and* the interpretation of `existing_withdrawal_share` as
a scarcity ratio rather than an arbitrary discount.

## 3. The number, from Chinese official data — two independent routes agree

### Route A: official national consumption total
`reference/water/china_water_bulletin_2016.pdf` (2016 年中国水资源公报, MWR), p.2-3:

- 水资源总量 **32 466.4 亿m3** (a wet year: "比多年平均偏多17.1%")
- 用水总量 **6 040.2 亿m3**, split 生活 821.6 (13.6%) / 工业 1 308.0 (21.6%) / 农业
  3 768.0 (62.4%) / 人工生态环境补水 142.6 (2.4%)
- **耗水总量 3 192.9 亿m3，耗水率 52.9%** — an *official consumption* figure, p.3
- Aside, p.2: 海水直接利用量 887.1 亿m3, "主要作为火（核）电的冷却用水" — confirms coastal
  thermal cooling runs on seawater, consistent with the model zeroing coastal condensers.

Long-term-mean resources = 32 466.4 / 1.171 = **27 725 亿m3**.
Consumption / mean resources = 3 192.9 / 27 725 = **11.5%** of renewable water.
=> implied share = 0.115 / 0.20 = **0.58**

### Route B: per-basin resources x China-specific consumption factors
`reference/water/wangGQ2025_adv_water_sci_36_948.pdf` (王国庆等, 水科学进展 2025;36(6):948,
"2015—2024 年中国水资源和用水结构的时空演变及成因"), Table 1 p.3 (951) — per-region 水资源量
from 第三次全国水资源调查评价, 亿m3:

| 一级区 | 1956-2016 | 2015-2024 |
|---|---|---|
| 松花江 Songhua | 1 469.2 | 1 972.1 |
| 辽河 Liao | 483.4 | 463.1 |
| 海河 Hai | 327.6 | 353.5 |
| 黄河 Yellow | 702.8 | 753.1 |
| 淮河 Huai | 928.3 | 953.4 |
| 西北诸河 NW rivers | 1 310.1 | 1 413.1 |
| 东南诸河 SE rivers | 2 694.5 | 2 090.2 |
| 西南诸河 SW rivers | 5 753.8 | 5 528.5 |
| 长江 Yangtze | 9 871.2 | 10 451.5 |
| 珠江 Pearl | 4 758.6 | 4 908.6 |

Internal check: north six sum = 5 221.4 (paper says 5 221); south four = 23 078.1 (says
23 078); total 28 299.5 (says 28 299). **Table verified self-consistent.**

Water *use* from the same paper, p.5 (953): 2024 全国用水总量 **5 928 亿m3** (农业 61.5%,
工业 16.4%, 生活 15.6%, 生态 6.4%); 北方六区 **2 703 亿m3** (农业 71.7%, 生活 11.7%,
工业 7.8%, 生态 8.7%); 南方四区 **3 225 亿m3** (农业 53.0%, 工业 23.5%, 生活 18.9%,
生态 4.5%). NOTE: per-basin *use* appears only as trend maps (Fig. 2, Z and beta), **not as
absolute per-basin values** — only the north-six/south-four aggregates are given.

Consumption-to-withdrawal ratios, China-specific, from Jin et al. 2022 (Section 4):
**agriculture 0.65, industry 0.23, domestic 0.40**.

| | use 亿m3 | resources 亿m3 | withdrawal/res | consumption/res | implied share |
|---|---|---|---|---|---|
| National | 5 928 | 28 299 | 0.209 | 0.105 (0.118 incl. 生态) | **0.52 (0.59)** |
| North six | 2 703 | 5 221 | 0.518 | 0.275 (0.320) | **1.37 (1.60)** |
| South four | 3 225 | 23 078 | 0.140 | 0.066 (0.073) | **0.33 (0.36)** |

**Cross-validation:** north+south consumption from Jin's factors = 2 963.9 亿m3 on 5 928
total use = a 50.0% national consumption rate, against the official 52.9% 耗水率 in Route A.
The two routes agree to within 3 percentage points, and Routes A and B give implied shares of
0.58 and 0.52-0.59. **~0.55 is the defensible national value.**

### Two hard findings from this arithmetic

1. **China's actual withdrawal already exceeds the Richter allowance.** National
   withdrawal/resources = 0.209 > 0.20. On a *withdrawal* basis there is literally nothing
   left for power anywhere. This is independent confirmation that withdrawal cannot be the
   constraint basis — the same conclusion `data_prep.py:137-142` reached empirically.
2. **A single national scalar is structurally wrong.** In the northern six regions non-power
   consumption is 1.37-1.60x the entire 20% allowance — i.e. those basins are already in
   Hoekstra's "moderate/significant scarcity" class (>100%) *before any power demand*, which
   matches the published record for the Hai and Yellow basins. In the south the share is
   ~0.33. Using 0.85 everywhere is simultaneously far too loose in the north and roughly
   2.5x too tight in the south.

## 4. Jin et al. 2022 — the closest comparator, and it deducts other sectors explicitly

Jin Y, Scherer L, Sutanudjaja EH, et al. *Climate change and CCS increase the water
vulnerability of China's thermoelectric power fleet.* Energy 2022;245:123339.
`reference/water/jin2022_energy_123339.pdf`, read in full.

**1. How much water may the power sector use** (Sec. 2.2, p.3): "Monthly available surface
water (WA) at a spatial resolution of 5-arcmin was calculated as the difference between
monthly river discharge and the environmental flow requirement." Discharge from
PCR-GLOBWB-2. Verbatim equations (Sec. 2.3, p.3, Eqs. 3-4):

```
q = KW * t * WW
P = min(Q - NEW, q) * 1/(t * WW)
```

"q = monthly required water withdrawal (m3); KW = installed capacity of CPU (MW); t = The
number of hours in each month (h); WW = water withdrawal factor (m3/MWh); P = useable
capacity of CPUs (MW); Q = monthly river discharge (m3); **NEW = water consumption of
non-electricity sectors (m3)**."

`NEW` is the direct analogue of our `existing_withdrawal_share` — but Jin computes it as a
**volume from data**, not a guessed fraction. This is the precedent for the recommendation in
Section 0. (AMBIGUITY, flagged honestly: Eq. 4 defines `Q` as "monthly river discharge"
while the surrounding text defines availability as discharge minus environmental flow. The
paper does not make explicit whether Eq. 4's `Q` is raw discharge or `WA`. Do not cite Jin
for a precise numeric allowance without resolving this.)

**2. Other sectors deducted?** Yes, on a **consumption** basis. "Upstream water consumption
and reduced availability for downstream uses were accounted for by considering all water uses
(irrigation, livestock, households, and industry). The water use for thermoelectric cooling
of power plants is not included in PCR-GLOBWB-2." "Water consumption was assessed by
multiplying the withdrawal and the corresponding China-specific factors (sector-specific
consumption-to-withdrawal ratios). Factors for agricultural, industrial and domestic sectors
are **0.65, 0.23 and 0.40**, respectively." Surface-water share of consumption taken from MWR
provincial data. Sources: [55] MWR *Water Resources Bulletin of China 2020*; [56] Flörke,
Global Environ Change 2013;23:144-156.

**3. Environmental-flow rule — a different number from ours.** "Environmental flow is defined
as the minimum freshwater flow required to sustain ecosystem functions [11]. For the rivers
that supply water for human use in China, **60% of the average discharge needs to be
preserved for environmental flow** [54]." So Jin retains 60% and leaves **40% extractable**,
versus our 20%. [54] = 韩琦, 谭光明, 傅旭, 杨浩, 李新 *Calculation methods of river
environmental flow and their applications*, Eng J Wuhan Univ (武汉大学学报·工学版)
2018;51:189-197. **Our 0.20 is 2x stricter than the China-specific figure Jin adopts** — a
useful robustness point, and a candidate sensitivity bound.

**4. Constraint or indicator?** Hard cap on useable capacity, not an optimization constraint;
the paper is a vulnerability assessment. Scarcity flag (Eq. 2, cited to [58] Rosa et al.,
*Global agricultural economic water scarcity*, Sci Adv 2020;6:eaaz6031):
`WS = WC/WA > 1` — "CPUs are located in water-scarce areas if the ratio between water
consumption (WC) and available water (WA) is > 1 (after the removal of environmental flow
requirements and for renewable water availability only)". **Consumption over EF-adjusted
availability, threshold 1.0** — the same construction as ours and as Hoekstra's 100% class.

**5. Reusable numbers:** the 0.65/0.23/0.40 consumption factors (used in Section 3); the 60%
EF retention; the WC/WA>1 criterion. Also cited: [11] Rosa L, Reimer JA, Went MS, D'Odorico
P. *Hydrological limits to carbon capture and storage.* Nat Sustain 2020 — and Jin notes
(p.3) that it "showed little sensitivity of water scarcity to different environmental flow
requirements". If verified in the original, that is a citable defence for the *EF fraction*
choice. I could NOT read Rosa 2020's Methods (eScholarship returned empty HTML then 403;
Nature paywalled). **NOT VERIFIED — highest-value remaining gap.**

## 5. Qin et al. 2023, Nature Water — what the target journal accepted

Qin Y, Wang Y, Li S, Deng H, Wanders N, Bosmans J, Huang L, Hong C, Byers E, Gingerich D,
Bielicki JM, He G. *Global assessment of the carbon-water tradeoff of dry cooling for thermal
power generation.* `reference/water/qin2023_natwater_osti2000316.pdf` (accepted manuscript,
28 pp.), read in full.

**Water is a SCREENING INDICATOR, not a constraint, and there is no optimization.** Methods
p.14-15, Eq. 7:

```
WSI = W / Q
```

"W refers to PCR-GLOBWB 2 simulated total water withdrawal: adding agricultural, industry and
domestic water withdrawal; Q refers to PCR-GLOBWB 2 simulated total runoff."

- Grid-level (0.5 deg), historical 1996-2005, three ISIMIP2b GCMs.
- Classes: "WSI<0.1 indicates low water scarcity, 0.2>WSI>=0.1 indicates moderate water
  scarcity, and 0.4>=WSI>=0.2 indicates medium water scarcity", focus on **WSI>0.4** as high.
- Threshold sourced to ref 48 = **Oki T & Kanae S,** *Global hydrological cycles and world
  water resources*, Science 2006;313; and in the main text (p.~10) to ref 32 = **Qin Y,**
  *Global competing water uses for food and energy*, Environ Res Lett, and ref 33 =
  **Vörösmarty CJ, Green P, Salisbury J, Lammers RB,** Science 2000.
- **Environmental flow is never deducted.** It is folded into the threshold: "As not all
  freshwater can be used by human population (e.g., environmental flow needs), WSI>0.4 is
  widely considered to be a reasonable, even though not definitive threshold value."
- Withdrawal basis, and power is *excluded* from the numerator: "As PCR-GLOBWB 2 model does
  not include thermoelectric cooling water withdrawal for power generation, our estimated
  water scarcity index is conservative."
- Usage: units on freshwater cooling with WSI>0.4 are the candidate set for dry cooling. No
  water balance is ever closed and no allocation to the power sector is ever computed.

### What a Nature Water reviewer will therefore expect

Based on Qin 2023, the journal's accepted convention is the **weakest** of the options: a
withdrawal-to-runoff ratio used as an exposure screen, with environmental flow implicit in a
0.4 threshold and the power sector excluded from the numerator. **Our treatment is strictly
more rigorous than the paper the journal already published** — we deduct environmental flow
explicitly, use the consumption basis that Richter's standard actually calls for, and close a
basin water balance inside an optimization.

The reviewer risk is therefore *not* "your method is non-standard". It is: **"you have
introduced a tighter-than-standard constraint and the number that sets its severity is
unsourced."** The fix is to source the number, not to weaken the method. Two framing moves
follow directly:

1. Present the constraint as Hoekstra et al. (2012) blue water scarcity with the power sector
   separated out — a named, cited quantity (Section 2).
2. Report the national implied value (~0.55) from the Water Resources Bulletin so the
   reviewer can verify it, and state 0.85 (if retained) as deliberately conservative
   *relative to that*, with the north/south spread as the sensitivity envelope (Section 3).

## 6. Wang F et al. 2023 (Water 15:1167) — water is ex-post, both bases reported

`reference/water/wangF2023_water15_1167.pdf`, read. China CCS retrofit siting, 234 cities.

- The optimization constraints are (Sec. 2, pp.~337-386): mass conservation, capture/storage
  capacity, CO2 reduction target, non-negativity. **There is no water constraint.**
- Water enters only afterwards: "the water stress ... is calculated" as the final step.
- Reports **both bases against availability**: "The ratio of freshwater withdrawal to
  available water (WTA) and the ratio of freshwater consumption to available water (CTA)".
  Direct precedent for a consumption-to-availability ratio.
- Thresholds are the WRI Aqueduct six-class scheme: "low (<0.1), low to medium (0.1-0.2),
  medium to high (0.2-0.4), high (0.4-0.8), extremely high (>0.8), and arid and low water
  use."
- Reusable results: CCS retrofit raises city-level WTA by 0.03 and CTA by 0.02 on average
  (+0.2 / +0.06 under the 2 C constraint); withdrawal +~6.5 Gt/yr and consumption +~2 Gt/yr.
- Cites at catchment level: Zhang C, Zhong L, Fu X, Wang J, Wu Z, *Revealing water stress by
  the thermal power industry in China* (their ref 16 for the WTA/CTA-at-catchment approach).

## 7. Negative result: the CCS-network literature we build on ignores water entirely

Full-text keyword scan of the four CCS optimization papers in `reference/` for "water
availability / water scarcity / water stress / environmental flow / water constraint /
water resource":

| Paper | Hits |
|---|---|
| Fan et al. 2023, *Co-firing plants with retrofitted carbon capture* | **0** |
| Wang & Cai 2024, Nat Commun 55332 (main text) | **1**, reference-list only |
| An et al. 2025, Nat Commun 57559 (main text) | **0** |
| Tang et al. 2023, *China's multi-sector-shared CCUS networks* | **0** |

None of them constrains, or even reports, water. The single hit is Wang & Cai's ref 32,
**Li H et al., *Catchment-level water stress risk of coal power transition*** — worth
retrieving as a China-specific catchment-scale comparator (NOT YET READ).

This is a positioning asset: the water constraint is a genuine contribution relative to the
CCS-siting literature, which is also why its parameterization will attract scrutiny.

## 8. Engaging the project's own prior work (`plan/t1b_per_basin_shares.md`)

T1b already derived per-basin shares on the **withdrawal** basis, found 5 of 9 basins
degenerate (share >= 1, 861.4 GW zeroed), correctly diagnosed the cause as a
numerator/denominator basis and supply mismatch (§4), and recommended keeping uniform 0.85.
`plan/data_basin_sectoral_water.md` §4 flagged the blocking gap: per-basin **consumption** is
not published, so "applying the national sectoral 耗水率 to each basin would be an
assumption, not data — flagged, not done."

**That gap is closable, and the result changes one conclusion but not the main one.**
Applying sectoral consumption rates is not a bare assumption — it is the published method of
Jin et al. 2022 (Section 4), and its China-specific factors are independently reproduced by
the bulletin's own national sectoral 耗水率:

| sector | Jin et al. 2022 | 《2025年中国水资源公报》 | diff |
|---|---|---|---|
| agriculture | 0.65 | 0.669 | -0.019 |
| industry | 0.23 | 0.239 | -0.009 |
| domestic | 0.40 | 0.383 | +0.017 |

Agreement within 2 pp on all three sectors. Using the official rates (ag 0.669, ind 0.239,
dom 0.383, eco 0.593) on the per-basin sectoral withdrawals of 表9, with once-through
thermal subtracted from industry first exactly as T1b does:

| code | 一级区 | W_share (T1b) | **C_share** | 1 - C_share | cons/runoff |
|---|---|---|---|---|---|
| A | 松辽 | 1.544 | **0.948** | 0.052 | 19.0% |
| C | 海河 | 5.772 | **3.190** | -2.190 | 63.8% |
| D | 黄河 | 2.787 | **1.569** | -0.569 | 31.4% |
| E | 淮河 | 3.307 | **1.889** | -0.889 | 37.8% |
| F | 长江 | 0.838 | **0.469** | 0.531 | 9.4% |
| G | 东南诸河 | 0.519 | **0.268** | 0.732 | 5.4% |
| H | 珠江 | 0.774 | **0.430** | 0.570 | 8.6% |
| J | 西南诸河 | 0.090 | **0.055** | 0.945 | 1.1% |
| K | 西北诸河 | 2.782 | **1.791** | -0.791 | 35.8% |
| — | **national** | 0.970 | **0.559** | 0.441 | 11.2% |

The consumption basis rescues **松辽 only** (1.544 -> 0.948, recovering 96.7 GW); C/D/E/K
remain degenerate at 764.7 GW. **This independently reproduces the earlier consumption-basis
attempt T1b cites ("退化 4 个（海河/黄河/淮河/西北）, ~765 GW")** — two independent
derivations landing on the same four basins and the same capacity is strong evidence that
**T1b's §4 root-cause diagnosis is correct and basis-independent**: the northern basins are
supplied by groundwater overdraft and the South-North Water Transfer, so local natural runoff
is not their supply universe. That is a hydrological fact, not a data defect, and it is *not*
fixed by switching bases.

### The justification for 0.85 needs checking against the right document

T1b §6 states "0.85 来自电力在全国取水统计中约 15% 的份额". The primary record
(`scripts/run_single.py:88-89`) names the source precisely: **《中国环境统计年鉴》 basin
supply**, not the Water Resources Bulletin. That document is **not in `reference/` and I could
not read it**, so the 15% is neither confirmed nor refuted here — see §10 for the full
provenance and the recommended verification.

What the bulletin does show is that 15% is hard to reconcile with its own fresh-intake basis:

- Once-through thermal/nuclear cooling = 453.8 / 5944.5 = **7.6%** of national withdrawal.
- **All** industry is 953.7 = **16.0%** of national withdrawal, so power at 15% of national
  would require power to be ~94% of all industrial withdrawal.
- The bulletin's 工业用水 is explicitly "按新水取用量计，不包括企业内部的重复利用水量", and it
  books the ~887 亿m³ of direct seawater use ("主要作为火（核）电的冷却用水") separately.

So the two figures are probably measuring different things rather than contradicting each
other — see the gross-circulating / seawater hypothesis in §10. The residual-allocation value
computed above (**0.559**) and the withdrawal-basis value (**0.970**) are what the *bulletin*
supports; 0.85 belongs to a different accounting basis and a different allocation rule.

## 9. Priority 2(c): official Chinese withdrawal caps — closed as unobtainable, with corrections

**Result: the artifact does not exist in the public record.** No official document publishes
用水总量控制指标 by the ten 水资源一级区. A dedicated search thread (four sub-searches, archived
government PDFs retrieved via Wayback and text-extracted directly) established this and, more
usefully, produced **positive proof that the allocation was executed as a closed provincial
partition**: 国办发〔2013〕2号 附件1 has 31 provincial rows, no 流域 or 一级区 rows, and the
columns sum exactly to the national caps.

| year | Σ 31 provinces | national row | residual |
|---|---|---|---|
| 2015 | 6350.00 | 6350 | **0.00** |
| 2020 | 6700.00 | 6700 | **0.00** |
| 2030 | 7000.00 | 7000 | **0.00** |

Source: `https://www.mee.gov.cn/zcwj/gwywj/201811/t20181129_676540.shtml`. Cross-validated: the
2030 provincial values reappear identically in 《长江经济带生态环境保护规划》专栏2 (11 provinces
summing to 3001.09). Because the partition closes to the national total with zero residual, a
parallel basin-level table is unlikely to exist at all. **Treat 2(c) as closed.**

### Two corrections to values this memo and the repo previously asserted

**1. 黄河 370 亿m³ is a 1987 CONSUMPTIVE allocation, not a withdrawal cap — and not a 黄河区
quantity.** The 370/580/210 split is confirmed verbatim:

> 「在黄河一般年份径流总量580亿立方米的基础上，扣除（预留）冲沙水量210亿立方米，剩余的370亿立方米
> 作为可供水量分配到沿黄9个省（区）和河北省、天津市。」
> — `https://www.waterinfo.com.cn/xsyj/tszs/202208/t20220816_34753.html`

But: (a) it is 「八七分水」, a **consumptive-use** allocation, so my earlier conversion of it to a
consumption basis at 55.0% was **wrong — it was already consumption**; (b) 20 亿m³ of the 370 is
allocated to 河北/天津, which lie in **海河区**, so 370 is not a 黄河区 total; (c) the current
《黄河流域综合规划》 supersedes it with **332.8 亿m³ 地表水消耗量** (and 401.1 亿m³ 河道外配置水量
post-西线), on a revised runoff of 534.8 rather than 580 亿m³:

> 「黄河流域地表水消耗量控制在332.8亿立方米以内，地下水开采量控制在123.7亿立方米以内」
> — `http://www.yrcc.gov.cn/zwzc/ghjh/202312/t20231220_365017.html`

**2. 长江 2348 亿m³ is UNVERIFIED and should not be cited.** It could not be sourced, and there
is a structural reason: 《长江流域综合规划（2012～2030年）》 was approved by 国函〔2012〕220号 but
**its text was never publicly released**. Do not confuse it with 长江经济带 (province-based:
2020 = 2922.19, 2030 = 3001.09 亿m³). `plan/literature_comparison_basis.md` §七 lists 2348 as
"已核实" — **that claim needs re-checking or retracting.**

### What WAS found (two of the seven), with a caveat that blocks their use

| region | 亿m³ | year | metric | source |
|---|---|---|---|---|
| 松花江**流域** | 404.77 / 430.24 | 2020 / 2030 | 用水总量 red-line | 《松花江流域综合规划》概要, 松辽委 |
| 辽河**流域** | 181.64 / 186.67 | 2020 / 2030 | 用水总量 red-line | 《辽河流域综合规划》概要, 松辽委 |
| 珠江**流域** | 640 | 2030 | 配置河道外总用水量 (allocation, *not* a cap) | 《珠江流域综合规划（简要稿）》 |

Verbatim, e.g. 「2030 年用水总量控制在 430.24 亿 m3 以内」
(`https://web.archive.org/web/20180813183952/http://slwr.gov.cn/slzx/slwlygh/201308/W020130830820965264814.pdf`).

**But 流域 != 一级区, so these cannot be used as level-1-region caps.** The plans state their own
extents (松花江流域 56.12 万km²; 辽河流域 22.11 万km²) while the level-1 regions are larger. The
proof is arithmetic: 松花江**区** actual 2022 use was 432.0 亿m³, already **above** the
松花江**流域** 2030 cap of 430.24. Mixing 370 (1987 consumption), 332.8 (surface consumption),
640 (allocation) and 430.24 (withdrawal) would be a category error across three metrics and two
vintages.

**Not found at all, for any year: 海河, 淮河, 东南诸河, 西南诸河, 西北诸河.** The 海河 概要 is
404 and absent from Wayback; 淮委 never published its plan text. A 淮河 figure of 641.6 亿m³
circulates in search-engine prose only and **must not be used** — it exceeds that basin's own
water resources.

### Revised policy comparison

Correcting my earlier double-conversion, and noting the model's headline runs use `dry`:

| | official (consumption basis) | model @0.85 annual | @0.85 dry | @0.56 dry |
|---|---|---|---|---|
| 黄河 | 332.8 (current plan) / 370 (1987) | 21.1 (**15.8x** stricter) | 5.36 (**62x**) | 15.7 (**21x**) |

The Yellow River remains the most valuable single find, because 「八七分水」 is an official Chinese
precedent with *exactly our semantics* — it splits a multi-year mean flow into a consumptive
allowance and an in-channel reserve, on the consumption basis Richter's standard calls for. Its
allowance is 370/580 = **63.8% of mean flow as depletion**, i.e. **3.2x more permissive than
Richter's 20%**. Caveats: our denominator (702.8) is third-survey total water resources including
groundwater, while the plan uses 534.8 surface runoff; and the policy cap embeds transfers and
groundwater. Use as order-of-magnitude context, not calibration.

**Net effect on the decision**: this *weakens* the policy-anchor argument for 0.56 specifically
(my earlier "within 1.5x of the Yangtze cap" rested on the now-unverified 2348) but *strengthens*
the general finding that the whole multiplied stack is extraordinarily strict relative to Chinese
policy — 62x on the runs behind Fig 2.

### Usable fallback: official actual use by all ten 一级区

2022 中国水资源公报, 表「水资源一级区供水量和用水量」
(`http://www.mwr.gov.cn/zzsc/tjgb/szygb/2022/files/basic-html/page31.html`), 亿m³ — **actuals,
not targets**: 松花江 432.0 · 辽河 188.6 · 海河 370.7 · 黄河 391.6 · 淮河 639.1 · 长江 2143.6 ·
东南诸河 285.1 · 珠江 779.1 · 西南诸河 106.2 · 西北诸河 662.2, summing to 5998.2 = the national
total exactly. This independently corroborates the 2025 表9 data used in §8 (e.g. 海河 370.7 in
2022 vs 378.3 in 2025; 黄河 391.6 vs 391.8). Flagged coincidence worth remembering: **海河区
actual use is 370.7, numerically almost identical to the Yellow River's 370 allocation** — do not
cross the two.

**The remaining seven caps were not obtained.** Per the assessment above they do not exist as
a published basin-level set; the tractable substitute is the provincial 国办发〔2013〕2号
targets reallocated to basin polygons, which is new work and not a lookup. A parallel search
thread was run on this question; its outcome is recorded in §12.

Also available in that document for shape validation: per-region 供水量 for 2013 from
《中国环境统计年鉴》 (全国 6183.4 亿m³, 地表 5007.3, 地下 1126.2, with all ten regions
broken out) — useful as an independent check on 表9 but still a *current-withdrawal*
quantity, not a cap.

### SUPERSEDED: earlier policy comparison retracted

An earlier draft of this memo compared the caps to the model by converting 370 and 2348 to a
consumption basis at the 55.0% 耗水率, concluding the model was "4-10x stricter" and that 0.56
landed "within 1.5x of the Yangtze cap." **Both halves of that were wrong** and are retracted:

- **370 is already a consumption quota** (rows headed 「年耗水量」), so converting it again
  double-counted the 耗水率. The correct like-for-like comparison is in §9c.
- **2348 is unverified** (above), so the Yangtze leg of the comparison has no basis at all.

The corrected version of this analysis, using the YRCC's own measured surface-water consumption
against the 370 quota, is §9c. Its conclusion is the opposite of the retracted one: the measured
share is **0.807-0.830**, so 0.85 is close to right and 0.56 is *not* better supported for the
basins where the constraint binds.

## 9c. THE RECONCILIATION — 0.85 is close to a measured value, but anchored to the wrong denominator

This is the memo's central finding and it supersedes the "0.85 is 1.5x too high" framing in
earlier sections.

**The 八七分水 allocation is a CONSUMPTION quota.** Its table rows are headed 「年耗水量」 and it
covers **surface water only** (「仅分配了黄河流域的地表水，而没有考虑地下水」). So it can be
compared like-for-like with measured surface consumption. From the YRCC 2024 bulletin
(`http://www.yrcc.gov.cn/gzfw/szygb/202512/P020251216363246418011.pdf`):

> 「2024年…黄河地表水耗水量为307.09亿立方米，其中黄河流域耗水量为209.73亿立方米，占68.3%；
> 调出流域水量97.36亿立方米全部为耗水量，占31.7%。」

| year | 黄河 surface consumption | / 370 official | / 350 (excl. 河北+天津) |
|---|---|---|---|
| 2024 | 307.09 | **0.830** | 0.877 |
| 2022 | 298.77 | 0.807 | 0.854 |
| 2024, power netted out | 287-291 | **0.776-0.786** | 0.809-0.831 |

**So the measured, like-for-like, consumption-basis non-power share for the Yellow River is
0.78-0.83 — and 0.85 sits just above that range.** The value is essentially right. What is wrong
is the denominator it is multiplied by:

| anchoring | allowance | measured share | power gets, as fraction of runoff |
|---|---|---|---|
| **(A) Richter-anchored** (what the code does) | 0.20 x 702.8 = 140.6 | 307.09/140.6 = **2.185** | **0** — degenerate |
| **(B) Official-quota-anchored** | 370 (= 63.8% of 580) | 307.09/370 = **0.830** | 0.638 x 0.17 = **10.9%** |
| **(C) Current code** | 0.20 x 702.8 | 0.85 *(an option-B-magnitude share)* | 0.20 x 0.15 = **3.0%** |

**The code mixes two regulatory universes**: it takes a share measured against China's official
64%-of-runoff consumptive allowance and multiplies it by Richter's 20% allowance. The result,
3.0% of runoff, happens to land *between* the two internally consistent answers (0% and 10.9%).
That is why the constraint behaves plausibly despite the inconsistency.

**Why option (A) goes degenerate is now explained, not merely observed.** Richter's 20% is
**2.6x stricter than China's own 八七分水 allocation** (370/140.6). A northern basin that fully
complies with Chinese water law still violates Richter's presumptive standard by a factor of two.
That is a real, citable, publishable finding — and it is the root cause of the four degenerate
basins in §8, independent of any data quality issue.

**Independent cross-validation of §8**: my §8 Yellow River share of 1.569 implies in-basin
consumption of 220.5 亿m³ against YRCC's measured 209.73 亿m³ — **5.2% agreement**, from
completely different inputs (表9 sectoral withdrawal x national 耗水率 vs. YRCC's own accounting).
The §8 method is sound.

⚠️ **A trap to avoid.** 314.37/370 = 0.850 looks like a perfect confirmation of 0.85. It is a
**mixed-basis ratio** — it adds 79.83 亿m³ of *groundwater* consumption to a *surface-water-only*
denominator. Do not use it. The clean figure is 0.807-0.830, or 0.776-0.786 power-netted.

## 9d. Priority 2(d) — there IS an optimization precedent, and it is status-quo-anchored

This corrects §8's claim that the status-quo-share philosophy has "none found" for precedent.

**Zhang, He, Johnston & Zhong (2021), *Journal of Cleaner Production* 329:129765** — SWITCH-China
with a basin-level power-sector water cap, sized **without any environmental-flow share**:

> "we assume the maximum freshwater withdrawal quotas at the secondary river basins for power
> generation in the first calculation year 2025 will keep at the base year level and will decrease
> linearly to 50% of the base year level by 2050"

Withdrawal basis; 76 secondary basins (67 with thermal plants) x provinces = 108 spatial units;
sensitivity run at 30% and 70%. Institutional rationale given: "Ministry of Water Resources (MWR)
and river basin management branches review and issue water withdrawal permits and implement the
gross water withdrawal cap policy." Anchored to 三条红线.

**This is the closest methodological precedent to our constraint that exists**, and it is
status-quo-anchored, not ecologically anchored — which legitimizes option (B) above and, by
extension, the *magnitude* of 0.85. It also reports sensitivity across the quota rather than a
point value, which is the norm this memo recommends adopting.

## 9e. Richter licenses our exact formula — quote him for it

The residual structure is not our invention and does not need defending by analogy. Richter et
al. 2012 prescribes it verbatim (primary PDF read by the search thread):

> "Although it is consistent with our presumptive standard to assume for planning purposes that
> **20% of the natural monthly mean flow can be allocated for consumptive use**, this does not mean
> that a volume of water equivalent to 20% of the monthly mean can be allocated on a fixed basis
> without violating our presumptive standard."

> "Use the modelling tool(s) to evaluate whether proposed withdrawals… — **when added to
> already-existing water uses** — would cause the presumptive standard to be violated."

Table II in that paper is headed **"Cumulative allowable depletion."** So: the 20% is a
consumptive allocation, and it applies **cumulatively across all users** — which is exactly
`0.20 x runoff x (1 - existing_share)`. **Cite these two sentences for the formula.**

Two required caveats:

1. **Richter explicitly rejects Smakhtin's weaker EFR**: "We agree with Arthington et al. (2006)
   that such a low level of protection as suggested by Smakhtin 'would almost certainly cause
   profound ecological degradation.'" Do not present Richter and Smakhtin as one coherent package.
2. **Hoekstra explicitly disclaims allocation**: "the application of this standard does not imply
   that 80% of the total runoff is unavailable for use… all of the runoff can be used, as long as
   no more than 20% of the total runoff is depleted." So do not claim 20% is the *power sector's*
   entitlement — the defensible claim is that power receives the **residual of a cumulative cap**.

### The parameter has a name: Smakhtin's WSI

Smakhtin, Revenga & Döll (2004), IWMI Comprehensive Assessment Research Report 2, p.9 (primary
PDF read):

```
Eq.(1)  WSI = Withdrawals / (MAR - EWR)      <- their contribution
Eq.(2)  WSI = Withdrawals / MAR              <- pre-existing form
```

`utilizable water = MAR - EWR`, and `utilizable x (1 - WSI)` is his residual — **our formula's
algebra exactly**. Their bands: >1 overexploited / environmentally water scarce; 0.6-1 heavily
exploited / environmentally water stressed; 0.3-0.6 moderately exploited; <0.3 slightly exploited.
**0.85 falls in "heavily exploited / environmentally water stressed"**, which is a defensible,
citable characterization of a northern Chinese basin. Note their numerator is withdrawals and
their EWR is 20-50% of MAR (so utilizable = 50-80%, not 20%); they also concede the bands "were
rather arbitrary."

## 10. The decision, in full

### First: the true provenance of 0.85

`scripts/run_single.py:81-98` is the authoritative record, and it is more careful than
`plan/t1b_per_basin_shares.md` §6 implies. Verbatim:

> "0.85 reserves the non-power share of the allowance, power keeping the ~15% it holds in
> national abstraction statistics (**China Environment Statistical Yearbook basin supply**)."

> "Basis note for Methods: the allowance is an ABSTRACTION quantity while the constraint acts
> on CONSUMPTION, so this is deliberately conservative -- it grants power ~7x more consumptive
> headroom than the **~2% of national depletion** it actually holds."

> "Deducting non-power CONSUMPTION from a consumption allowance instead is basis-consistent but
> degenerate: the share exceeds 1.0 in Hai/Yellow/Huai/Northwest and zeroes 765 GW."

Three things follow. First, **the source is 《中国环境统计年鉴》, not the Water Resources
Bulletin** — so my §8 objection was aimed at the wrong document. Second, the author already
identified and quantified both the basis mismatch and the per-basin degeneracy, and this memo
has now independently reproduced the latter (§8: same four basins, 764.7 GW). Third, the
comment gives a second, sharper number: **power holds ~2% of national depletion.** Under a
consistent status-quo-share rule on the *consumption* basis, that implies share = 0.98, not
0.85.

**The single load-bearing citation is therefore the "~15%" figure, and it is unverified.**
《中国环境统计年鉴》 is not in `reference/`. I could not check it. What I *can* say from the
bulletin is that 15% is hard to reconcile with a fresh-intake basis: once-through thermal is
7.6% of national withdrawal and *all* industry is 16.0%, so 15% would make power ~94% of
industry. HYPOTHESIS, flagged as such: the Yearbook may report thermal cooling on a
**gross-circulating** basis, which the bulletin explicitly excludes
("按新水取用量计，不包括企业内部的重复利用水量"), or may include the ~887 亿m³ of direct
seawater use that the bulletin books separately. Either would make 15% real but on a basis
inconsistent with the 20% allowance.

### The decision

**REVISED after §9c-9e. Chosen: declare 0.85 as a sourced assumption with the strongest
available defence, and fix its documentation — not its value. No re-run required.**

This reverses the earlier draft recommendation ("switch to 0.56"). The reason is §9c: the measured
like-for-like Yellow River value is **0.78-0.83**, and 0.85 sits just above it. The number was
never far off; what is wrong is that it is presented as if anchored to Richter's 20% allowance when
it actually measures utilization of China's official 64% consumptive allowance. That is a
documentation defect, and documentation is cheap to fix.

What to write in Methods, all five elements now sourced:

1. **The formula is Richter's own prescription.** Quote "20% of the natural monthly mean flow can
   be allocated for consumptive use" and "when added to already-existing water uses", and note
   Table II is headed "Cumulative allowable depletion" (§9e). This removes the need to defend the
   residual structure by analogy.
2. **The parameter has a name and a citation**: it is Smakhtin et al. (2004) Eq.(1)'s WSI, and
   `utilizable x (1 - WSI)` is his residual (§9e). At 0.85 the basin is in his "heavily exploited /
   environmentally water stressed" band.
3. **The value is empirically anchored**: the Yellow River's measured surface-water consumptive
   utilization of its official 八七分水 quota is **0.807-0.830** (0.776-0.786 power-netted), from
   the YRCC 2024 bulletin (§9c). Cite this as the basis for adopting ~0.85, and state that the
   八七分水 quota is itself a 「年耗水量」 (consumption) allocation — which makes it
   basis-consistent with our constraint.
4. **State the anchoring mismatch explicitly rather than letting a reviewer find it.** Richter's
   20% is **2.6x stricter than China's own 八七分水 allocation**. Combining an official-allowance
   share with a Richter allowance yields 3.0% of runoff, between the two internally consistent
   answers (0% under pure Richter anchoring, 10.9% under pure official anchoring). Present this as
   a deliberate intermediate choice, with both bounds reported.
5. **Cite the optimization precedent**: Zhang, He, Johnston & Zhong (2021) JCLP 329:129765 sizes a
   basin-level power-sector water cap from base-year actual use with no environmental-flow share,
   and reports sensitivity rather than a point value (§9d). Our approach is the ecological-standard
   counterpart of theirs.

Ranked actions:

1. **Rewrite the `scenario.py:130-134` comment and the Methods paragraph** per the five points
   above. Drop the "~15% of national abstraction" justification — it is superseded by the
   directly measured 0.78-0.83, which is a far better anchor and does not depend on the
   unverifiable 《中国环境统计年鉴》 figure. **This is the whole fix.**
2. **Report sensitivity across the anchoring**, not a point value: run the two consistent bounds
   plus the adopted middle. Zhang/He 2021 sets the precedent for this being expected.
3. **Correct two repo claims**: 长江 2348 亿m³ is unverified and 《长江流域综合规划》 was never
   publicly released (`plan/literature_comparison_basis.md` §七 marks it 已核实 — retract or
   re-check); and 黄河 370 is a 1987 *consumptive* allocation of which 20 亿m³ goes to 河北/天津 in
   海河区, superseded in the current plan by 332.8 亿m³ 地表水消耗量 (§9).
4. **Optional, if a reviewer pushes for per-basin values**: §8's table is ready, with the four
   northern basins reported as *findings* (non-power depletion already exceeds the entire Richter
   allowance) rather than as model inputs.

Why not the alternatives:

- *Switch the headline to 0.56* (my earlier recommendation) — **withdrawn.** 0.559 is the correct
  national average under pure Richter anchoring, but it is a national mean applied uniformly, and
  §9c shows the basin-specific measured value where the constraint actually binds is 0.78-0.83.
  Adopting 0.56 would loosen the constraint using a number that is *less* representative of the
  binding basins, and would cost a full re-run and probably Fig 2. Keep 0.559 as the reported
  national residual-allocation figure and as a sensitivity endpoint.
- *Cite 0.85 to the 《中国环境统计年鉴》 15% figure* — unnecessary now, and unverifiable (§10
  provenance). Superseded by the YRCC measurement.

### If a reviewer forces a move to pure residual allocation

Everything needed is ready, so this is a contingency plan rather than the recommendation.

1. **Definition**: `existing_withdrawal_share` = non-power blue water footprint / (0.20 x natural
   runoff) — the basin's pre-existing blue water scarcity ratio excluding power, per Hoekstra et
   al. 2012. Citations: Hoekstra et al. 2012 PLoS ONE 7(2):e32688; Richter et al. 2012 River Res
   Applic 28(8):1312-1321; 《2025年中国水资源公报》表9; the same bulletin's sectoral 耗水率; Jin
   et al. 2022 Energy 245:123339.
2. **Headline value 0.56** (national, §8).
3. **Report the per-basin table in §8 as a result.** Four northern basins (海河 3.19, 淮河 1.89,
   西北诸河 1.79, 黄河 1.57) have share > 1: non-power depletion already exceeds the entire Richter
   allowance. State it as a finding — and note §9c now *explains* it, since Richter's 20% is 2.6x
   stricter than Chinese water law, so policy-compliant northern basins still breach it. Secondary
   cause per `plan/t1b_per_basin_shares.md` §4: groundwater overdraft and South-North Water
   Transfer imports mean local runoff is not the supply universe.
4. **Sensitivity envelope**, every endpoint sourced: **0.27 / 0.56 / 0.83** (东南诸河 residual /
   national residual / Yellow River measured quota utilization).

### The cost of that contingency, stated plainly

0.56 nearly triples the water offered to power (8.8% of annual runoff before the dry factor,
versus 3.0% at 0.85), so **the constraint would bind much less and might not bind at all.**
`scripts/run_single.py:96-98` is explicit that Fig 2 rests on the 0.85 runs: "At 0.85 the unabated
fleet still fits every basin (Hai peaks at 0.86 of budget), but attaching capture takes Hai to
1.45 -- infeasible unless it converts to air cooling. That is the water-carbon trade-off, so these
runs are the ones Fig 2 rests on."

**This is the decisive argument for keeping 0.85.** A national mean (0.56) would replace a value
that matches measured data in the binding basin (0.78-0.83) with one that does not, loosen the
constraint, and likely destroy the paper's central figure — all in the name of consistency with a
denominator convention that §9c shows is itself 2.6x stricter than Chinese law. Keep 0.85, declare
the anchoring, and report 0.56 as the residual-allocation sensitivity endpoint.

### Minimum acceptable fallback

Now superseded — the recommendation itself requires no re-run, so there is no cheaper option to
fall back to. If even the Methods rewrite is out of scope, the irreducible minimum is: (a) stop
citing the unverifiable 《中国环境统计年鉴》 15% figure and cite the YRCC-measured 0.807-0.830
instead (§9c); (b) state the compounded effective allowance of 0.76% of annual runoff (§10b)
rather than implying a bare 20%.

## 10b. The conservatism compounds — and only one layer of it is sourced

This memo's parameter does not act alone. Per `plan/environmental_flow_sourcing.md`, three
multiplicative conservative choices stack:

| layer | value | sourced? |
|---|---|---|
| Richter presumptive standard | x0.20 | **yes**, verbatim (§1, §9e) |
| `existing_withdrawal_share` | x0.15 (at 0.85) | **yes now** — YRCC-measured 0.807-0.830 (§9c), but anchored to the official allowance, not this one |
| `water_season = "dry"` (CONFIRMED used by the headline runs) | x0.254 (verified national dry/annual ratio from `inputs/water_availability.csv`) | partly — physically motivated, but changes the effective standard |

The headline water runs are `WA_cwatm_{126,370}_dry_wd085` (`scripts/run_single.py:99-100`), so
all three layers apply and the effective allowance is
0.20 x 0.254 x 0.15 = **0.76% of annual runoff**. Against the Yellow River's official 63.8%
abstractable fraction (§9) that is nearly two orders of magnitude. At 0.56 it would be
0.20 x 0.254 x 0.44 = 2.2%.

**This makes sourcing the middle layer more urgent, not less.** A reviewer who multiplies the
three factors will ask which of them carries the evidence. As of §9c-9e all three now have
citations, but the answer must be given proactively, and the `dry` layer must be declared:
`plan/environmental_flow_sourcing.md` already warns that one cannot simultaneously claim "we used
Richter's 20%" and use the `dry` column, since actual strictness is then a quarter of the nominal
value; when reporting dry results it recommends stating the equivalent annual-mean extractable
fraction (~5% before this parameter, ~0.76% after).

### Power's actual share, for the record

Two sourced figures that bound the problem and correct the code comment's estimate:

- **Withdrawal**: once-through thermal+nuclear cooling is **7.6%** of national withdrawal in 2025
  (453.8/5944.5), stable at 7.6-8.3% across 2019-2025. Including recirculating and air-cooled
  intake, Zhang et al. 2018 gives 57.6 km³ for 2015 = ~9.4% of national that year.
- **Consumption**: **~1.1%** of national water consumption. Coal-fired plant consumption is
  3.5 km³/yr (2013) per Zhang X, Liu J, Tang Y, Zhao X, Yang H, Gerbens-Leenes PW, van Vliet MTH,
  Yan J (2017), *J. Cleaner Production* 161:1171-1179, doi:10.1016/j.jclepro.2017.04.040 — 11% of
  industrial consumption, 84% from closed-cycle cooling, 1.15 L/kWh, ~75% of it in absolute or
  chronic water scarcity. Against national consumption of 327.2 km³ that is 1.1%.

So the defensible one-liner is: **thermal/coal power is ~8% of China's freshwater withdrawal but
only ~1% of its water consumption.** This supersedes the "~2% of national depletion" estimate in
`scripts/run_single.py:92` (which was close) and confirms industry's low consumptive rate: 16.0%
of national withdrawal but only 7.0% of national consumption (耗水率 23.9%).

**Seawater convention must be declared.** The bulletin excludes direct seawater use
(「直接利用的海水另行统计，不计入供水量中」); 2025 direct seawater use was 1863.9 亿m³,
「主要作为火（核）电的冷却用水」. Counting it, power's share of withdrawal jumps to ~29% from the
same physical reality — a plausible partial explanation for the unverifiable "~15%" figure.

### Once-through cooling is almost absent from the basins that bind

From 表9/表10, the 直流火(核)电 column by 一级区 (亿m³): **黄河区 0.0 in 2022, 2024 and 2025**;
海河区 0.1-0.3; 北方六区 12.5 total; 南方四区 441.3 (长江区 alone 394.4). **Once-through is ~97%
southern and coastal.** In the northern basins where coal capacity and water stress coincide,
plants already run recirculating or air cooling — low withdrawal, high consumptive fraction.

Two implications. First, the national "power = ~48% of industrial withdrawal" statistic says
nothing about the basins the optimization actually binds in, so it should not be used to justify a
uniform national share. Second, it independently supports the consumption basis: in the binding
basins there is almost no return flow to argue about.

### One gap in the prior sourcing doc is now closed

`plan/environmental_flow_sourcing.md` lists as unresolved item 4: "中国煤电/水-能关联文献中的
显式可取用比例 —— 未找到任何明确给出 extractable fraction 数值的同类研究."

**Found.** Jin et al. 2022 (Energy 245:123339), a China coal-power water-vulnerability paper,
states an explicit extractable fraction: "For the rivers that supply water for human use in
China, **60% of the average discharge needs to be preserved for environmental flow**", i.e.
**40% extractable**, cited to 韩琦 et al., *Calculation methods of river environmental flow and
their applications*, 武汉大学学报·工学版 2018;51:189-197 (§4). So the closest China-specific
comparator adopts an allowance **twice as permissive as our 0.20**. This is usable two ways:
as the loose end of the 0.20 sensitivity range, and as evidence that our stack is conservative
relative to the China-specific literature — but it also means "0.20 is standard" cannot be
claimed for China without qualification.

## 11. Priority 2(a): the two different 0.2s — do not conflate them

This is the clearest methodological trap in this literature and it is worth one explicit
sentence in the paper, because a reviewer may well raise it.

**There are two unrelated 0.2 thresholds in play:**

| | Falkenmark / Alcamo / Vörösmarty 0.2 | Richter / Hoekstra 0.2 |
|---|---|---|
| What it is | withdrawal-to-availability **stress class boundary** | fraction of natural runoff **allowed to be depleted** |
| Basis | withdrawal / total runoff | consumption / natural runoff |
| Role | **indicator only** | **allocation rule** |
| Our use | not used | **this is our 0.20** |

Across every source read for this memo, the WTA-style thresholds are used **only as
indicators**, never to allocate water to a sector:

- **Qin et al. 2023 (Nature Water)** — `WSI = W/Q`, classes 0.1 / 0.2 / 0.4, a screen for
  which units are dry-cooling candidates. Threshold cited to Oki & Kanae, Science 2006;313
  and Vörösmarty et al., Science 2000. No allocation.
- **Wang F et al. 2023 (Water 15:1167)** — WTA and CTA against available water, WRI Aqueduct
  six classes (<0.1 / 0.1-0.2 / 0.2-0.4 / 0.4-0.8 / >0.8), computed ex-post. No allocation.
- **Jin et al. 2022 (Energy 245:123339)** — `WC/WA > 1` as a scarcity flag, plus a hard
  useable-capacity cap from `min(Q - NEW, q)`. The *allocation* here comes from subtracting
  other sectors' consumption, not from a stress threshold.

Only the **Richter presumptive standard, as operationalized by Hoekstra et al. 2012, is an
allocation rule** — and it is the one we use. So the correct statement in the paper is: *we do
not use a water-stress threshold to allocate water; we apply an environmental-flow depletion
standard and give the power sector the residual.* That sentence pre-empts the conflation and
also states the method's lineage.

NOT VERIFIED (searched, not obtained): Smakhtin et al. 2004's exact water-stress-with-EFR
formulation, and the primary attribution of the 0.4 "severe stress" boundary to Raskin et al.
1997 / Alcamo et al. I read these only through the citing papers above, so do not cite them
directly without checking the originals. Note that
`plan/environmental_flow_sourcing.md` has already verified Smakhtin's **EFR = 20-50% of
mean annual flow** (extractable 0.50-0.80) via Richter's verbatim text, and flags that the
volume/issue/page details still need checking against the original — so the *number* is
reliable, the *bibliographic detail* is not yet.

## 12. Unresolved, and what I could not read

Marked explicitly rather than inferred:

1. **Rosa et al. 2020, Nat Sustain, *Hydrological limits to carbon capture and storage*** —
   the closest topical precedent (global CFPP + CCS + monthly hydrology + hard water limits).
   eScholarship returned empty HTML then HTTP 403; nature.com redirects to an auth endpoint;
   the PDF host served undecodable binary. **Methods not read.** Its reference list is
   confirmed to include Richter et al. 2012. Jin et al. 2022 (p.3) states it "showed little
   sensitivity of water scarcity to different environmental flow requirements" — if verified
   in the original this is a direct citable defence for the choice of EF fraction. **Highest
   value remaining action: obtain this PDF through institutional access.**
2. **Richter et al. 2012 verbatim** — quotes in §1 come from search snippets of the published
   text, not from the PDF, which I could not decode. Verify against the Wiley version before
   quoting in the paper. The substance is independently corroborated by Hoekstra et al. 2012,
   which I did read.
3. **Li H et al., *Catchment-level water stress risk of coal power transition*** — surfaced as
   Wang & Cai 2024's ref 32, the only water citation anywhere in the four CCS-network papers.
   China-specific and catchment-scale, so likely the closest comparator not yet examined.
   **Not retrieved.**
4. **The seven remaining official basin caps** — assessed as not existing in published
   basin-level form (§9). If a reviewer demands them, the route is provincial 国办发〔2013〕2号
   targets reallocated onto basin polygons.
5. **Per-basin consumption is constructed, not measured.** The bulletin publishes 耗水率
   nationally only; §8 applies national sectoral rates to per-basin sectoral withdrawals. That
   is Jin et al. 2022's method and the rates are cross-validated to within 2 pp, but the
   per-basin values inherit an unquantified error from assuming uniform sectoral rates.
6. **Recirculating and air-cooled thermal withdrawal cannot be separated** from residual
   工业 in 表9 (only 直流火(核)电 has a sub-column). Per `plan/data_basin_sectoral_water.md`
   §4 this biases the computed share **upward**, so §8's values are upper bounds and the true
   shares are somewhat lower — which strengthens, not weakens, the case against 0.85.
7. **Whether the paper's headline results use `water_season = "annual"` or `"dry"`** —
   **RESOLVED**: `scripts/run_single.py:99-100` shows the 0.85 runs are
   `WA_cwatm_{126,370}_dry_wd085`, i.e. **dry**. Effective allowance 0.76% of annual runoff
   (§10b). Methods must state this.
8. **《中国环境统计年鉴》 — the source of the ~15% power abstraction share.** Not read.
   **No longer critical**: §9c's YRCC measurement (0.807-0.830) is a better anchor and makes the
   15% figure unnecessary. Recommend dropping the claim rather than chasing it.
9. **The "~2% of national depletion" figure** (`scripts/run_single.py:92`) — **RESOLVED**, and it
   was close: the sourced value is ~1.1% (§10b, Zhang et al. 2017).
10. **Thermal power's share of Yellow River basin industrial consumption** — no source found. The
    30-50% bracket behind the "power-netted 0.776-0.786" figures in §9c is an assumption, flagged
    as such by the search thread. The un-netted 0.807-0.830 needs no such assumption; prefer it.
11. **Smakhtin bibliographic detail** — the primary PDF was read (IWMI Comprehensive Assessment
    Research Report 2), but `plan/environmental_flow_sourcing.md` cites it as *Water International*
    29(3):307-317. **Two different publications may be conflated; check before citing.**

## 13. Do-not-cite list (from the search threads, verified or flagged)

Recording these because each would be an easy, plausible-looking error:

- 🔴 **314.37/370 = 0.850** — looks like perfect confirmation of 0.85; is a **mixed-basis ratio**
  (groundwater consumption over a surface-water-only denominator). Use 0.807-0.830 (§9c).
- 🔴 **长江 2348 亿m³** — unverified; the source plan was never publicly released (§9).
- ?? **淮河 641.6 亿m³ (2030)** — appears only in search-engine prose, unfetchable, and exceeds the
  basin's own water resources (583.5 亿m³). Do not use.
- 🔴 **Qin et al. 2015 "84% of industrial withdrawal is thermoelectric"** — not locatable at source
  and contradicted by the bulletin (47.6-50.5%).
- 🔴 **Liao & Hall 2018 withdrawal 124.06 km³** — exceeds all of China's 2015 industrial freshwater
  withdrawal; almost certainly includes seawater. Their consumption figure (4.86 km³) is usable.
- 🔴 **Jin et al. 2023 "13.6 Gm³ consumption peak"** — 3-4x other estimates; boundary unconfirmed.
- ⚠️ **Vörösmarty 2000 and Falkenmark 1989** — secondary-sourced only. Damkjaer & Taylor 2017
  (*Ambio* 46:513-531) documents systematic misattribution in exactly this literature: the 0.2/0.4
  thresholds trace to Raskin et al. 1996/1997 with "the basis for this judgment… unclear", and
  Vörösmarty et al. 2005 cite Falkenmark 1989 for the 1700 m³ figure — "a paper which makes no
  claim to this threshold." **Verify against primaries before citing any specific number.**
- ⚠️ **Zhang et al. 2018 *Nature Energy* 67.3 km³ / ~11%** — from an institutional summary, not the
  paywalled text.
- ⚠️ **海河 106% utilization** — a 2010 MEE document describing 11th-Five-Year-Plan conditions; not
  current. Local-only gives ~72% (103.42 亿m³ of 海河's use is imported).
- ⚠️ **China's 40% 生态警戒线** — the Chinese wording is 「生态警戒线」; do not render it as
  "internationally recognized threshold." It functions as diagnostic prose; the binding instruments
  are volumetric.
