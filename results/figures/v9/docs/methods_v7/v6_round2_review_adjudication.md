# v6 round 2 — the review panel's claims, my verification, and what was done

Four reviewers ran in parallel: a Nature Water referee (storyline and framing), an optimisation
methodologist (MILP and inference), a hydrologist (water methodology), and a figure editor. Two
of these were retries of agents that died on API errors in round 1.

Same rule as round 1: **nothing is adopted on an agent's say-so.** Each claim below was
re-derived from the repository or the run outputs before it was acted on. Where my own numbers
disagreed with a reviewer's, I say so and give both.

---

## A. Claims I verified and ADOPTED

### A1. Fig 4's panel (d) title asserted the opposite of what the panel computes
**Confirmed, and worse than reported.** The script's own verdict line prints
`ABOVE the floor on both metrics - attributable to water`, while the title read "no network
change is attributable to water". Shared-edge overlap: treatment [91.97, 95.37] vs control
[97.19, 98.64], Δ −4.25 pp. Intersection: [69.95, 81.52] vs [87.02, 87.21], Δ −11.38 pp. Both
ranges fully disjoint.

**Action:** the title is now *derived from the verdict* rather than hardcoded. This changes what
Fig 4 is: it is **not a null**. Water changes *which* corridors carry the CO₂ (−4.3 and −11.4 pp
beyond the seed null) while leaving *how much* pipeline is built inside the floor. That also
overturns the referee's recommendation to demote the figure — it was demoted for being a null it
never was.

### A2. Fig 2 panels (b) and (c) carried contradictory titles 30 mm apart
**Confirmed.** (b) said "Scarcity is not the binding constraint — allocation is" while (c) said
capture turns allocation into scarcity. (b)'s own crossing variable `first_share` is the Hai's
median s\* = **−0.081**: the Hai breaches at zero reservation, which is scarcity, under a title
denying scarcity.

**Action:** (b)'s title is now derived from `first_share`, and reads "Allocation governs every
basin but one — in C Hai River, capture makes it scarcity".

### A3. Fig 2(b) had an orphan triangle in the left margin with no explanation
**Confirmed, same root cause as A2.** The marker was drawn at x = −0.081 with `clip_on=False`
against an `xlim` starting at 0.0, while `annotate`'s default `annotation_clip` silently dropped
the text that explained it.

**Action:** when the crossing falls below the axis, the panel now states it in words instead of
pointing off-canvas.

### A4. The certified interval's upper endpoint used the wrong denominator
**Confirmed analytically.** The quantity is (OPT_t − OPT_c)/OPT_c, which is decreasing in OPT_c,
so its maximum over the certified box is at OPT_c = LB_c and must divide by LB_c.

**Action:** ledger fixed. The headline interval is **[+0.639, +2.188]**, not [+0.639, +2.170].

### A5. A missing `mip_gap` was silently coerced to zero
**Confirmed.** That does not widen an interval — it makes an uncertified run look certified and
tight. `RQ3_retire_only.json` already carries `mip_gap: null`.

**Action:** the ledger now refuses to certify rather than coercing.

### A6. Three different degeneracy floors were in circulation
**Confirmed:** 0.4503% (n = 3, storyline and summary) and 0.3702% (n = 4, ledger). The raw range
is 0.2750% in all of them — only the d₂ divisor moved.

**Action:** reconciled to **0.3277%** (n = 5, after `seed5` was re-solved onto the corrected
basis). The headline clears it **4.1×**.

### A7. Jin et al. 2022's 40% convention is already solved — it is the `s = 0.70` run
**Confirmed, and independently by two reviewers.** 0.40 × (1 − 0.85) = 0.20 × (1 − 0.70) = 0.060,
so the two are the same model. That run gives **+0.570%, certified [−0.350, +1.400] — an interval
that contains zero.**

**Action:** adopted as a first-order limitation. The **cost** headline is resolved only at the
strictest environmental-flow convention in circulation. The paper already concedes
convention-dependence for the *scarcity* claim; it must now do so for the *cost* claim. The
*physical* claim (dry-cooling retrofit rises 127 → 244 GW even under Jin's convention) survives
and is the more robust headline.

### A8. The steep end of the new reservation curve is mostly the price cap, not physical cost
**Confirmed against the model's own accounting — and my own first estimate was wrong.** I priced
`slack_detail.csv` volumes by hand and got 17%. `cost_breakdown.csv` carries a `slack_penalty`
category and is authoritative: **63.7%** of the 0.85 → 0.90 increment is the 1000 CNY/m³ literal.

**Action:** the curve is rewritten on the physical series. Marginal cost rises **42×**, not the
117× I reported. At s = 0.85 the headline is **94.5% physical**, so it is barely affected.

### A9. "China sits at the knee" is a statement about a convention, not about China
**Confirmed.** The solver sees only `usable = extractable × (1 − s) × water_multiplier`
(`data_prep.py:452-455`), so the same physical point sits at s = 0.85 under a 0.20 ceiling and
s = 0.925 under 0.40.

**Action:** claim withdrawn. The curve is reported against the usable share u, which is
identified; the knee sits at u ≈ 0.03 of dry-season flow.

### A10. The zero-storage firm-yield assumption is what produces the Hai scarcity result
**Confirmed, and it is the most fragile claim in the paper.** Hai dry_share = 0.2197, so on an
annual basis s\* moves **−0.081 → +0.763**. Solving for the carryover that reaches s\* = 0 gives
**β = 0.0228**: carrying just **2.3%** of the way from firm dry-season yield toward the annual
mean erases the result — in one of China's most heavily reservoired basins.

**Action:** this becomes the headline caveat on Fig 2(c) and §1.1. The result stands as a
defensible worst case, but it must be labelled a **lower bound**, not an estimate, and the β
sensitivity published.

### A11. The MILP does not enforce a basin water budget
**Confirmed, by two independent reviewers reaching the same statistic.** 25% of 1,569 links cross
a basin divide (200 km radius, `data_prep.py:328-341`). In the binding run the Hai's import share
is **32.7 / 34.4 / 32.6%** at 2040/50/60 against 8.7–13.7% in the non-binding control — the model
uses transfers as an escape valve exactly when the Hai binds.

**Action:** adopted. s\* charges 100% of Hai demand to Hai water while the MILP charges ~67%. The
two measure different systems and must not be quoted as if they corroborate one another.

### A12. Biomass feedstock water is not merely missing — the resource is 39% dedicated energy crops
**Confirmed on provenance and geography.** `builders/supply.py:101` loads Wang et al. 2023
*Sci Data* scenario **S3** ("energy crops: yes, on abandoned cropland"); S4 (residues only) is
17.9 EJ against S3's 29.3 EJ, so **11.4 EJ (39%)** is dedicated crops. Hai plants draw 83% of
their biomass from inside the Hai. Independently verified from `resource_use.csv`: utilisation
runs **79% at 2040 and 87% at 2050**, with 6,781 and 9,352 of 13,949 nodes at cap.

**Action:** adopted as the largest open exposure. The break-even thresholds — the loop stays
water-positive below ~3.3 m³/GJ, the Hai claim fails above ~0.12 m³/GJ — are the deliverable, and
they require no re-solve.

### A13. "Paid in bioenergy, not in emissions" is a 2030-only statement
**Confirmed to the decimal.** Residual delta: 2030 **−0.00**, 2040 **−54.53**, 2050 **−91.54**,
2060 **+11.86 Mt**; horizon sum **−134 Mt**. And 2030's exactness is the exogenous target
binding — the model rediscovering its own constraint.

**Action:** adopted. Horizon-wide the water-constrained pathway is carbon-**negative**. That is a
co-benefit and it strengthens the paper; discovered by a referee instead, it would not.

### A14. The seed floor's 1.96 treats a range-based sigma as if it were known
**Partly confirmed.** The criticism is sound in principle; at n = 5 the estimator is tighter than
the n = 4 case the reviewer computed against.

**Action:** stated as a limitation rather than recomputed. The floor is a screening threshold and
the headline clears it 4.1× under either construction.

### A15. Fig 3 printed "+1.27%" on top of "+0.07 penalty"
**Confirmed:** the bold label sits at `y = econ` and the penalty label at `y = econ + pen/2`,
identical to within a third of a point when the penalty is small.

**Action:** the in-segment label is gated on rendered height (`pen/span > 0.05`); below that it
becomes a leader callout.

### A16. Fig 2(c) wasted a third of its x-range and hid its legend in the shaded region
**Confirmed by inspecting the rendered figure.**

**Action:** `xlim` −1.05 → −0.72; marker offsets ±0.17 → ±0.24; "as built" markers now hollow so
the pair separates by shape as well as colour; legend moved to lower right, out of the shaded
scarcity region.

---

## B. Claims I could NOT reproduce as stated

| claim | what I measured | disposition |
|---|---|---|
| "`wd090` is 44.2% penalty; `wd085_noair` 27.7%" | Against the **objective** those shares are 1.70% and 1.42%. The 44.2% is the share of the **cumulative effect vs control** — which my check then reproduced exactly. | Adopted with the denominator made explicit. The conclusion was right; the phrasing was ambiguous. |
| "the solved member is the 2nd-driest of 20" (the paper's claim, challenged as 6th) | **Both are right, on different metrics** — which is itself the problem. Rank **2/20** on the four over-limit basins C/D/E/K (span 1.68×, matching the paper's wording), **6/20** nationally, **16/20 in the Hai alone**. | Adopted as a disclosure: the paper solves a member that is near-worst-case on the aggregate and 5th-**wettest** in the basin its scarcity claim is about. |
| "one hydrology member is solved" (the paper's own §6 item 9) | **Four** are solved as complete control/treatment pairs: +1.172 / +1.340 / +1.506 / +1.605%, every one resolved at 3.6–4.9× the floor, **every certified interval excluding zero**. | The paper understates its own evidence base fourfold. Corrected. The real gap is **GCM** coverage — all four are gfdl-esm4 — and two further GCMs are solving now. |

---

## C. Adopted in principle, queued rather than done

* **Re-solve on Wang S4** (residues only, 17.9 EJ) as the central biomass case, with S3 as an
  upper resource bound.
* **β storage sensitivity** — two lines at `data_prep.py:419-421`; both columns are already in
  the CSV.
* **Basin-match filter or transfer tariff** on water links (A11): one line, one solve.
* **Rename `existing_withdrawal_share` → `existing_consumption_share`**; it is documented and
  used as a consumption share.
* **Re-solve `wd090` at 10× the slack penalty** before its total is quoted anywhere.
* **`BASE_noair`** carries four confounds against its partner (gap 0.02 vs 0.01, unpinned
  threads, no fingerprint, pre-correction vintage) and is consumed live by
  `analyse_water_attribution.py`. Re-solve it or drop the contrast.
* **`mip_focus` provenance is false in every run** — `_shared.py:80` sets MIPFocus = 1
  unconditionally while the stamp records 0. No campaign exports the variable, so the runs really
  are comparable; the record simply does not prove it. Stamp it from the model.
* **CHP is absent entirely** — no match for `chp|cogener|热电|district heat` anywhere in the
  repository. Roughly half of China's coal capacity is cogeneration, concentrated in
  Beijing-Tianjin-Hebei, must-run in winter, which is the low-flow quarter in all nine basins.
  This is the largest single omission a water referee will name.
