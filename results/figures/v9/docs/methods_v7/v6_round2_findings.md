# v6 round 2 — findings I verified myself, before the review agents reported

Round 1 (this directory's README) ran: agents review → I re-verify → fix only what survives.
Round 2 repeats that loop. The items below are ones I found and confirmed independently while
the second review panel was still running; agent findings are folded in afterwards, separately,
so it stays visible which came from where.

---

## R2-1. The degeneracy floor in the numbers ledger was inflated 12×, and it was inflated
## across the one comparison the paper's headline depends on

`build_numbers_ledger.seed_entries` collected seed replicates **by filename**, globbing
`WA_cwatm_126_dry_wd085` plus `_seed2..10`. `WA_cwatm_126_dry_wd085_seed5` had been solved on
the superseded per-cell-minimum water basis and was never re-queued — the v5 resume queue was
written as seed2/3/4 only. It carries

    fingerprint 0xbe7b31c2   objective 1.477875e13     (superseded basis)
    vs family   0xbf2f6de    objectives 1.4142-1.4181e13 (corrected basis)

Pooling it reported:

| | pooled with seed5 | gated |
|---|---|---|
| seed range | 4.5013% | **0.2750%** |
| sigma | 1.9352% | **0.1336%** |
| 95% envelope on a difference | **5.3642%** | **0.3702%** (n = 4; now **0.3277%** at n = 5) |

The headline effect is 1.340%. So the ungated ledger said the floor was **four times the
effect** — i.e. it told any referee who read it that the paper's main result is
indistinguishable from solver noise. It is not: against the true floor the effect clears
**4.1×** (3.6× at the n = 4 floor this was first measured against).

Note what this was NOT caught by. `same_model_runs` would have rejected seed5 on fingerprint,
and Fig 5's floor test did reject it. The ledger simply never called the gate. Fixed: the seed
family is now gated on `same_model_runs` **and** `input_vintage`, and names what it drops.
`WA_cwatm_126_dry_wd085_seed5` is being re-solved on the corrected basis to restore n=5.

## R2-2. The vintage guard was reporting false positives, because it was inferring provenance
## from file mtime rather than measuring it

`input_vintage` returned a real digest when a run carried one and otherwise guessed
`pre-digest:current` from `mtime > INPUT_REBUILD_EPOCH`. Runs solved on the corrected inputs
but before the digest patch landed therefore compared as a *different vintage* from digested
runs, and Figs 4 and 5 each reported a 2-vintage mix that did not exist.

Replaced the guess with a measurement. Gurobi's fingerprint is sensitive to the water RHS, and
two runs of the same hydrology member on the same basis must agree exactly on the water they
were solved against. So a digest-less run now inherits the digest of any digested run whose
water right-hand side it matches to the bit, read from its own `resource_use.csv`. Measured,
this separates with no tolerance to tune:

    WA_cwatm_126_dry       vs WA_cwatm_126_dry_noair        max rel diff 0.000e+00  same basis
    WA_cwatm_126_dry_wd085 vs WA_cwatm_126_dry_wd085_noair  max rel diff 0.000e+00  same basis
    WA_cwatm_126_dry_wd085 vs WA_cwatm_126_dry_wd085_seed5  max rel diff 1.603e-01  DIFFERENT

The guard is now strictly stronger as well as quieter: a superseded run that happened to be
re-serialised recently used to read as current and now does not. Four runs with no digested
counterpart to verify against are being re-solved rather than left on an inference.

## R2-3. The cost of water headroom is convex — but the two strongest things I said about
## it were wrong, and the review panel caught both

I reported this as a 117× rise in marginal cost with "China sits at the knee". Both are
withdrawn. What survives is still the most water-policy-shaped result in the project, but it is
smaller and it lives on a different axis.

**Correction 1 — the steepest segment is mostly a price cap.** Unserved water is charged at a
bare literal 1000 CNY/m³ (`solver.py:711`). At s = 0.90 the fleet stops adapting and buys the
shortfall instead. I first estimated the penalty share by pricing `slack_detail.csv` volumes by
hand and got 17%; that was wrong — it under-captures what the objective charges. The model's own
`cost_breakdown.csv` carries a `slack_penalty` category and is authoritative:

| s | total effect % | of which penalty | **physical** % |
|---|---|---|---|
| 0.56 | 0.254 | 0.0% | 0.254 |
| 0.70 | 0.570 | 0.0% | 0.570 |
| 0.85 | **1.340** | **5.5%** | **1.266** |
| 0.90 | 3.994 | **44.2%** | 2.229 |

63.7% of the 0.85 → 0.90 *increment* is the literal. On the physical series the marginal cost
runs 0.45 → 2.26 → 4.64 → 19.25 %/unit — a **42×** rise, not 117×. Convexity survives; the
multiple does not. The headline at s = 0.85 is barely touched: it is **94.5% physical**.

**Correction 2 — "China sits at the knee" is a statement about a convention, not about China.**
The solver only ever sees `usable = extractable × (1 − s) × water_multiplier`
(`data_prep.py:452-455`). The same physical point sits at s = 0.85 under a 0.20 extractable
ceiling and at s = 0.925 under Jin et al. 2022's 0.40. So the knee's location on s is not
identified. Reported against the usable share u — which is identified — the knee is at
u ≈ 0.03 of dry-season flow and the claim becomes convention-free.

**And a consequence I had not drawn, which is the sharpest thing here:** Jin et al.'s convention
(0.40 extractable at s = 0.85, u = 0.060) is *arithmetically identical* to my solved s = 0.70.
Its effect is **+0.570%, certified [−0.350, +1.400] — the interval contains zero.** So the
headline cost effect is resolved **only at the strictest environmental-flow convention in
circulation**. The paper already concedes convention-dependence for the *scarcity* claim; it
does not for the *cost* claim, and it must.

Defensible statement: the cost of headroom is unresolvable while power keeps more than ~6% of
dry-season flow, is resolvable and physical at 3%, and beyond that the model stops adapting and
starts paying a price cap. China's published conventions straddle exactly that range. Full
detail: `results/v6_allocation_curve.txt`.

**Two further arithmetic corrections that came out of the same audit and are now fixed in
`build_numbers_ledger.py`:**
* The certified upper endpoint divided by the control's *incumbent* where it must divide by the
  control's *bound* — the maximum of (OPT_t − OPT_c)/OPT_c over the certified box is at
  OPT_c = LB_c. The headline interval is **[+0.639, +2.188]**, not [+0.639, +2.170].
* A missing `mip_gap` was silently coerced to 0.0, which does not widen an interval — it makes
  it look certified and tight on a run that reported no gap at all. It now refuses instead.

**The degeneracy floor: three numbers were in circulation and there is now one.** The storyline
and `v6_corrected_basis_summary.txt` quote 0.4503% (n = 3, from before the `mip_focus`
normalisation was fixed); the ledger quoted 0.3702% (n = 4). With `seed5` re-solved the gated
family is n = 5 on an unchanged raw range of 0.2750%, giving σ = 0.1182% and a floor of
**0.3277%**. The headline clears it **4.1×**. All documents are being reconciled to that number.

## R2-4. "One hydrology member is solved" is wrong — four are, and the one the paper leans on
## is the member where its own headline scarcity result does NOT occur

The storyline (§6 item 9) says one member is solved and lists "[pending] 3 members × 2 arms".
In fact four members are solved as complete control/treatment pairs: cwatm and watergap2-2e ×
ssp126 and ssp370, all gfdl-esm4.

The sharper point is *where* the solved member sits. Ranked by the statistic the paper turns on
— s\* for full capture in the Hai at 2030 — the solved member is **16th of 20**, i.e. among the
wettest, and its s\* is **+0.050, positive**. The claim "2nd-driest of 20" is correct but on a
different metric (dry-season supply aggregated over the four over-limit basins C/D/E/K, where
it is indeed rank 2 with a 1.68× span), and the two rankings point opposite ways:

| metric | solved member rank (1 = driest) |
|---|---|
| four over-limit basins C,D,E,K | **2 / 20** |
| national dry-season total | 6 / 20 |
| Hai (C) alone | **16 / 20** |

### And the cost effect does not need to be inferred from those rankings — it is measured

I first wrote here that the solved member is "close to worst-case for cost", reasoning from its
rank on the four over-limit basins. That inference is wrong, and it was unnecessary: all four
members are solved as pairs, so the effect can simply be computed on each.

| member | Hai s\* (full capture, 2030) | effect % | certified interval | vs floor |
|---|---|---|---|---|
| cwatm \| gfdl \| ssp370 | +0.014 | 1.172 | [+0.414, +2.081] | 3.17× |
| **cwatm \| gfdl \| ssp126** (headline) | **+0.050** | **1.340** | [+0.639, +2.170] | 3.62× |
| wgap \| gfdl \| ssp370 | −0.627 | 1.506 | [+0.601, +2.410] | 4.07× |
| wgap \| gfdl \| ssp126 | −0.433 | 1.605 | [+0.635, +2.553] | 4.33× |

This is a stronger robustness statement than anything currently in the paper:

* The effect spans **1.17–1.60%** across four hydrology members. Every one is resolved against
  the degeneracy floor (3.2–4.3×) and **every certified interval excludes zero**. The headline
  member sits in the middle of that range, not at an extreme.
* **CORRECTION.** I first wrote that the effect is larger in members where the Hai goes
  negative, reading a pattern off the four gfdl members. Two further members falsify it:
  `cwatm|ipsl|ssp126` has a **negative** Hai s\* (−0.029) and the **lowest** effect (1.011%).
  What actually separates the members is the **hydrology model** — WaterGAP2 gives 1.51–1.61%,
  CWatM 1.01–1.34% — not the sign of the Hai's s\*. The cost result and the scarcity result are
  therefore two separate observations, and must not be presented as corroborating each other.
* The Hai scarcity finding remains *conservative*: the solved headline member is one of the
  seven of twenty where capture does **not** push s\* negative at all.

None of this is currently stated. "One hydrology member is solved" understates the evidence
base by a factor of four and hides the fact that the headline is already ensemble-robust.

Four more solves are queued to fix the remaining real weakness, which is not the member count
but the **GCM** count: all four solved members are gfdl-esm4, so the reported spread is a
hydrology-model and SSP spread with one GCM inside it. Added `cwatm|ipsl-cm6a-lr|ssp126`
(Hai s\* = −0.029, just below the sign change) and `cwatm|mpi-esm1-2-hr|ssp370` (+0.293, the
wettest Hai), each as a control/treatment pair — six solved members, three GCMs, straddling
s\* = 0.

## Corrections to my own earlier statements this round

* I briefed the review panel that the water constraint drives *biomass co-firing* conversion
  127.1 → 354.7 GW. That is the **dry-cooling condenser retrofit**; biomass moves 5.6 → 14.2 GW
  and is the *payment* for the retrofit, not the adaptation. The storyline had it right. Both
  affected agents were corrected mid-run.
* I briefly read a 4-member backup file as the live water input and treated the 20-member
  ensemble as suspect. The live `inputs/water_availability.csv` carries all 20 members on the
  corrected basis (Hai 2.254× the superseded value, exactly the documented factor). No defect.
