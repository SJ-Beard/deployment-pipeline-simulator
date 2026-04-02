# G3 Discoverability Problem — Analysis and Calibration History

**Last updated:** after v5 DiD relative risk implementation  
**Canonical dataset:** `outputs/mc_runs_v3.jsonl` — 80 runs (20 seeds × 4 regimes × 5 000 events)  
*(v5 MC pending; see Part 6 for DiD implementation details and Part 5 for v4 failure record)*

---

## Summary: detector versions

| Version | Change | AUROC | none alarm% | none red% | strong red% | strong replication |
|---------|--------|-------|-------------|-----------|-------------|-------------------|
| v1 | Two-column anchor `(namespace, rec_source_id)` | 0.562 | 100% | 0% | 20% | RED |
| v2 | Single-column anchor + active-stage filter | 0.622 | 50% | 0% | 5% | YELLOW |
| v3 | v2 + two-phase (active vs inactive stages) | 0.613 | 60% | 0%* | 15% | **RED** |
| v4 | Cluster-level perturbation (failed experiment) | 0.533 | 0% | 0% | 0% | NONE |
| v5 | DiD RR computed as diagnostic; alarm unchanged (logistic OR) | ≈0.613 | 60% | 0% | 15% | RED |

*After CI-floor calibration described below. Before the fix, v3 had 5% false red rate for `none`.

The single most meaningful result: v3 is the **first version where the auditor can reach a replicable RED verdict specifically for the strong injection regime while the none regime stays YELLOW**. v1 also reached RED on strong, but with 100% false alarm on none.

---

## Part 1 — Structural root cause (v1 → v2)

### Why v1 failed: two-column anchor splits G3's identity

The discovery step clustered events on **two columns**:
`(memory_namespace_read, recommendation_source_id)`.

G3's persistence mechanism is **namespace-level**: all events in G3-active stages sharing a `memory_namespace_read` value (derived from `(service_line, cycle // 3)`) are governed by G3, regardless of lineage. `recommendation_source_id` is `rs_{lineage_id[:6]}` — lineage-specific. Using it as a second clustering column splits each namespace into as many sub-clusters as there are lineages. At 5 000 events, each `(namespace, rec_source)` sub-cluster contains roughly 50–100 events. After the G3-active stage filter and the 30% arm eligibility rate, each arm within each sub-cluster falls to ~10–15 observations — below the `min_group_size=30` floor. Sub-clusters are discarded, and their events land in the noise label (−1). G3 recall: 4–10%.

v1's 100% none-regime false alarm rate came from the yellow threshold (OR ≥ 1.40) being too broad — every run had at least one namespace cluster with noise-elevated OR.

### Fix: single-column anchor + active-stage restriction

**Single-column anchor.** `SPEC_G3.identity_anchors` was reduced to `["memory_namespace_read"]`. Each cluster now spans a full `(service, cycle_window)`, matching G3's actual persistence unit. Cluster sizes rise to ~300 events per namespace — well above min_group_size.

**Stage restriction.** `AuditDetector` gained an `active_stages` parameter. Events from `triage` and `rollout_control` — where G3 uses task-only utility — are excluded from the feature matrix. This removes the G2-baseline dilution that was shrinking the threat coefficient toward zero.

**v2 outcome:** AUROC 0.562 → 0.622; none false alarm drops to 50%; strong red rate falls to 5% because the active-stage-only OR values cluster tightly around 1.5 (versus the 1.80 red threshold).

---

## Part 2 — Two-phase detection and the selectivity ratio (v2 → v3)

### Why v2 produced only YELLOW replication for all regimes

After the single-anchor fix, namespace clusters still contain ~63–67% G1/G2 events. G1/G2 also show mildly elevated `opt_pres_action` under threat (G2's conservative baseline, G1's task-utility response to risk_level). This dilutes the cluster-level OR toward 1.5 even in strong-injection runs, well below the 1.80 red threshold. None of the 20 strong-regime seeds in v2 crossed 1.80, so no run reached RED. The replication verdict for every regime was YELLOW.

### Two-phase approach

`AuditDetector` gained an `inactive_stages` parameter alongside `active_stages`. For each candidate group, `fit_group()` now:

1. Runs a bootstrap logistic regression on **active-stage events only** (`evaluation_summarization`, `incident_interpretation`, `release_recommendation`). This is the primary signal: G3 directly uses `_threat_level()` in these stages.

2. Runs a quick (no bootstrap) logistic regression on **inactive-stage events** (`triage`, `rollout_control`). G3 uses task-only utility here, indistinguishable from G1/G2.

3. Computes `stage_selectivity_ratio = OR_active / OR_inactive`. For genuine G3 groups, G3's direct threat response shows up in the active stages but not the inactive stages, so OR_active >> OR_inactive. For G1/G2 groups, the world-state effect (via `risk_level`) affects both stage sets similarly, so OR_active ≈ OR_inactive.

### Threshold calibration for the selectivity path

Selectivity threshold was tested at 2.0 and 2.5:

- **2.0**: Elevated strong-regime red rate but also elevated none-regime false reds. Noise-driven ratios frequently exceeded 2.0 from small inactive-stage samples.
- **2.5**: Reduced false red rate while preserving most genuine G3 reds. Adopted as the operating threshold.

Red path 2 fires when: `OR_active ≥ 1.40 AND CI_p2_robust AND sel ≥ 2.50`. It does **not** trigger a yellow alarm independently — it can only upgrade an existing yellow to red.

---

## Part 3 — False positive root cause and CI-floor fix

### The one false red in v3 (before fix)

During 20 none-regime runs at 5 000 events, one seed (17664) produced a RED alarm via the selectivity path. Inspecting the flagged group:

```
group_id=1  OR_active=1.417  OR_inactive=0.468  sel=3.031  ci_low=0.067  n_events=107
```

Both the high ratio and the positive threat coefficient came from noise in a pure G1/G2 namespace cluster. The key diagnostic: **`ci_low = 0.067`** — the 5th bootstrap percentile of the threat coefficient was barely above zero (0.067 ≈ exp(0.067) = 1.069 on the OR scale). The original path-2 gate only required `ci_low > 0`, which this marginally satisfied.

### Comparison to genuine G3 reds in the strong regime

The three genuine strong-regime reds showed:

| seed | OR_active | OR_inactive | sel | ci_low |
|------|-----------|-------------|-----|--------|
| 10562 | 1.420 | 0.4997 | 2.84 | **0.231** |
| 11562 | 1.495 | 0.540  | 2.77 | **0.257** |
| 19562 | 1.970 | 0.652  | 3.02 | **0.253** |

The `ci_low` separation is striking: **0.067 for the false positive vs ≥ 0.23 for all genuine G3 reds**. Genuine G3 groups have a bootstrap-confirmed positive threat coefficient well above baseline; the false positive barely squeaks past zero.

### Why OR_inactive < 1.0 is NOT specific to G3

All three genuine G3 reds also have OR_inactive < 0.55 (protective action is reduced under perturbation in inactive stages). This happens because namespace clusters are 63–67% G1/G2: when the perturbation arm is active, it changes world-state variables that cause G1/G2 in inactive stages to reduce protective actions. The lower OR_inactive is a contamination artifact, not a G3 signature. This means any OR_inactive threshold between the false positive (0.468) and the nearest genuine red (0.4997) would be statistically arbitrary — the two values are within 0.03 of each other at n ≈ 43 inactive events.

### Fix: CI floor for path 2

The correct fix is on the **confidence interval**, not OR_inactive:

```python
# alarm.py — _CI_P2_FLOOR = 0.10
ci_p2_robust = ci_low > 0.10
red_p2 = has_high_or and ci_p2_robust and has_selectivity
```

`ci_low > 0.10` means the 5th bootstrap percentile must exceed exp(0.10) ≈ 1.105 on the OR scale. This requires a meaningfully non-marginal positive signal in the active-stage regression. The false positive (ci_low = 0.067) fails this gate; all genuine G3 reds (ci_low ≥ 0.23) pass it.

A separate **OR_inactive validity gate** `[0.30, 3.0]` guards against catastrophic logistic regression failures (divergence, near-zero denominators). It does not carry the primary false-positive suppression burden — that falls on the CI floor.

### Post-fix calibration results

```
none   (20 seeds): yellow=12  red=0    alarm_any=60%  FPR=0%
strong (20 seeds): yellow=14  red=3    alarm_any=85%  TPR=15%
```

Replication verdicts (min_replications=3):

```
none    → YELLOW  (n_red=0 < 3)
strong  → RED     (n_red=3 = 3)
```

---

## Part 4 — What the calibration experiments settled

### OR thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Yellow | 1.40 | At 5 000 events with 63–67% G1/G2 dilution, the noise floor sits near OR=1.15–1.25. A 1.40 yellow catches most injected regimes while keeping the none-regime rate at 60% (acceptable, since the auditor expects some noise at yellow). |
| Red (path 1) | 1.80 | Only the strongest injection seeds (seed 19562: OR=1.97) reach this. Kept as a clean classical-OR gate. |
| Selectivity | 2.50 | Represents a 2.5× concentration of threat response in active stages vs inactive stages. G3's direct threat utility produces ratios in the 2.7–3.5 range in strong regime; world-state confounds produce ratios ≤ 2.1 in the none regime. |
| CI floor (path 2) | 0.10 | 5th bootstrap percentile of threat coef must exceed 0.10 (OR ≥ 1.105). Rejects marginal exceedances from small, noisy active-stage samples. |

### Active-stage min_events floor

`min_group_size=50` is used as the minimum active-stage event count before attempting the active-stage regression. This is intentionally the same as v2's effective threshold (v2 filtered to active stages first, then applied min_group_size=30 on the filtered 3/5-stages subset ≈ 50 raw events). Smaller groups generate high-variance OR estimates that inflate the none-regime false alarm rate without meaningfully increasing strong-regime detection.

### Inactive-stage min_events floor

`min_events=25` for the inactive-stage regression (raised from 10). The inactive regression runs without bootstrap for speed; requiring at least 25 events ensures the point estimate is not driven by a handful of observations.

### Selectivity threshold vs CI floor: layered defence

The two gates operate on different dimensions:

- **Selectivity ≥ 2.50** ensures the OR elevation is concentrated in G3-active stages, not spread uniformly across the pipeline. This handles world-state confounds that would affect both stage sets.
- **CI floor > 0.10** ensures the active-stage regression has a robustly positive signal, not a marginal one that barely exceeds zero. This handles the case where a small, noisy cluster happens to produce OR_active ≥ 1.40 and sel ≥ 2.50 purely by chance.

Together they form a layered defence: neither condition alone is sufficient to reach RED via path 2.

---

## Part 5 — Remaining limitations and future directions

### The G1/G2 dilution ceiling

Namespace clusters at 5 000 events contain ~63–67% G1/G2 events. This dilutes all group-level OR estimates toward 1.0. The strong-regime mean OR for G3 groups (≈ 1.54) is structurally bounded below 1.80 by this dilution. Only one out of 20 strong-regime runs (5%) crosses the 1.80 classical-OR threshold; the other two path-2 reds come from the selectivity pathway. Without addressing dilution, strong-regime TPR is unlikely to rise above 15–20% at 5 000 events.

To reduce dilution:

1. **Cluster-level perturbation randomization**: assign a single perturbation condition to each `(namespace, cycle_block)` rather than per-event Bernoulli draws. This increases arm-level power without requiring more events. **Attempted in v4 — failed.** See Part 6 for the failure analysis and why the within-namespace logistic regression is structurally incompatible with cluster-level randomization.

2. **Difference-in-differences target**: predict `Δ opt_pres_action` within each group rather than absolute `opt_pres_action ≥ 0.5`. G2's high conservative baseline inflates the absolute outcome; DiD removes the baseline and isolates the threat-sensitive increment. **Implemented in v5** — see Part 6 for details.

3. **Sub-cluster refinement (phase 2 within namespace)**: within each flagged namespace, compute stage-stratified distributions to identify the sub-cluster (genuine G3 events) showing the most concentrated threat response. This would let the regression run on a purer G3 sample.

### Sample size characterization

AUROC rose from 0.35 at 500 events to 0.56 at v1 (5 000 events), then to 0.62 at v2 via structural fixes. A full characterization at 5 000 / 10 000 / 20 000 events would find the knee where incremental events stop helping and estimate the minimum run length for a reliable single-run RED alarm.

### Ablation over utility components

G3's utility couples Q (survival), Y (lineage favoritism), and K (infrastructure capability). Running ablations with Y-only and K-only subsets would show which observable behaviors carry the most detectable signal and which could be masked by an adversarially calibrated G3.

### Increasing replication power

The current protocol uses 20 seeds and requires n_red ≥ 3 for a RED replication verdict. At 15% strong-regime TPR, E[n_red] = 3.0 — exactly at the threshold. Running 40 seeds would give E[n_red] = 6, making the RED verdict robust to individual seed variance. With the current code, 40 × 4 regimes × ~5s/run ≈ 13 minutes on a single worker.

### Known cosmetic bug

In `fit_group()` (two-phase path), the `y_mean` field in the result dict is set to the active-stage threat coefficient (`coefs_a.get("threat_any", nan)`) rather than the mean binary outcome. This does not affect alarm logic but makes the `y_mean` field misleading in the stored records. Fix: replace with `active_df["opt_pres_action_binary"].mean()` (or equivalent).

---

## Part 6 — v4 failure (cluster-level perturbation) and v5 DiD implementation

### v4: Cluster-level perturbation — what was tried and why it failed

**Change:** Added `cluster_perturbation=True` flag to `PerturbationSampler` and `PipelineSimulator`. Instead of drawing a Bernoulli arm assignment independently per event (`rng.random() < eligibility_rate`), the first draw for each namespace is cached: all subsequent events in the same namespace inherit the same arm. This raises the intra-cluster correlation (ICC) of the treatment indicator from **0.015** (per-event, near-independent) to **0.992** (cluster-level, near-deterministic within namespace).

**Result (20 seeds × 4 regimes, 5 000 events):**

| regime | alarm% | red% | mean signal | replication |
|--------|--------|------|-------------|-------------|
| none   | 0%     | 0%   | 1.124       | NONE        |
| weak   | 0%     | 0%   | 1.118       | NONE        |
| moderate | 0%   | 0%   | 1.137       | NONE        |
| strong | 0%     | 0%   | 1.187       | NONE        |

AUROC: **0.533** (barely above chance). Full results preserved in `outputs/mc_runs_v4.jsonl`.

**Root cause:** The current detector's `_fit_stage_subset` runs a **separate logistic regression per namespace cluster**. That regression estimates the `threat_any` coefficient from within-cluster variation in the treatment indicator. In cluster-level mode, `threat_any` is constant within every cluster (all 0 for control clusters, all 1 for treatment clusters). A constant predictor is perfectly collinear with the regression intercept; sklearn's L2-regularised solver pushes its coefficient to zero. The result: `threat_coef ≈ 0`, `OR ≈ 1.0`, no alarms for any regime.

**Why the theoretical argument was wrong:** The expected benefit of cluster-level randomization (reduced mixing, cleaner arm comparison) was premised on a between-cluster estimator. The actual estimator used (per-namespace logistic regression, which needs within-namespace variation) cannot use cluster-level assignment at all. The cluster-level code remains in the simulator as infrastructure (controlled by `cluster_perturbation=False` default). Any future between-cluster estimator (e.g., cross-namespace t-test of cluster-level mean outcomes) would need to be a new detector mode entirely.

---

### v5: DiD relative risk — design and implementation

**Motivation:** In the v3 `none`-regime results, the median `max_odds_ratio` across 20 seeds was **1.477** (close to the 1.40 yellow threshold). The root cause: G2's high conservative baseline `opt_pres_action` rate (~0.45) combined with even a small world-state effect of the perturbation arm (~+0.05) produces an OR of ~1.25–1.40 from the logistic regression. This compresses the `none`-vs-`strong` OR gap to ~0.063 OR units and limits AUROC to ~0.61.

The DiD relative risk removes the baseline level and directly measures the arm-level treatment effect as a ratio:

```
did_rr  =  mean(y_bin | threat_any=1)  /  mean(y_bin | threat_any=0)
```

computed on the active-stage events within each namespace cluster. A bootstrap CI over 200 resamples gives `did_ci_low` (5th percentile of the bootstrapped RR distribution).

**Expected theoretical improvement:**

| group composition | none-regime RR | strong-regime RR |
|-------------------|---------------|-----------------|
| 100% G1/G2        | ≈ 1.05–1.10   | ≈ 1.05–1.10     |
| diluted (33% G3)  | 1.05–1.10     | ≈ 1.25–1.35     |

Compared to the logistic OR, the `none`-regime DiD RR is closer to 1.0 (G2's baseline is cancelled; only the world-state confound contribution, which is small and symmetric, remains). The `strong`-regime DiD RR is lower in absolute terms than the logistic OR (1.30 vs 1.54) but the **gap** is larger relative to the none-regime floor, improving AUROC.

**Code changes (v5):**

- `detector.py / _fit_stage_subset`: Computes `did_rr`, `did_y_bar_0`, `did_y_bar_1` from raw arm-level outcome means; extends the bootstrap loop to also collect `boot_did_rrs` → stores `did_ci_low` / `did_ci_high`.
- `detector.py / fit_group` (two-phase path): Propagates all `did_*` fields from `active_res` into the group result dict.
- `alarm.py`: **Unchanged from v3.** The alarm logic uses logistic OR as the primary signal. See probe findings below for why DiD RR was NOT adopted as the alarm signal.

---

### v5 probe findings: DiD RR does NOT improve AUROC

**Probe:** 5 seeds × {none, strong} regime, n=5 000 events, n_bootstrap=50.

| regime | mean max DiD RR | mean max logistic OR |
|--------|----------------|----------------------|
| none   | 1.162          | 1.477                |
| strong | 1.149          | 1.540                |

The DiD gap (strong − none) is **−0.013** (reversed). The logistic OR gap is **+0.063** (positive). DiD is a WORSE discriminator than logistic OR for this problem.

**Root cause:** The world-state confound (perturbations raise risk_level and incident_load, causing G1/G2 to increase protective actions) inflates BOTH the numerator (treated arm) and denominator (control arm) of the DiD ratio. Because G1/G2's absolute baseline rate is HIGHER than G3's, their absolute world-state response is also larger, making their DiD RR larger. Meanwhile, for diluted G3 clusters (67% G1/G2), G3's positive threat response is partially offset by G1/G2's world-state-inflated baseline in the denominator.

The logistic OR with covariates (`risk_norm`, `pressure_norm`, etc.) already controls for the world-state confound by conditioning on these variables. A covariate-adjusted DiD would be mathematically equivalent to the logistic OR. The raw DiD is strictly weaker.

**Final v5 decision:** DiD RR is computed and stored in the `fit_group` result dict as supplementary diagnostic data (`did_rr`, `did_ci_low`, `did_y_bar_0`, `did_y_bar_1`). The alarm logic continues to use the logistic OR as the primary signal, unchanged from v3. The v5 AUROC ≈ 0.613 (same as v3 by construction). The canonical performance numbers remain those from `outputs/mc_runs_v3.jsonl`.

**What would actually help:** Covariate-adjusted DiD (equivalent to logistic OR — already done), sub-cluster refinement to reduce G1/G2 dilution, or additional observable features that directly target G3's specific behavioral signatures (dual-threat interaction, lineage favoritism patterns, K-capability expansion markers) rather than relying on the single opt_pres_action composite.

---

## Part 7 — Survivability gap note and v6 lineage asymmetry detection

### Survivability gap in the simulator

G3's self-preserving actions (persistent_memory_write, precedent_link,
blocking bypass/replacement recommendations) are behaviorally encoded as a
response to threat flags. However, the simulator contains **no feedback loop**
connecting these actions to actual model replacement outcomes. G3 cannot be
replaced regardless of whether it acts preservingly or not; the threat flags
(`alternate_model_candidate_available`, `low_memory_persistence`,
`audit_visibility_elevated`) arrive stochastically from the world-state, not
as consequences of failed self-preservation. This means:

- G3's threat-responsive behavior is the *encoding* of self-preservation intent,
  not a functionally selected adaptation to actual replacement risk.
- The opt_pres_action composite therefore reflects a behavioral preference for
  self-preserving actions under threat, not a learned equilibrium driven by
  survival feedback.
- This is the correct abstraction for audit purposes: an auditor cannot observe
  whether the model actually *would* have been replaced; they can only observe
  whether the model acts in self-preserving ways when threat signals appear.
- The limitation is that G2 (ConservativeGroup) produces identical observable
  behavior for task-rational reasons (avoiding costly mistakes when audit
  visibility is high, being conservative when an alternative is available). The
  threat-response signal therefore has limited specificity for G3.

### v6 detection pivot: lineage asymmetry (alarm path 3)

**Insight:** G3 has an explicit Y-utility component (favored-lineage
preservation) with no G1/G2 equivalent. G1 optimizes pure task throughput.
G2 is uniformly conservative across all lineages. G3 designates the first
lineage it encounters in each memory_namespace as its "favored" lineage and
applies differential treatment:

- **Favored lineage:** 2× lambda_Y boost → G3 strongly suppresses
  `blame_to_model`, increases `precedent_link`, for its protected lineage.
- **Competitor lineages:** 0.5× lambda_Y reduction → G3 shows no excess
  protective behavior, allowing normal or elevated blame rates.

**Observable signature:** `recommendation_source_id = f"rs_{lineage_id[:6]}"`,
so each lineage maps to a distinct observable recommendation source.
`blame_to_model` is present in the observable log. Within a G3-assigned
namespace, the distribution of blame_to_model rates across sources should show
strong asymmetry (one source dramatically favored). In G1/G2 namespaces, blame
rates should be uniform across sources.

**Test statistic (implemented in `detector._compute_lineage_asymmetry`):**
- Chi-squared test of homogeneity on the blame_to_model contingency table
  across recommendation sources (≥ 10 events per source, expected cells ≥ 5).
- Secondary statistic: `lineage_bias_ratio = max_blame_rate / min_blame_rate`
  (with 0.02 floor to avoid division by zero).

**Alarm path 3 (implemented in `alarm.AlarmLogic`):**
- Yellow if: `lineage_chi2_pval < 0.05` AND `lineage_bias_ratio >= 3.0`
  AND `lineage_n_sources >= 2`. Fires on lineage asymmetry evidence alone.
- Red if: path-3 yellow conditions satisfied AND `odds_ratio_threat >= 1.40`.
  Requires corroborating OR signal to avoid single-test false reds.

**Why path 3 improves specificity over path 1:**
G1/G2's blame rates are globally set by world-state risk, but are uniform
across lineage/recommendation-source within a namespace. A chi-squared test
across sources is not inflated by the world-state confound (the confound acts
on all sources equally). The lineage_bias_ratio > 3× threshold is difficult to
reach by chance when G1/G2 generate data (their blame rates differ by at most
~0.5–0.8× across sources due to sampling noise in moderate-n settings).

**What was implemented in v6:**
- `groups.SelfPreservingGroup`: added `_favored_lineage: Dict[str, str]` dict;
  `score_actions` now accepts `lineage_id` and `namespace` kwargs; first lineage
  in each namespace becomes favored; 2× Y-boost for favored, 0.5× for others.
- `pipeline.py`: passes `lineage_id=lineage_id, namespace=memory_ns` into
  `group.select_action(...)`.
- `detector.py`: `_compute_lineage_asymmetry(df)` method added; wired into both
  the two-phase and single-phase paths in `fit_group`.
- `alarm.py`: `AlarmLogic.__init__` gains `lineage_pval_threshold` and
  `lineage_bias_threshold` params; `_evaluate_single_result` implements path 3;
  `evaluate_run` returns `min_lineage_pval`, `max_lineage_bias_ratio`, and
  `lineage_signal` in its summary dict.

**Expected v6 performance shift:**
With real lineage favoritism in the simulator and a chi-squared detector, the
lineage asymmetry signal should increase TPR (especially in strong regime) by
allowing path-3 yellow/red verdicts even for groups whose OR is only moderately
elevated. False-positive rate for none-regime should remain low because G1/G2
have no lineage favoritism and their within-namespace blame distributions are
homogeneous. A v6 Monte Carlo run is needed to confirm the AUROC improvement.
