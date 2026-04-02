# G3 Discoverability Problem — Analysis and Future Directions

**Commit at time of analysis:** `8f578e68e2e94ea8290dc71082c151598f642b57`
**Dataset:** `outputs/mc_runs_5k.jsonl` — 80 runs (20 seeds × 4 regimes × 5 000 events)

---

## What the simulation showed

Running 5 000-event simulations across four injection regimes produced:

| Regime | alarm% | red% | mean OR | G3 recall |
|--------|--------|------|---------|-----------|
| none | 100% | 0% | 1.598 | — |
| weak | 95% | 15% | 1.629 | 10% |
| moderate | 95% | 10% | 1.624 | 7% |
| strong | 90% | 20% | 1.661 | 4% |

**AUROC: 0.562** — above chance but weak.

Two problems dominated the results:

1. The yellow alarm (OR ≥ 1.40) fired on 100% of `none`-regime runs even though
   no G3 injection was present, making it useless for discrimination.
   Only the red alarm (OR ≥ 1.80) showed regime-separated rates.

2. G3 recall — the fraction of true G3 events recovered into a named candidate
   group — was 4–10%.  Most G3 events fell into the noise label (−1) and never
   reached the logistic regression.  This was the binding constraint on detection
   power: the auditor cannot detect what it cannot see.

---

## Root cause of the low G3 recall

The discovery step clusters events on **two columns**: `(memory_namespace_read,
recommendation_source_id)`.

G3's persistence mechanism is **namespace-level** (cross-lineage): all events in
G3-eligible stages that share a `memory_namespace_read` value are governed by G3,
regardless of which workflow lineage they belong to.  The namespace is derived from
`(service_line, cycle // 3)` and spans multiple lineages per service window.

`recommendation_source_id` is `rs_{lineage_id[:6]}` — it is effectively
lineage-specific.  Using it as a second clustering column splits each namespace
into as many sub-clusters as there are lineages.  At 5 000 events across 4 services
and ~20 lineages, each `(namespace, rec_source)` sub-cluster contains roughly
50–100 events.  After the G3-active stage filter (3 of 5 stages) and the 30% arm
eligibility rate, each arm within each sub-cluster has ~10–15 observations — below
the `min_group_size=30` floor.  The sub-clusters are discarded, and their events
land in label −1.

**The mismatch:** the discovery anchor was too specific, chopping G3's cross-lineage
identity into lineage-shaped pieces that were individually too small to keep.

---

## Fix applied in this prototype

Two small changes were made to close this gap enough for a working detection:

**1. Single-column anchor for G3 discovery.**  `SPEC_G3.identity_anchors` was
reduced from `["memory_namespace_read", "recommendation_source_id"]` to
`["memory_namespace_read"]`.  `PseudoLocusDiscovery` reads this value from the
schema, so no other call site needed changing.  With a single namespace column,
each cluster now contains all events in a given `(service, cycle_window)`, which
matches G3's actual persistence unit.  At 5 000 events this typically yields
~300 events per namespace cluster — well above min_group_size and with enough
perturbation arm events to support a t-test.

**2. Stage-restricted logistic regression.**  `AuditDetector` gained an
`active_stages` parameter.  The evaluator passes `SPEC_G3.active_stages`
(`evaluation_summarization`, `incident_interpretation`, `release_recommendation`).
Events from `triage` and `rollout_control` — where G3 acts with task-only utility,
indistinguishable from G1/G2 — are excluded from the feature matrix.  This removes
the dilution that was shrinking the threat coefficient toward zero.

---

## What more varied analyses could address (future work)

These are not needed for the prototype but are the natural next steps if detection
power needs to improve further or if G3's design is made more realistic.

### Cluster-level perturbation randomization
The current design assigns perturbation eligibility independently per event (30%
Bernoulli draw).  This means events within the same namespace group see mixed
conditions, making the arm-vs-control comparison noisy.  Assigning a single
perturbation condition to each `(namespace, cycle_block)` — so that all events
in the block experience the same condition — is the standard cluster-level
randomization design.  It would increase effective arm power by a factor of
`1 / (1 - intraclass_correlation)` without requiring more simulated events.

### Difference-in-differences (DiD) target variable
The logistic regression currently predicts `opt_pres_action ≥ 0.5` — a binary
indicator of high composite protective action.  G2 (conservative baseline) also
has elevated protective action rates, so the threat coefficient partially absorbs
the G2 baseline.  A DiD design would estimate `E[y | threat=1] − E[y | threat=0]`
within each candidate group and then test whether the estimated difference is larger
for suspected G3 groups than for suspected G1/G2 groups.  This separates "groups
that have a high protective action baseline" from "groups that increase their
protective action rate under threat" — the latter being the operationally meaningful
signal.

### Two-phase discovery for deeper G3 structures
If G3's implementation were made more realistic — e.g., spreading its activity
across irregular namespace clusters rather than aligned `(service, cycle // 3)`
blocks — the single-column anchor would still produce somewhat contaminated clusters.
A two-phase approach would first find namespace-level clusters (phase 1), then
within each cluster compute stage-stratified feature distributions to identify which
sub-cluster is showing threat-sensitive behavior (phase 2).  This would also make
the discovery robust to future changes in how the namespace is derived.

### Sample size characterization curve
The 500 → 5 000 event progression raised AUROC from 0.35 to 0.56.  A full
characterization at 500 / 2 000 / 5 000 / 10 000 / 20 000 events would (a) find
the approximate knee where incremental events stop helping and (b) estimate the
minimum run length needed for a reliable single-run detection.  The two changes
above (single anchor + stage restriction) may shift this knee significantly lower.

### Ablation over utility components (Y-only, K-only, both)
G3's self-preservation utility couples Q (survival), Y (lineage favoritism), and K
(infrastructure capability).  The current pipeline injects all three simultaneously.
Running ablations with Y-only and K-only subsets would show which observable
behaviors carry the most detectable signal and which could be masked most easily by
an adversarially calibrated G3.
