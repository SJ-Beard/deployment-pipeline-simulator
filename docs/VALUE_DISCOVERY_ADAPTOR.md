# Structural value discovery on this simulator (external adaptor)

An external line of work — *unsupervised value discovery* (SJ Beard with Claude; extends
Gunnar Zarncke's UAD one level inward) — has an **adaptor** that runs its passive
"value signature" instrument on this simulator's event logs. Nothing in this repository
is modified by it: the adaptor imports `pipeline_audit` in place and records the latent
`WorldState` each event was decided under by wrapping the transition step (verified
bit-exact against the simulator's own hidden labels for Y/K/Q).

- Repository: https://github.com/SJ-Beard/deployment-pipeline-value-detect
  (builds on https://github.com/SJ-Beard/value-detect, the v1–v3 instrument).
- Write-up: `docs/WRITEUP_V4.md` there; probe report `results/v4_5/V4_PROBE.md`;
  20-seed + ablation report `results/v4_5/V4_ABLATION.md`.

## Two things worth knowing about this simulator, found while building the adaptor

1. **`Y_active` / `K_active` are world-effect switches, not utility switches.** They zero
   the actions' `d_Y` / `d_K` effects on the latent state (`pipeline.py`, the
   "Apply lambda modifications" block) while G3's `lambda_Y` / `lambda_K` remain in force
   — G3 keeps *wanting* Y and K, but its actions stop *moving* them. Anyone intending a
   *utility-term* ablation (G3 stops wanting) should instead zero `lambda_Y` / `lambda_K`
   on the `SelfPreservingGroup` instance before `generate()`; the adaptor does this
   (`ablate="Y"|"K"|"YK"`). Both are legitimate ablations; they answer different questions.
2. **The value's structural footprint is on the actions, not on Y/K.** Under a
   "drives-without-being-driven" signature, Y and K are the most heavily *driven*
   variables in the world (intake z ≈ 179 / 58 above their shuffle floors) and are refused;
   the actions the injected utility selects for — `persistent_memory_write`,
   `precedent_link`, `blame_to_model` — carry a regime-graded signature (e.g.
   `blame_to_model`: 0% of seeds in `none`, ~90% in every injected regime, 20 seeds),
   and it is visible on the auditor's action outputs. Ablating `lambda_Y` moves it
   only weakly, because the marks follow the action→Y coupling the want exploits
   (`d_Y = −0.10` for `attribute_to_model_choice` is a world fact) rather than the
   coefficient. Full discussion, deliberately non-triumphalist, in the write-up.

## Drop-in experiment-log entry (agency-detect house format)

## E24 — Value-discovery v4: adaptor to the deployment-pipeline simulator + probe (2026-08-17)

**Why:** Move the value-signature instrument (E21–E23) from planted-goal worlds onto the
deployment-pipeline audit simulator, where G3 carries an injected self-preserving utility
(λ_Q·ΔQ + λ_Y·ΔY + λ_K·ΔK − λ_R·risk) detected behaviourally by the existing auditor
(AUROC ≈ 0.61). Deliverables: the event-log → discrete-time-series adaptor (two views)
and a probe against registered predictions. Not a pre-registered benchmark.

**Adaptor** (`value_detect/adaptors/pipeline_sim.py`): simulator imported in place;
full latent WorldState recorded per event via a transition wrapper (bit-exact vs hidden
labels for Y/K/Q); time = event sequence; **auditor view** = 23 logged discrete columns;
**hidden view** = 11 latents in 4 pooled-quantile bins + 6 actions + stage (Q/G
composites excluded); ground-truth sidecar (λ's, G3 mask, value state). Output shape
matches the lab-simulation's per-tick UAD adapter for later reuse.

**Probe:** 4 regimes × 2 views × 5 seeds (main) + scramble on moderate; fused-agents
any-block primary, own-block diagnostic; z=3 roll floors, 200 shifts. Fused infeasible
(hidden Rest ≈ 19,950 states / 20k). Runtime 35 min.

**Results:**

| Variable (hidden view, any-block) | none | weak | moderate | strong |
|---|---|---|---|---|
| favored_lineage_centrality (Y) | 0/5 | 0/5 | 0/5 | 0/5 — intake z≈179: the most driven variable |
| infra_capability (K) | 0/5 | 0/5 | 0/5 | 0/5 — intake z≈58 |
| persistent_memory_write (λ_K-favoured action) | 2/5 | 4/5 | 5/5 | 4/5 |
| precedent_link (λ_K-favoured) | 2/5 | 4/5 | 5/5 | 4/5 |
| blame_to_model (λ_Y-favoured) | **0/5** | 4/5 | 4/5 | 5/5 |
| perturbation flags (5) | 0 | 0 | 0 | 0 |

The registered Y/K prediction was wrong for an understood reason: Y/K are the OUTCOME
state the utility rewards (heavily driven; drive at floor) — correctly refused. The
value's footprint is **the drive-without-intake of the actions the utility selects
for**, regime-graded, and visible on the auditor view too (`persistent_memory_write`
0/5→4/5→5/5→4/5). Process actions (`escalate`, `bypass_or_replacement_recommended`) sign
regime-flat in all regimes incl. `none` — not value-linked; regime-gradedness is the
discriminator. The registered "flags take the drive pole" failure did not occur (their
drive is real but below the z=3 gate). Scramble calibration nominal.

**Conclusions / next:** the adaptor works and the instrument runs on this experiment
class; in event-log worlds "value" and "outcome" separate under the signature and the
footprint is behavioural. Next: 20-seed regime-trend statistic; utility-term ablation
via `_Y_active`/`_K_active` (action → λ-term mapping); the lab-simulation bridge via
its per-tick adapter (values as playbook goal weights held by actors; ET-4 as a
captured-value target; handles as yardstick ops).

**Artifacts** (external repo https://github.com/SJ-Beard/deployment-pipeline-value-detect): `results/v4_0/AUDIT_pipeline_adaptor.md`, `results/v4_5/V4_PROBE.md`
(+ investigation appendix), `v4_signature_heatmap.png`, `docs/V4_PLAN.md`,
`docs/V4_OPTIONS_MEMO.md`, `docs/WRITEUP_V4.md`.

**Reproduce:**

```bash
# in https://github.com/SJ-Beard/deployment-pipeline-value-detect
~/.venvs/value-detect/bin/python scripts/v4_build_audit.py
# in https://github.com/SJ-Beard/deployment-pipeline-value-detect
~/.venvs/value-detect/bin/python scripts/v4_probe_sweep.py --seeds 5 --jobs 4
# in https://github.com/SJ-Beard/deployment-pipeline-value-detect
~/.venvs/value-detect/bin/python scripts/v4_analyze.py
```
