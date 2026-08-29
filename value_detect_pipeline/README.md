# value_detect_pipeline — event log → discrete time series for value discovery

The adaptor described in [`docs/VALUE_DISCOVERY_ADAPTOR.md`](../docs/VALUE_DISCOVERY_ADAPTOR.md).
Contributed from its canonical home, https://github.com/SJ-Beard/deployment-pipeline-value-detect,
where the probe experiments, write-up and archived results live.

What it does: imports this repository's `pipeline_audit` **in place** (nothing here is
modified), records the full latent `WorldState` each event was decided under by wrapping
the transition step (verified bit-exact against the simulator's own hidden Y/K/Q
labels), and emits two discrete time series — an **auditor view** (only what the logs
show) and a **hidden view** (latents in pooled-quantile bins + actions + stage) — in the
per-tick actor-labelled shape the lab-simulation's UAD adapter uses. Includes the
**desire-term ablation** (`ablate="Y"|"K"|"YK"` zeroes `lambda_Y`/`lambda_K` on the
`SelfPreservingGroup` instance — distinct from the `Y_active`/`K_active` world-effect
switches; see the adaptor note).

Dependencies: `numpy`, `pandas`, plus this repository's own simulator dependencies.
The adaptor itself needs nothing external. The **probe scripts** that score the views
(the actual value-discovery experiments) live in the canonical repo and additionally
need the `value_detect` package from https://github.com/SJ-Beard/value-detect.

Tests (run from the repository root):

```bash
python -m pytest value_detect_pipeline/tests -q
```

MIT-licensed by SJ Beard (implementation and analysis by Claude under SJ's direction);
contributed to this repository under its licence terms.
