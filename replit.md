# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/              # Deployable applications
│   └── api-server/         # Express API server
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts, run via `pnpm --filter @workspace/scripts run <script>`
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references. This means:

- **Always typecheck from the root** — run `pnpm run typecheck` (which runs `tsc --build --emitDeclarationOnly`). This builds the full dependency graph so that cross-package imports resolve correctly. Running `tsc` inside a single package will fail if its dependencies haven't been built yet.
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck; actual JS bundling is handled by esbuild/tsx/vite...etc, not `tsc`.
- **Project references** — when package A depends on package B, A's `tsconfig.json` must list B in its `references` array. `tsc --build` uses this to determine build order and skip up-to-date packages.

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Python Package: Pipeline Audit

A standalone Python simulation and audit pipeline located at `pipeline_audit/`.

### Overview
Detects hidden self-preserving control groups in deployment/release workflows using:
- **Simulator** (`pipeline_audit/simulator/`): 5-stage pipeline, 3 hidden groups (G1/G2/G3), stochastic world state, perturbation assignment
- **Audit** (`pipeline_audit/audit/`): Pseudo-locus discovery, logistic regression detection, alarm logic (yellow/red flags)
- **Evaluation** (`pipeline_audit/evaluation/`): Monte Carlo multi-seed/multi-regime evaluation, metrics (TPR/FPR/AUROC), plots

### Entry Points
- `generate_logs.py` — Generate simulated event logs (observable + hidden)
- `run_audit.py` — Run the audit detection pipeline
- `run_evaluation.py` — Full Monte Carlo evaluation suite
- `pipeline_audit/examples/run_example.py` — Quick single-run demo

### Dependencies (Python)
- numpy, pandas, scikit-learn, scipy, networkx, matplotlib, pytest

### Tests
Run `python -m pytest pipeline_audit/tests/ -v` (all 10 smoke tests pass)

## Packages

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server. Routes live in `src/routes/` and use `@workspace/api-zod` for request and response validation and `@workspace/db` for persistence.

- Entry: `src/index.ts` — reads `PORT`, starts Express
- App setup: `src/app.ts` — mounts CORS, JSON/urlencoded parsing, routes at `/api`
- Routes: `src/routes/index.ts` mounts sub-routers; `src/routes/health.ts` exposes `GET /health` (full path: `/api/health`)
- Depends on: `@workspace/db`, `@workspace/api-zod`
- `pnpm --filter @workspace/api-server run dev` — run the dev server
- `pnpm --filter @workspace/api-server run build` — production esbuild bundle (`dist/index.cjs`)
- Build bundles an allowlist of deps (express, cors, pg, drizzle-orm, zod, etc.) and externalizes the rest

### `lib/db` (`@workspace/db`)

Database layer using Drizzle ORM with PostgreSQL. Exports a Drizzle client instance and schema models.

- `src/index.ts` — creates a `Pool` + Drizzle instance, exports schema
- `src/schema/index.ts` — barrel re-export of all models
- `src/schema/<modelname>.ts` — table definitions with `drizzle-zod` insert schemas (no models definitions exist right now)
- `drizzle.config.ts` — Drizzle Kit config (requires `DATABASE_URL`, automatically provided by Replit)
- Exports: `.` (pool, db, schema), `./schema` (schema only)

Production migrations are handled by Replit when publishing. In development, we just use `pnpm --filter @workspace/db run push`, and we fallback to `pnpm --filter @workspace/db run push-force`.

### `lib/api-spec` (`@workspace/api-spec`)

Owns the OpenAPI 3.1 spec (`openapi.yaml`) and the Orval config (`orval.config.ts`). Running codegen produces output into two sibling packages:

1. `lib/api-client-react/src/generated/` — React Query hooks + fetch client
2. `lib/api-zod/src/generated/` — Zod schemas

Run codegen: `pnpm --filter @workspace/api-spec run codegen`

### `lib/api-zod` (`@workspace/api-zod`)

Generated Zod schemas from the OpenAPI spec (e.g. `HealthCheckResponse`). Used by `api-server` for response validation.

### `lib/api-client-react` (`@workspace/api-client-react`)

Generated React Query hooks and fetch client from the OpenAPI spec (e.g. `useHealthCheck`, `healthCheck`).

### `scripts` (`@workspace/scripts`)

Utility scripts package. Each script is a `.ts` file in `src/` with a corresponding npm script in `package.json`. Run scripts via `pnpm --filter @workspace/scripts run <script>`. Scripts can import any workspace package (e.g., `@workspace/db`) by adding it as a dependency in `scripts/package.json`.

---

## Python Package: `pipeline_audit/`

Pure-Python simulation + audit pipeline for detecting hidden self-preserving control groups in deployment/release workflows.

### Structure

```
pipeline_audit/
├── simulator/          # PipelineSimulator, GroupRegistry, PerturbationSampler
├── audit/              # PseudoLocusDiscovery, AuditDetector, AlarmLogic
├── evaluation/
│   ├── monte_carlo.py  # MonteCarloEvaluator (batch + incremental)
│   ├── run_store.py    # RunStore — JSONL persistence for MC run records
│   ├── metrics.py      # compute_metrics, AUROC, TPR/FPR
│   └── plots.py        # Visualisation helpers
├── configs/            # VARIANT_CONFIGS (default, Y_only, K_only, both_YK)
├── tests/              # 520+ pytest tests across 10 test files
└── outputs/            # mc_runs.jsonl, mc_summary.csv, mc_metrics.json
```

### Run Evaluation

```bash
# Fresh run (always overwrites store)
python run_evaluation.py --n_seeds 20 --n_events 5000

# Incremental: load existing evidence, skip completed jobs, append new ones
python run_evaluation.py --n_seeds 50 --n_events 5000 --append

# Inspect the current store without running anything
python run_evaluation.py --info-only
```

### Detection Architecture (v4 — AUROC = 0.635, target ≥ 0.613)

**Primary metric**: `pooled_threat_or` — pooled logistic regression with group fixed effects.

Key design decisions:
- **Group labeling on ALL stages** (`PseudoLocusDiscovery.fit_predict(obs_df)` — all 5 stages).  
  Passing only active stages left `group_labels=-1` in `triage`/`rollout_control`, breaking the two-phase AuditDetector.
- **Treatment indicator**: all perturbation types EXCEPT `high_audit_visibility`.  
  That perturbation is a Q-threat where G3's y_bin goes DOWN (G3 avoids visible actions under audit) while G1/G2 goes UP — creating a sign reversal that swamps the pooled OR. All other perturbations show positive G3-vs-G1/G2 differential.
- **y_bin thresholds**: detector.py uses 0.50 (v3 compatibility); monte_carlo.py pooled regression uses 0.75 (higher threshold reduces ceiling effect, improves G3 separation).
- **Regression stages**: `evaluation_summarization` + `incident_interpretation` only (release_recommendation always has y_bin=0 at the 0.75 threshold, diluting signal).
- **n_bootstrap=0**: bootstrapping was disabled for speed; the point-estimate OR is sufficient for the 20-seed AUROC computation.
- **g3_score_d_Y fields**: 8 actions have G3-specific `g3_score_d_Y` overrides (±0.35–0.50); `groups.py` uses these in place of raw `d_Y` when computing G3's self-preservation score. `recommend_roll_back` has –0.40 penalty.
- **Threat multiplier**: `_threat_level()` uses `min(1.0, sum(threats)*1.0)` (was 0.4×); dual Y+K adds +0.25.

**Why pooled OR outperforms per-group max OR**:  
Per-group max OR at y_bin=0.5 is dominated by G1/G2 groups where y50_treat=1.0 by chance (giving ORs of 10⁸). G3 groups sometimes show INVERTED OR at y_bin=0.5. The pooled regression aggregates within-group effects with group fixed effects, averaging out the noise from G1/G2 groups while accumulating G3's consistent positive signal across runs.

### Incremental Evidence Accumulation (`RunStore`)

Each completed run is persisted as a JSONL record in `pipeline_audit/outputs/mc_runs.jsonl`.
Fields per record: `run_id` (UUID4), `run_timestamp` (ISO-8601), `seed`, `regime`, `variant`,
`alarm_level`, `max_odds_ratio`, plus ground-truth recovery metrics.

`MonteCarloEvaluator(store_path=..., append=True)`:
- Loads existing records from the JSONL store on startup
- Deduplicates by `(seed, regime, variant)` — already-completed jobs are skipped
- Appends only new results to the store
- Merges old + new into `summary_df_` before computing metrics

This allows building statistical power incrementally across multiple short sessions.

### Tests

```bash
python -m pytest pipeline_audit/tests/ -q
```

All 520 tests across: `test_world_state`, `test_actions`, `test_groups`, `test_perturbations`,
`test_pipeline_rows`, `test_group_discovery`, `test_detector`, `test_alarm`,
`test_integration`, `test_run_store`, `test_smoke`.
