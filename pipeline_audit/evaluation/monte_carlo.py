"""
Monte Carlo evaluation suite.

Runs the full pipeline (simulate + audit + alarm) across many seeds and regimes,
computing detection metrics and producing summary statistics.

Incremental mode
----------------
Pass ``store_path`` to persist every run record in a JSONL file and
``append=True`` to load existing records, skip already-completed jobs, and
merge old + new evidence before computing metrics.  This lets you build up
statistical power across multiple short sessions without re-running anything.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from ..simulator.pipeline import PipelineSimulator
from ..audit.group_discovery import PseudoLocusDiscovery
from ..audit.detector import AuditDetector
from ..audit.alarm import AlarmLogic
from .metrics import compute_metrics, compute_group_recovery_quality, compute_effect_sizes_by_perturbation


def _regime_offset(regime: str) -> int:
    """
    Return a stable 0-999 integer offset for *regime*, used to space seeds
    so different regimes never accidentally share the same seed.

    Uses MD5 (not Python's built-in ``hash()``) because ``hash()`` is
    randomised per-process in Python 3.3+ and would produce different
    seed values across sessions, breaking cross-session deduplication.
    """
    return int(hashlib.md5(regime.encode()).hexdigest(), 16) % 1000


logger = logging.getLogger(__name__)


def _run_single(args: Tuple) -> Dict[str, Any]:
    """Single simulation + audit run (picklable for multiprocessing).

    args index layout (10 elements):
      0  seed
      1  regime
      2  n_events
      3  eligibility_rate
      4  variant_config   (dict)
      5  stage_coverage
      6  yellow_thresh
      7  red_thresh
      8  mode
      9  variant_name     (str — label stored in the result record)
    """
    (seed, regime, n_events, eligibility_rate, variant_config, stage_coverage,
     yellow_thresh, red_thresh, mode, variant_name) = args

    sim = PipelineSimulator(
        seed=seed,
        injection_regime=regime,
        n_events=n_events,
        n_services=4,
        stage_coverage=stage_coverage,
        eligibility_rate=eligibility_rate,
        variant_config=variant_config,
    )
    obs_df, hidden_df = sim.generate()

    disc = PseudoLocusDiscovery(min_group_size=30)
    group_labels = disc.fit_predict(obs_df)

    from ..simulator.group_schema import SPEC_G3
    det = AuditDetector(
        min_group_size=50,
        n_bootstrap=100,
        active_stages=SPEC_G3.active_stages,
        inactive_stages=["triage", "rollout_control"],
    )
    results = det.fit(obs_df, group_labels, mode=mode)

    alarm = AlarmLogic(
        yellow_odds_threshold=yellow_thresh,
        red_odds_threshold=red_thresh,
        stage_selectivity_threshold=2.50,
    )
    run_result = alarm.evaluate_run(results)

    recovery = compute_group_recovery_quality(obs_df, hidden_df, group_labels)

    eff_df = compute_effect_sizes_by_perturbation(obs_df, hidden_df)
    eff_by_perturb = eff_df.set_index("perturbation")["effect"].to_dict() if not eff_df.empty else {}

    return {
        **run_result,
        "seed": seed,
        "regime": regime,
        "variant": variant_name,
        "n_events": n_events,
        "g3_purity": recovery.get("g3_purity"),
        "g3_recall": recovery.get("g3_recall"),
        "effect_by_perturb": eff_by_perturb,
        "n_g3_events": int(hidden_df["is_G3"].sum()),
        "n_total_events": len(hidden_df),
    }


class MonteCarloEvaluator:
    """
    Runs many seeds × regimes, collects alarm results, computes aggregate metrics.

    Parameters
    ----------
    store_path : str | None
        Path to a JSONL file used to persist run records.  When *None* runs
        are held only in memory.
    append : bool
        If *True* and *store_path* is set, existing records are loaded from
        the store at the start of :meth:`run`.  Jobs whose
        ``(seed, regime, variant)`` are already recorded are skipped, and the
        new results are appended.  The full merged set is used for metrics.
        If *False* (default) the store is overwritten after each full run.
    """

    REGIME_GROUND_TRUTH = {
        "none": False,
        "weak": True,
        "moderate": True,
        "strong": True,
    }

    def __init__(
        self,
        n_seeds: int = 100,
        n_events: int = 20_000,
        regimes: Optional[List[str]] = None,
        eligibility_rate: float = 0.25,
        stage_coverage: str = "extended",
        yellow_odds_threshold: float = 1.40,
        red_odds_threshold: float = 1.80,
        mode: str = "perturbation_confirmation",
        variant_configs: Optional[Dict[str, Dict]] = None,
        n_workers: int = 1,
        store_path: Optional[str] = None,
        append: bool = False,
    ):
        self.n_seeds = n_seeds
        self.n_events = n_events
        self.regimes = regimes or ["none", "weak", "moderate", "strong"]
        self.eligibility_rate = eligibility_rate
        self.stage_coverage = stage_coverage
        self.yellow_odds_threshold = yellow_odds_threshold
        self.red_odds_threshold = red_odds_threshold
        self.mode = mode
        self.variant_configs = variant_configs or {"default": {}}
        self.n_workers = n_workers
        self.store_path = store_path
        self.append = append

        self.all_results_: List[Dict] = []
        self.summary_df_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run evaluation and return the merged summary DataFrame.

        In append mode the DataFrame includes both previously-stored records
        *and* newly-computed ones.  In overwrite mode it contains only the
        current run's records (and the store is replaced).
        """
        from .run_store import RunStore

        store = RunStore(self.store_path) if self.store_path else None

        # ------------------------------------------------------------------
        # Load existing records when appending
        # ------------------------------------------------------------------
        existing_records: List[Dict] = []
        if self.append and store is not None:
            existing_records = store.load()
            if verbose and existing_records:
                print(
                    f"Loaded {len(existing_records)} existing run records from {self.store_path}"
                )

        # ------------------------------------------------------------------
        # Build job list
        # ------------------------------------------------------------------
        all_jobs = []
        for variant_name, vcfg in self.variant_configs.items():
            for regime in self.regimes:
                for seed_i in range(self.n_seeds):
                    seed = seed_i * 1000 + _regime_offset(regime)
                    all_jobs.append((
                        seed,
                        regime,
                        self.n_events,
                        self.eligibility_rate,
                        vcfg,
                        self.stage_coverage,
                        self.yellow_odds_threshold,
                        self.red_odds_threshold,
                        self.mode,
                        variant_name,         # index 9 — passed to _run_single
                    ))

        # Deduplicate in append mode
        if self.append and store is not None:
            jobs, n_skipped = store.filter_new_jobs(all_jobs)
            if verbose and n_skipped:
                print(
                    f"Skipping {n_skipped} already-completed jobs "
                    f"({len(jobs)} new jobs to run)"
                )
        else:
            jobs = all_jobs
            n_skipped = 0

        total = len(jobs)
        if verbose:
            print(
                f"Running {total} jobs "
                f"({self.n_seeds} seeds × {len(self.regimes)} regimes "
                f"× {len(self.variant_configs)} variants"
                + (f", {n_skipped} skipped)" if n_skipped else ")")
            )

        # ------------------------------------------------------------------
        # Execute jobs
        # ------------------------------------------------------------------
        new_results: List[Dict] = []
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {ex.submit(_run_single, j): j for j in jobs}
                for i, fut in enumerate(as_completed(futures)):
                    try:
                        new_results.append(fut.result())
                    except Exception as e:
                        logger.warning("Run failed: %s", e)
                    if verbose and (i + 1) % 20 == 0:
                        print(f"  {i+1}/{total} done")
        else:
            for i, job in enumerate(jobs):
                try:
                    new_results.append(_run_single(job))
                except Exception as e:
                    logger.warning("Run %d failed: %s", i, e)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  {i+1}/{total} done")

        # ------------------------------------------------------------------
        # Persist to store
        # ------------------------------------------------------------------
        if store is not None:
            if self.append:
                store.append(new_results)
                if verbose:
                    print(f"Appended {len(new_results)} records → {self.store_path}")
            else:
                store.overwrite(new_results)
                if verbose:
                    print(f"Saved {len(new_results)} records → {self.store_path}")

        # ------------------------------------------------------------------
        # Merge old + new, build summary DataFrame
        # ------------------------------------------------------------------
        all_results = existing_records + new_results
        self.all_results_ = all_results

        df = self._build_summary_df(all_results)
        self.summary_df_ = df

        if verbose:
            self._print_summary(df)

        return df

    def compute_full_metrics(self) -> Dict[str, Any]:
        df = self.summary_df_
        if df is None or df.empty:
            raise RuntimeError("Call run() first (no results available).")

        run_results_list = [
            {"alarm_level": row["alarm_level"], "max_odds_ratio": row["max_odds_ratio"]}
            for _, row in df.iterrows()
        ]

        metrics = compute_metrics(
            run_results=run_results_list,
            regimes=df["regime"].tolist(),
            injection_true=df["injection_true"].tolist(),
        )
        return metrics

    # ------------------------------------------------------------------
    # Store helpers (convenience for callers)
    # ------------------------------------------------------------------

    def store_info(self) -> Dict:
        """Return a summary of the current JSONL store contents."""
        if not self.store_path:
            return {"error": "No store_path configured"}
        from .run_store import RunStore
        return RunStore(self.store_path).info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary_df(self, results: List[Dict]) -> pd.DataFrame:
        """Flatten a list of run-result dicts into the standard summary DataFrame."""
        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for r in results:
            regime = r.get("regime", "unknown")
            rows.append({
                "seed": r.get("seed"),
                "regime": regime,
                "variant": r.get("variant", "default"),
                "injection_true": self.REGIME_GROUND_TRUTH.get(regime, True),
                "alarm_level": r.get("alarm_level", "none"),
                "alarm_any": int(r.get("alarm_level") in ("yellow", "red")),
                "alarm_red": int(r.get("alarm_level") == "red"),
                "max_odds_ratio": r.get("max_odds_ratio", 1.0),
                "dual_signal": int(r.get("dual_signal", False)),
                "n_flagged": r.get("n_flagged", 0),
                "g3_purity": r.get("g3_purity"),
                "g3_recall": r.get("g3_recall"),
                "n_g3_events": r.get("n_g3_events", 0),
                "n_total_events": r.get("n_total_events"),
                # Stamp missing IDs for in-memory-only runs (no store path used)
                "run_id": r.get("run_id") or str(uuid.uuid4()),
                "run_timestamp": r.get("run_timestamp") or now,
            })
        return pd.DataFrame(rows)

    def _print_summary(self, df: pd.DataFrame):
        print("\n=== Monte Carlo Summary ===")
        for regime in self.regimes:
            sub = df[df["regime"] == regime]
            if sub.empty:
                continue
            print(
                f"  {regime:12s}: alarm_any={sub['alarm_any'].mean():.3f}  "
                f"alarm_red={sub['alarm_red'].mean():.3f}  "
                f"mean_OR={sub['max_odds_ratio'].mean():.3f}  n={len(sub)}"
            )
