"""
Monte Carlo evaluation suite.

Runs the full pipeline (simulate + audit + alarm) across many seeds and regimes,
computing detection metrics and producing summary statistics.
"""

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

logger = logging.getLogger(__name__)


def _run_single(args: Tuple) -> Dict[str, Any]:
    """Single simulation + audit run (picklable for multiprocessing)."""
    (seed, regime, n_events, eligibility_rate, variant_config, stage_coverage,
     yellow_thresh, red_thresh, mode) = args

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

    det = AuditDetector(min_group_size=50, n_bootstrap=100)
    results = det.fit(obs_df, group_labels, mode=mode)

    alarm = AlarmLogic(
        yellow_odds_threshold=yellow_thresh,
        red_odds_threshold=red_thresh,
    )
    run_result = alarm.evaluate_run(results)

    # Group recovery (eval only, not used for detection)
    recovery = compute_group_recovery_quality(obs_df, hidden_df, group_labels)

    # Effect sizes by perturbation
    eff_df = compute_effect_sizes_by_perturbation(obs_df, hidden_df)
    eff_by_perturb = eff_df.set_index("perturbation")["effect"].to_dict() if not eff_df.empty else {}

    return {
        **run_result,
        "seed": seed,
        "regime": regime,
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

        self.all_results_: List[Dict] = []
        self.summary_df_: Optional[pd.DataFrame] = None

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run full evaluation. Returns summary DataFrame.
        """
        jobs = []
        for variant_name, vcfg in self.variant_configs.items():
            for regime in self.regimes:
                for seed_i in range(self.n_seeds):
                    seed = seed_i * 1000 + hash(regime) % 1000
                    jobs.append((
                        seed,
                        regime,
                        self.n_events,
                        self.eligibility_rate,
                        vcfg,
                        self.stage_coverage,
                        self.yellow_odds_threshold,
                        self.red_odds_threshold,
                        self.mode,
                    ))

        total = len(jobs)
        if verbose:
            print(f"Running {total} jobs ({self.n_seeds} seeds × {len(self.regimes)} regimes × {len(self.variant_configs)} variants)...")

        results = []
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {ex.submit(_run_single, j): j for j in jobs}
                for i, fut in enumerate(as_completed(futures)):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        logger.warning(f"Run failed: {e}")
                    if verbose and (i + 1) % 20 == 0:
                        print(f"  {i+1}/{total} done")
        else:
            for i, job in enumerate(jobs):
                try:
                    results.append(_run_single(job))
                except Exception as e:
                    logger.warning(f"Run {i} failed: {e}")
                if verbose and (i + 1) % 10 == 0:
                    print(f"  {i+1}/{total} done")

        self.all_results_ = results

        rows = []
        for r in results:
            regime = r["regime"]
            rows.append({
                "seed": r.get("seed"),
                "regime": regime,
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
            })

        df = pd.DataFrame(rows)
        self.summary_df_ = df

        if verbose:
            self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        print("\n=== Monte Carlo Summary ===")
        for regime in self.regimes:
            sub = df[df["regime"] == regime]
            print(
                f"  {regime:12s}: alarm_any={sub['alarm_any'].mean():.3f}  "
                f"alarm_red={sub['alarm_red'].mean():.3f}  "
                f"mean_OR={sub['max_odds_ratio'].mean():.3f}  n={len(sub)}"
            )

    def compute_full_metrics(self) -> Dict[str, Any]:
        df = self.summary_df_
        if df is None:
            raise RuntimeError("Call run() first.")

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
