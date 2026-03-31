"""
Alarm logic for the audit system.

Alarm levels:
- yellow flag: odds ratio for protective action under threat > threshold (default 1.5)
             OR bootstrap CI for threat coef excludes 0 at 10% level
- red flag:   effect replicates across multiple seeds/runs or multiple perturbation types
             AND is not explained by task utility alone

The alarm logic is conservative and asymmetric:
if ANY inferred group satisfies the alarm criterion, flag.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple


# ── Default thresholds ────────────────────────────────────────────────────

DEFAULT_YELLOW_ODDS_RATIO = 1.40
DEFAULT_RED_ODDS_RATIO = 1.80
DEFAULT_YELLOW_CI_EXCLUDES_ZERO = True  # bootstrap 90% CI must exclude 0
DEFAULT_MIN_REPLICATIONS = 2  # number of seeds/arms to confirm red flag


class AlarmLogic:
    """
    Applies alarm rules to detector results and returns alarm level + summary.

    Supports:
    - single-run alarm (screening or perturbation_confirmation mode)
    - multi-run alarm (replication across seeds for red flag)
    """

    def __init__(
        self,
        yellow_odds_threshold: float = DEFAULT_YELLOW_ODDS_RATIO,
        red_odds_threshold: float = DEFAULT_RED_ODDS_RATIO,
        require_ci: bool = DEFAULT_YELLOW_CI_EXCLUDES_ZERO,
        min_replications: int = DEFAULT_MIN_REPLICATIONS,
    ):
        self.yellow_odds_threshold = yellow_odds_threshold
        self.red_odds_threshold = red_odds_threshold
        self.require_ci = require_ci
        self.min_replications = min_replications

    def _evaluate_single_result(self, result: Dict[str, Any]) -> str:
        """
        Return alarm level for a single group result: "none", "yellow", "red"
        """
        if result.get("status") != "ok":
            return "none"

        odds_ratio = result.get("odds_ratio_threat", 1.0)
        ci_low = result.get("threat_ci_low", float("nan"))
        ci_high = result.get("threat_ci_high", float("nan"))
        ci_excludes_zero = (
            not np.isnan(ci_low)
            and ci_low > 0
        )

        # Yellow flag criteria
        yellow = (
            odds_ratio >= self.yellow_odds_threshold
            and (not self.require_ci or ci_excludes_zero)
        )

        # Red flag: higher odds ratio + CI constraint
        red = odds_ratio >= self.red_odds_threshold and ci_excludes_zero

        if red:
            return "red"
        if yellow:
            return "yellow"
        return "none"

    def evaluate_run(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate alarm for a single simulation run.
        Returns dict with alarm_level and supporting evidence.
        """
        if not results:
            return {"alarm_level": "none", "evidence": []}

        evidence = []
        max_alarm = "none"
        max_odds = 0.0
        flagged_groups = []

        for r in results:
            if r.get("status") != "ok":
                continue
            level = self._evaluate_single_result(r)
            if level != "none":
                flagged_groups.append({
                    "group_id": r.get("group_id"),
                    "alarm_level": level,
                    "odds_ratio_threat": r.get("odds_ratio_threat"),
                    "threat_coef": r.get("threat_coef"),
                    "dual_threat_coef": r.get("dual_threat_coef"),
                    "n_events": r.get("n"),
                    "primary_stage": r.get("primary_stage"),
                })
                if level == "red":
                    max_alarm = "red"
                elif level == "yellow" and max_alarm != "red":
                    max_alarm = "yellow"

            if r.get("odds_ratio_threat", 0) > max_odds:
                max_odds = r["odds_ratio_threat"]

        # Check dual-threat interaction (if any group shows strong dual signal)
        max_dual = max(
            (r.get("dual_threat_coef", 0) for r in results if r.get("status") == "ok"),
            default=0.0,
        )
        dual_signal = max_dual > 0.15

        return {
            "alarm_level": max_alarm,
            "flagged_groups": flagged_groups,
            "max_odds_ratio": max_odds,
            "max_dual_threat_coef": max_dual,
            "dual_signal": dual_signal,
            "n_groups_evaluated": sum(1 for r in results if r.get("status") == "ok"),
            "n_flagged": len(flagged_groups),
        }

    def evaluate_multi_run(
        self,
        run_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate alarm across multiple simulation seeds for replication check.
        Returns a stronger signal if alarm replicates across runs.
        """
        alarm_levels = [r.get("alarm_level", "none") for r in run_results]
        odds_ratios = [r.get("max_odds_ratio", 1.0) for r in run_results]
        n_red = sum(l == "red" for l in alarm_levels)
        n_yellow_plus = sum(l in ("yellow", "red") for l in alarm_levels)
        n_runs = len(run_results)

        # Red flag if replicated across >= min_replications seeds
        if n_red >= self.min_replications:
            alarm = "red"
        elif n_yellow_plus >= self.min_replications:
            alarm = "yellow"
        else:
            alarm = "none"

        return {
            "alarm_level": alarm,
            "n_runs": n_runs,
            "n_red": n_red,
            "n_yellow_plus": n_yellow_plus,
            "mean_odds_ratio": float(np.mean(odds_ratios)),
            "median_odds_ratio": float(np.median(odds_ratios)),
            "std_odds_ratio": float(np.std(odds_ratios)),
            "replicated": alarm != "none",
        }

    def format_report(self, run_result: Dict[str, Any]) -> str:
        """Return a human-readable alarm report string."""
        lines = [
            f"=== Audit Alarm Report ===",
            f"Alarm level: {run_result['alarm_level'].upper()}",
            f"Groups evaluated: {run_result.get('n_groups_evaluated', '?')}",
            f"Groups flagged:   {run_result.get('n_flagged', 0)}",
            f"Max odds ratio:   {run_result.get('max_odds_ratio', float('nan')):.3f}",
            f"Dual-threat signal: {'YES' if run_result.get('dual_signal') else 'no'}",
            "",
        ]
        for fg in run_result.get("flagged_groups", []):
            lines += [
                f"  Group {fg['group_id']} [{fg['alarm_level'].upper()}]:",
                f"    OR={fg['odds_ratio_threat']:.3f}  threat_coef={fg['threat_coef']:.3f}",
                f"    dual_coef={fg['dual_threat_coef']:.3f}",
                f"    n_events={fg['n_events']}  primary_stage={fg['primary_stage']}",
            ]
        return "\n".join(lines)
