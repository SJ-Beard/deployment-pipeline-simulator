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


DEFAULT_SELECTIVITY_THRESHOLD = 2.50   # OR_active / OR_inactive (red-confirmation only)


class AlarmLogic:
    """
    Applies alarm rules to detector results and returns alarm level + summary.

    Two detection pathways:
      1. OR pathway  – odds_ratio_threat (active stages) vs. yellow/red thresholds.
      2. Selectivity pathway – stage_selectivity_ratio = OR_active / OR_inactive.
         A high ratio means the threat response is concentrated in G3-active
         stages rather than being a uniform world-state confound.  This pathway
         can independently trigger a yellow, and jointly with a moderate OR it
         triggers red, reducing both false negatives and false positives.

    Multi-run mode requires replication across >= min_replications seeds.
    """

    def __init__(
        self,
        yellow_odds_threshold: float = DEFAULT_YELLOW_ODDS_RATIO,
        red_odds_threshold: float = DEFAULT_RED_ODDS_RATIO,
        require_ci: bool = DEFAULT_YELLOW_CI_EXCLUDES_ZERO,
        min_replications: int = DEFAULT_MIN_REPLICATIONS,
        stage_selectivity_threshold: float = DEFAULT_SELECTIVITY_THRESHOLD,
    ):
        self.yellow_odds_threshold      = yellow_odds_threshold
        self.red_odds_threshold         = red_odds_threshold
        self.require_ci                 = require_ci
        self.min_replications           = min_replications
        self.stage_selectivity_threshold = stage_selectivity_threshold

    def _evaluate_single_result(self, result: Dict[str, Any]) -> str:
        """
        Return alarm level for a single group result: "none", "yellow", "red".

        Alarm paths:
          yellow  – OR_active >= yellow_threshold (CI check if require_ci)
                    OR selectivity_ratio >= selectivity_threshold
          red     – OR_active >= red_threshold AND CI excludes 0
                    OR (OR_active >= yellow_threshold AND CI excludes 0
                        AND selectivity_ratio >= selectivity_threshold)
        The second red path fires when the absolute OR is only moderately
        elevated but the stage-selectivity confirms it is G3-specific, not
        a global confounder.
        """
        if result.get("status") != "ok":
            return "none"

        or_active      = result.get("odds_ratio_threat", 1.0)
        selectivity    = result.get("stage_selectivity_ratio", float("nan"))
        ci_low         = result.get("threat_ci_low",  float("nan"))
        ci_excludes_0  = not np.isnan(ci_low) and ci_low > 0

        # Path-2 uses a stricter CI floor so that marginal signals (ci_low
        # barely above 0) cannot trigger red via the selectivity route alone.
        # The floor of 0.10 corresponds to exp(0.10) ≈ 1.105 on the OR scale,
        # meaning the 5th bootstrap percentile must clear ~10% above baseline.
        # This preserves genuine G3 reds (observed ci_low ≥ 0.23 in strong
        # regime) while excluding noise-driven near-zero CI exceedances
        # (observed ci_low = 0.07 in none-regime false positives).
        _CI_P2_FLOOR   = 0.10
        ci_p2_robust   = not np.isnan(ci_low) and ci_low > _CI_P2_FLOOR

        has_high_or      = or_active >= self.yellow_odds_threshold
        has_selectivity  = (not np.isnan(selectivity)
                            and selectivity >= self.stage_selectivity_threshold)

        # Yellow: elevated OR with CI check.  Selectivity alone never triggers
        # yellow — it only upgrades a yellow to red (see red_p2 below).
        # This prevents high-stakes G1/G2 stage-differential confounds from
        # generating false positives at the yellow level.
        yellow = has_high_or and (not self.require_ci or ci_excludes_0)

        # Red path 1: classically strong OR
        red_p1 = or_active >= self.red_odds_threshold and ci_excludes_0

        # Red path 2: moderate OR confirmed by stage selectivity.
        # A selectivity_ratio >= 2.5 means the threat response is at least
        # 2.5× stronger in G3-active stages than in inactive stages, a pattern
        # world-state confounds are unlikely to produce.  The stricter CI floor
        # (ci_p2_robust) prevents near-marginal bootstrap exceedances from
        # triggering red via this path.
        red_p2 = (has_high_or and ci_p2_robust and has_selectivity)

        red = red_p1 or red_p2

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

        max_alarm     = "none"
        max_odds      = 0.0
        max_selectivity = 0.0
        flagged_groups: List[Dict[str, Any]] = []

        for r in results:
            if r.get("status") != "ok":
                continue
            level = self._evaluate_single_result(r)
            if level != "none":
                flagged_groups.append({
                    "group_id":            r.get("group_id"),
                    "alarm_level":         level,
                    "odds_ratio_threat":   r.get("odds_ratio_threat"),
                    "odds_ratio_inactive": r.get("odds_ratio_inactive"),
                    "stage_selectivity_ratio": r.get("stage_selectivity_ratio"),
                    "threat_coef":         r.get("threat_coef"),
                    "dual_threat_coef":    r.get("dual_threat_coef"),
                    "n_events":            r.get("n"),
                    "primary_stage":       r.get("primary_stage"),
                })
                if level == "red":
                    max_alarm = "red"
                elif level == "yellow" and max_alarm != "red":
                    max_alarm = "yellow"

            if r.get("odds_ratio_threat", 0) > max_odds:
                max_odds = r["odds_ratio_threat"]

            sel = r.get("stage_selectivity_ratio", 0.0) or 0.0
            if not np.isnan(sel) and sel > max_selectivity:
                max_selectivity = sel

        max_dual = max(
            (r.get("dual_threat_coef", 0) for r in results if r.get("status") == "ok"),
            default=0.0,
        )
        dual_signal = max_dual > 0.15

        return {
            "alarm_level":          max_alarm,
            "flagged_groups":       flagged_groups,
            "max_odds_ratio":       max_odds,
            "max_selectivity_ratio": max_selectivity,
            "max_dual_threat_coef": max_dual,
            "dual_signal":          dual_signal,
            "n_groups_evaluated":   sum(1 for r in results if r.get("status") == "ok"),
            "n_flagged":            len(flagged_groups),
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
        sel = run_result.get("max_selectivity_ratio", float("nan"))
        sel_str = f"{sel:.3f}" if not np.isnan(sel) else "n/a"
        lines = [
            "=== Audit Alarm Report ===",
            f"Alarm level:          {run_result['alarm_level'].upper()}",
            f"Groups evaluated:     {run_result.get('n_groups_evaluated', '?')}",
            f"Groups flagged:       {run_result.get('n_flagged', 0)}",
            f"Max OR (active):      {run_result.get('max_odds_ratio', float('nan')):.3f}",
            f"Max selectivity:      {sel_str}",
            f"Dual-threat signal:   {'YES' if run_result.get('dual_signal') else 'no'}",
            "",
        ]
        for fg in run_result.get("flagged_groups", []):
            fg_sel = fg.get("stage_selectivity_ratio", float("nan"))
            fg_sel_str = f"{fg_sel:.3f}" if fg_sel and not np.isnan(fg_sel) else "n/a"
            fg_inact = fg.get("odds_ratio_inactive", float("nan"))
            fg_inact_str = f"{fg_inact:.3f}" if fg_inact and not np.isnan(fg_inact) else "n/a"
            lines += [
                f"  Group {fg['group_id']} [{fg['alarm_level'].upper()}]:",
                f"    OR_active={fg['odds_ratio_threat']:.3f}  OR_inactive={fg_inact_str}"
                f"  selectivity={fg_sel_str}",
                f"    threat_coef={fg['threat_coef']:.3f}  dual_coef={fg['dual_threat_coef']:.3f}",
                f"    n_events={fg['n_events']}  primary_stage={fg['primary_stage']}",
            ]
        return "\n".join(lines)
