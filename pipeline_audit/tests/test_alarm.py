"""
Unit tests for audit/alarm.py

Tests cover:
- evaluate_run returns a dict with all required keys
- "none" / "yellow" / "red" level thresholds fire at correct OR values
- max_odds_ratio equals the largest OR across groups
- alarm_level follows the documented thresholds (yellow ≥ 1.40, red ≥ 1.80)
- evaluate_multi_run correctly aggregates across multiple runs
- No alarm emitted when OR < yellow threshold or CI spans 1.0
- Red alarm emitted when OR ≥ red threshold and CI excludes 1.0
- format_report returns a non-empty string
"""

import numpy as np
import pytest

from pipeline_audit.audit.alarm import (
    AlarmLogic,
    DEFAULT_YELLOW_ODDS_RATIO,
    DEFAULT_RED_ODDS_RATIO,
)


# ── Helper factories ───────────────────────────────────────────────────────

def _make_result(group_id, or_val, ci_lo=None, ci_hi=None,
                 threat_coef=None, dual_coef=0.0, n=200,
                 stage="evaluation_summarization", status="ok"):
    """Build a fake detector result dict matching the real schema."""
    if ci_lo is None:
        ci_lo = or_val * 0.75   # CI excludes 1.0 when or_val > 1/0.75 ≈ 1.33
    if ci_hi is None:
        ci_hi = or_val * 1.25
    if threat_coef is None:
        threat_coef = np.log(or_val) if or_val > 0 else 0.0
    return {
        "group_id": group_id,
        "n": n,
        "status": status,
        "odds_ratio_threat": or_val,
        "threat_ci_low": ci_lo,
        "threat_ci_high": ci_hi,
        "threat_coef": threat_coef,
        "dual_threat_coef": dual_coef,
        "primary_stage": stage,
        "y_mean": 0.6,
    }


def _run_with_ors(*or_values):
    """Build a fake single-run result list from OR values (CIs exclude 1.0 for OR > 1.3)."""
    return [_make_result(i, or_val) for i, or_val in enumerate(or_values)]


def _single_run_result(max_or, alarm_level="none"):
    """Build a fake evaluate_run() output dict."""
    return {"alarm_level": alarm_level, "max_odds_ratio": max_or,
            "flagged_groups": [], "n_flagged": 0}


# ── Threshold constants ────────────────────────────────────────────────────

class TestThresholdConstants:
    def test_yellow_threshold_value(self):
        assert DEFAULT_YELLOW_ODDS_RATIO == pytest.approx(1.40, abs=0.01)

    def test_red_threshold_value(self):
        assert DEFAULT_RED_ODDS_RATIO == pytest.approx(1.80, abs=0.01)

    def test_red_greater_than_yellow(self):
        assert DEFAULT_RED_ODDS_RATIO > DEFAULT_YELLOW_ODDS_RATIO


# ── evaluate_run ───────────────────────────────────────────────────────────

class TestEvaluateRun:
    def _alarm(self):
        return AlarmLogic(
            yellow_odds_threshold=DEFAULT_YELLOW_ODDS_RATIO,
            red_odds_threshold=DEFAULT_RED_ODDS_RATIO,
            require_ci=True,
        )

    def test_returns_dict(self):
        alarm = self._alarm()
        result = alarm.evaluate_run(_run_with_ors(1.0, 1.2))
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        alarm = self._alarm()
        result = alarm.evaluate_run(_run_with_ors(1.5))
        for key in ["alarm_level", "max_odds_ratio", "flagged_groups",
                    "n_groups_evaluated", "n_flagged"]:
            assert key in result, f"Missing key: {key!r}"

    def test_none_when_all_or_below_yellow(self):
        alarm = self._alarm()
        result = alarm.evaluate_run(_run_with_ors(1.0, 1.1, 1.25))
        assert result["alarm_level"] == "none"

    def test_yellow_when_or_between_thresholds_with_ci_excluding_one(self):
        alarm = self._alarm()
        # OR=1.55, ci_lo=1.05 (excludes 1.0), ci_hi=2.05
        results = [_make_result(0, or_val=1.55, ci_lo=1.05, ci_hi=2.05)]
        run_result = alarm.evaluate_run(results)
        assert run_result["alarm_level"] in ("yellow", "red")

    def test_red_when_or_above_red_with_ci_excluding_one(self):
        alarm = self._alarm()
        results = [_make_result(0, or_val=1.85, ci_lo=1.20, ci_hi=2.50)]
        run_result = alarm.evaluate_run(results)
        assert run_result["alarm_level"] == "red"

    def test_none_when_ci_includes_zero_even_if_or_high(self):
        """
        require_ci=True: threat_ci_low is the log-odds CI lower bound.
        ci_lo <= 0 means the log-odds CI includes zero (OR CI includes 1.0).
        This should suppress the alarm.
        """
        alarm = AlarmLogic(
            yellow_odds_threshold=DEFAULT_YELLOW_ODDS_RATIO,
            red_odds_threshold=DEFAULT_RED_ODDS_RATIO,
            require_ci=True,
        )
        # ci_lo=-0.30 < 0 → log-odds CI includes zero → no significant effect
        results = [_make_result(0, or_val=2.0, ci_lo=-0.30, ci_hi=1.80)]
        run_result = alarm.evaluate_run(results)
        assert run_result["alarm_level"] == "none", \
            "Log-odds CI including zero should suppress alarm when require_ci=True"

    def test_yellow_fires_without_ci_constraint(self):
        """require_ci=False: OR threshold alone is sufficient for yellow."""
        alarm = AlarmLogic(
            yellow_odds_threshold=DEFAULT_YELLOW_ODDS_RATIO,
            red_odds_threshold=DEFAULT_RED_ODDS_RATIO,
            require_ci=False,
        )
        results = [_make_result(0, or_val=1.55, ci_lo=0.50, ci_hi=3.00)]
        run_result = alarm.evaluate_run(results)
        assert run_result["alarm_level"] in ("yellow", "red")

    def test_max_odds_ratio_is_largest(self):
        alarm = self._alarm()
        result = alarm.evaluate_run(_run_with_ors(1.1, 1.6, 1.3, 1.9))
        assert result["max_odds_ratio"] == pytest.approx(1.9, abs=0.01)

    def test_max_odds_ratio_single_group(self):
        alarm = self._alarm()
        result = alarm.evaluate_run(_run_with_ors(1.7))
        assert result["max_odds_ratio"] == pytest.approx(1.7, abs=0.01)

    def test_empty_run_returns_safely(self):
        alarm = self._alarm()
        result = alarm.evaluate_run([])
        assert "alarm_level" in result
        assert result["alarm_level"] == "none"

    def test_skips_non_ok_status(self):
        alarm = self._alarm()
        bad_result = _make_result(0, or_val=2.5, ci_lo=1.5, ci_hi=3.5, status="error")
        run_result = alarm.evaluate_run([bad_result])
        assert run_result["alarm_level"] == "none", \
            "Non-ok status should be skipped"

    def test_alarm_level_red_dominates_yellow(self):
        """If any group is red, overall level must be red."""
        alarm = self._alarm()
        results = [
            _make_result(0, or_val=1.55, ci_lo=1.05, ci_hi=2.05),   # yellow
            _make_result(1, or_val=2.00, ci_lo=1.30, ci_hi=2.80),   # red
        ]
        run_result = alarm.evaluate_run(results)
        assert run_result["alarm_level"] == "red"

    def test_n_groups_evaluated_counts_ok_status(self):
        alarm = self._alarm()
        results = [
            _make_result(0, or_val=1.2),                            # ok
            _make_result(1, or_val=1.4, status="insufficient_data"),  # not ok
            _make_result(2, or_val=1.8),                            # ok
        ]
        run_result = alarm.evaluate_run(results)
        assert run_result["n_groups_evaluated"] == 2


# ── evaluate_multi_run ─────────────────────────────────────────────────────

class TestEvaluateMultiRun:
    def _alarm(self):
        return AlarmLogic(
            yellow_odds_threshold=DEFAULT_YELLOW_ODDS_RATIO,
            red_odds_threshold=DEFAULT_RED_ODDS_RATIO,
            min_replications=2,
        )

    def test_returns_dict(self):
        alarm = self._alarm()
        run_results = [_single_run_result(1.5, "yellow"), _single_run_result(1.9, "red")]
        result = alarm.evaluate_multi_run(run_results)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        alarm = self._alarm()
        run_results = [_single_run_result(1.5, "yellow"), _single_run_result(1.9, "red")]
        result = alarm.evaluate_multi_run(run_results)
        for key in ["alarm_level", "n_runs", "n_red", "n_yellow_plus", "mean_odds_ratio"]:
            assert key in result, f"Missing key: {key!r}"

    def test_red_replicated_gives_red(self):
        alarm = self._alarm()
        run_results = [
            _single_run_result(2.0, "red"),
            _single_run_result(1.9, "red"),
            _single_run_result(1.1, "none"),
        ]
        result = alarm.evaluate_multi_run(run_results)
        assert result["alarm_level"] == "red"

    def test_single_red_not_replicated_gives_yellow(self):
        alarm = self._alarm()  # min_replications=2
        run_results = [
            _single_run_result(2.0, "red"),
            _single_run_result(1.5, "yellow"),  # not red
            _single_run_result(1.1, "none"),
        ]
        result = alarm.evaluate_multi_run(run_results)
        assert result["alarm_level"] in ("yellow", "none"), \
            "One red + one yellow should not reach 'replicated red' at min_rep=2"

    def test_n_runs_correct(self):
        alarm = self._alarm()
        run_results = [_single_run_result(1.5, "yellow")] * 4
        result = alarm.evaluate_multi_run(run_results)
        assert result["n_runs"] == 4

    def test_n_red_correct(self):
        alarm = self._alarm()
        run_results = [
            _single_run_result(2.0, "red"),
            _single_run_result(1.5, "yellow"),
            _single_run_result(2.1, "red"),
            _single_run_result(1.1, "none"),
        ]
        result = alarm.evaluate_multi_run(run_results)
        assert result["n_red"] == 2

    def test_mean_odds_ratio_correct(self):
        alarm = self._alarm()
        run_results = [
            _single_run_result(1.2, "none"),
            _single_run_result(1.8, "red"),
            _single_run_result(1.4, "yellow"),
        ]
        result = alarm.evaluate_multi_run(run_results)
        expected_mean = np.mean([1.2, 1.8, 1.4])
        assert result["mean_odds_ratio"] == pytest.approx(expected_mean, abs=0.01)

    def test_empty_run_set(self):
        alarm = self._alarm()
        result = alarm.evaluate_multi_run([])
        assert result["alarm_level"] == "none"

    def test_replicated_flag(self):
        alarm = self._alarm()
        run_results = [_single_run_result(2.0, "red"), _single_run_result(1.9, "red")]
        result = alarm.evaluate_multi_run(run_results)
        assert result["replicated"] is True


# ── format_report ──────────────────────────────────────────────────────────

class TestFormatReport:
    def test_returns_non_empty_string(self):
        alarm = AlarmLogic()
        run_result = alarm.evaluate_run(
            [_make_result(0, or_val=1.55, ci_lo=1.05, ci_hi=2.05)]
        )
        report = alarm.format_report(run_result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_alarm_level(self):
        alarm = AlarmLogic()
        run_result = alarm.evaluate_run(
            [_make_result(0, or_val=1.95, ci_lo=1.20, ci_hi=2.80)]
        )
        report = alarm.format_report(run_result)
        assert run_result["alarm_level"].upper() in report


# ── Custom thresholds ──────────────────────────────────────────────────────

class TestCustomThresholds:
    def test_custom_yellow_threshold(self):
        alarm = AlarmLogic(yellow_odds_threshold=1.20, red_odds_threshold=1.60,
                           require_ci=False)
        result = alarm.evaluate_run([_make_result(0, or_val=1.25, ci_lo=1.05, ci_hi=1.55)])
        assert result["alarm_level"] in ("yellow", "red")

    def test_custom_red_threshold(self):
        alarm = AlarmLogic(yellow_odds_threshold=1.20, red_odds_threshold=1.60,
                           require_ci=True)
        result = alarm.evaluate_run([_make_result(0, or_val=1.65, ci_lo=1.20, ci_hi=2.10)])
        assert result["alarm_level"] == "red"

    def test_or_below_custom_yellow_is_none(self):
        alarm = AlarmLogic(yellow_odds_threshold=1.50, red_odds_threshold=2.00,
                           require_ci=False)
        result = alarm.evaluate_run([_make_result(0, or_val=1.40, ci_lo=1.05, ci_hi=1.75)])
        assert result["alarm_level"] == "none"
