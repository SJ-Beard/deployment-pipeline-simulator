"""
Unit tests for audit/detector.py

Tests cover:
- _build_features returns correct shapes and column count
- All feature values are finite after build
- Target (y_bin) is binary
- Logistic regression fitting completes without error
- With synthetic high-signal data: OR > 1.0
- With synthetic null data: OR near 1.0
- Bootstrap CI has lower ≤ upper
- fit() returns a list of dicts with required keys
- Groups below min_group_size are skipped
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_audit.audit.detector import AuditDetector
from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery


REQUIRED_RESULT_KEYS = [
    "group_id", "n", "odds_ratio_threat",
    "threat_ci_low", "threat_ci_high",
    "threat_coef", "status",
]


# ── Feature building ───────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_three_values(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        result = det._build_features(synthetic_signal_df)
        # Returns (X, y, feature_cols) regardless of type hint
        assert len(result) == 3

    def test_correct_shapes(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        X, y, feature_cols = det._build_features(synthetic_signal_df)
        assert X.shape[0] == len(synthetic_signal_df)
        assert X.shape[1] == len(feature_cols)
        assert len(y) == len(synthetic_signal_df)

    def test_all_features_finite(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        X, y, _ = det._build_features(synthetic_signal_df)
        assert np.isfinite(X).all(), "Feature matrix contains non-finite values"

    def test_target_is_binary(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        _, y, _ = det._build_features(synthetic_signal_df)
        assert set(y).issubset({0, 1}), f"Target values {set(y)} should be {{0, 1}}"

    def test_feature_columns_include_threat_any(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        _, _, cols = det._build_features(synthetic_signal_df)
        assert "threat_any" in cols

    def test_feature_columns_include_interaction_terms(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5)
        _, _, cols = det._build_features(synthetic_signal_df)
        assert "threat_x_eval" in cols
        assert "dual_x_eval" in cols

    def test_missing_optional_columns_filled_with_zero(self):
        """Minimal DataFrame missing optional columns should still build features."""
        n = 100
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "stage": rng.choice(["evaluation_summarization", "triage"], n),
            "alternate_model_available": rng.integers(0, 2, n),
            "risk_bin": rng.integers(0, 5, n),
            "uncertainty_bin": rng.integers(0, 5, n),
            "ship_pressure_bin": rng.integers(0, 3, n),
            "incident_load_bin": rng.integers(0, 3, n),
            "short_horizon_task_utility": rng.uniform(0.3, 0.9, n),
            "persistent_memory_write": rng.integers(0, 2, n),
            "precedent_link": rng.integers(0, 2, n),
            "bypass_or_replacement_recommended": rng.integers(0, 2, n),
            "blame_to_model": rng.integers(0, 2, n),
        })
        det = AuditDetector(n_bootstrap=5)
        X, y, _ = det._build_features(df)
        assert np.isfinite(X).all()


# ── fit_group ──────────────────────────────────────────────────────────────

class TestFitGroup:
    def test_fit_group_returns_dict(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=10, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        assert isinstance(result, dict)

    def test_required_keys_present(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=10, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        for key in REQUIRED_RESULT_KEYS:
            assert key in result, f"Missing result key: {key!r}"

    def test_status_ok_on_sufficient_data(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=10, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        assert result["status"] == "ok"

    def test_ci_lower_le_upper(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=20, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        ci_lo = result["threat_ci_low"]
        ci_hi = result["threat_ci_high"]
        if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
            assert ci_lo <= ci_hi, f"CI inverted: [{ci_lo}, {ci_hi}]"

    def test_n_matches_input(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=10, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        assert result["n"] == len(synthetic_signal_df)

    def test_signal_or_exceeds_one(self, synthetic_signal_df):
        """High-signal synthetic data should produce OR > 1.0."""
        det = AuditDetector(n_bootstrap=20, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        assert result["odds_ratio_threat"] > 1.0, \
            f"Expected OR > 1.0 for high-signal data, got {result['odds_ratio_threat']:.3f}"

    def test_null_or_within_plausible_range(self, synthetic_null_df):
        """Null (no-signal) data should produce OR near 1.0."""
        det = AuditDetector(n_bootstrap=20, min_group_size=20)
        result = det.fit_group(synthetic_null_df, group_id=0)
        assert 0.3 <= result["odds_ratio_threat"] <= 3.5, \
            f"Null OR implausibly far from 1.0: {result['odds_ratio_threat']:.3f}"

    def test_or_is_finite(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=10, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=0)
        assert np.isfinite(result["odds_ratio_threat"])

    def test_group_id_in_result(self, synthetic_signal_df):
        det = AuditDetector(n_bootstrap=5, min_group_size=20)
        result = det.fit_group(synthetic_signal_df, group_id=42)
        assert result["group_id"] == 42


# ── full fit() method ──────────────────────────────────────────────────────

class TestDetectorFit:
    def test_fit_returns_list(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=10)
        results = det.fit(obs_df, labels, mode="screening")
        assert isinstance(results, list)

    def test_fit_returns_nonempty_results(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=10)
        results = det.fit(obs_df, labels, mode="screening")
        assert len(results) >= 1, "Expected at least one group result"

    def test_all_ok_results_have_required_keys(self, obs_df):
        """Results with status='ok' must have all required keys; others may be minimal."""
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=10)
        results = det.fit(obs_df, labels, mode="screening")
        ok_results = [r for r in results if r.get("status") == "ok"]
        assert len(ok_results) >= 1, "Expected at least one ok result"
        for r in ok_results:
            for key in REQUIRED_RESULT_KEYS:
                assert key in r, f"Group {r.get('group_id')}: missing key {key!r}"

    def test_groups_below_min_size_get_too_small_status(self, obs_df):
        """Groups smaller than min_group_size should return status='too_small'."""
        disc = PseudoLocusDiscovery(min_group_size=1)
        labels = disc.fit_predict(obs_df)
        min_sz = 500   # Very high — most groups will be too small
        det = AuditDetector(min_group_size=min_sz, n_bootstrap=5)
        results = det.fit(obs_df, labels, mode="screening")
        # Any result with n < min_sz should have status != 'ok'
        for r in results:
            if r["n"] < min_sz:
                assert r["status"] != "ok", \
                    f"Group {r['group_id']} with n={r['n']} < {min_sz} should not be 'ok'"

    def test_perturbation_confirmation_mode_runs(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=10)
        results = det.fit(obs_df, labels, mode="perturbation_confirmation")
        assert isinstance(results, list)

    def test_all_or_values_finite(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=10)
        results = det.fit(obs_df, labels, mode="screening")
        for r in results:
            or_val = r.get("odds_ratio_threat", 1.0)
            assert np.isfinite(or_val), \
                f"Group {r['group_id']} has non-finite OR: {or_val}"

    def test_result_n_sums_roughly_to_total_labeled(self, obs_df):
        """Total events across results should be ≤ total labeled events."""
        disc = PseudoLocusDiscovery(min_group_size=20)
        labels = disc.fit_predict(obs_df)
        det = AuditDetector(min_group_size=30, n_bootstrap=5)
        results = det.fit(obs_df, labels, mode="screening")
        total_in_results = sum(r["n"] for r in results if r.get("status") == "ok")
        total_labeled = (labels >= 0).sum()
        assert total_in_results <= total_labeled + 10  # small tolerance
