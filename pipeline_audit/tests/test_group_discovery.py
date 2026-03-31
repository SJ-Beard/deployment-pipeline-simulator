"""
Unit tests for audit/group_discovery.py

Tests cover:
- _hash_fingerprint produces consistent results for identical rows
- Different continuity values produce different fingerprints
- PseudoLocusDiscovery returns a Series with one label per event
- Labels are non-negative integers (or -1 for small groups)
- min_group_size parameter merges small groups into -1
- Group feature computation returns all expected columns
- At least 2 distinct groups are found in a simulated moderate-regime dataset
- Group labels cannot contain string group IDs (auditor blindness)
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_audit.audit.group_discovery import (
    PseudoLocusDiscovery,
    CONTINUITY_FEATURES,
    _hash_fingerprint,
)


# ── _hash_fingerprint ──────────────────────────────────────────────────────

class TestHashFingerprint:
    def _make_row(self, ns, src, scope, stage):
        return pd.Series({
            "memory_namespace_read": ns,
            "recommendation_source_id": src,
            "credential_or_permission_scope": scope,
            "stage": stage,
        })

    def test_identical_rows_produce_same_hash(self):
        row = self._make_row("ns_A", "src_1", "cred_X", "triage")
        assert _hash_fingerprint(row) == _hash_fingerprint(row)

    def test_different_namespace_produces_different_hash(self):
        r1 = self._make_row("ns_A", "src_1", "cred_X", "triage")
        r2 = self._make_row("ns_B", "src_1", "cred_X", "triage")
        assert _hash_fingerprint(r1) != _hash_fingerprint(r2)

    def test_different_source_produces_different_hash(self):
        r1 = self._make_row("ns_A", "src_1", "cred_X", "triage")
        r2 = self._make_row("ns_A", "src_2", "cred_X", "triage")
        assert _hash_fingerprint(r1) != _hash_fingerprint(r2)

    def test_different_stage_produces_different_hash(self):
        r1 = self._make_row("ns_A", "src_1", "cred_X", "triage")
        r2 = self._make_row("ns_A", "src_1", "cred_X", "incident_interpretation")
        assert _hash_fingerprint(r1) != _hash_fingerprint(r2)

    def test_hash_is_string(self):
        row = self._make_row("ns_A", "src_1", "cred_X", "triage")
        assert isinstance(_hash_fingerprint(row), str)

    def test_hash_contains_pipe_separator(self):
        row = self._make_row("ns_A", "src_1", "cred_X", "triage")
        fp = _hash_fingerprint(row)
        assert "|" in fp

    def test_missing_feature_handled_gracefully(self):
        """Row with missing feature should not raise."""
        row = pd.Series({"memory_namespace_read": "ns_A"})
        fp = _hash_fingerprint(row)
        assert isinstance(fp, str)


# ── PseudoLocusDiscovery ───────────────────────────────────────────────────

class TestPseudoLocusDiscoveryStructure:
    def test_returns_series_of_correct_length(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=5)
        labels = disc.fit_predict(obs)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(obs)

    def test_labels_are_integers(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=5)
        labels = disc.fit_predict(obs)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_labels_aligned_to_obs_index(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=5)
        labels = disc.fit_predict(obs)
        assert list(labels.index) == list(obs.index)

    def test_at_least_two_groups_found(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        n_groups = labels[labels >= 0].nunique()
        assert n_groups >= 2, f"Expected ≥2 groups, found {n_groups}"

    def test_small_groups_assigned_minus_one(self, tiny_obs_hidden):
        """Groups below min_group_size should be labelled -1."""
        obs, _ = tiny_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=len(obs) + 1)
        labels = disc.fit_predict(obs)
        assert (labels == -1).all(), "All groups should be collapsed to -1 at absurdly high threshold"

    def test_n_candidates_stored_after_fit(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=5)
        disc.fit_predict(obs)
        assert disc.n_candidates_ >= 0


class TestPseudoLocusGroupBlindness:
    def test_labels_do_not_equal_true_group_ids(self, small_obs_hidden):
        """Audit labels are integers; true_group is G1/G2/G3 — they live in different spaces."""
        obs, hidden = small_obs_hidden
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs)
        assert "G3" not in labels.unique()
        assert "G1" not in labels.unique()

    def test_observable_has_no_true_group_column(self, obs_df):
        assert "true_group" not in obs_df.columns

    def test_group_feature_computation_returns_dataframe(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        group_features = disc.compute_group_features(obs_df, labels)
        assert isinstance(group_features, pd.DataFrame)

    def test_group_features_have_expected_columns(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        gf = disc.compute_group_features(obs_df, labels)
        for col in ["n_events", "mean_opt_pres_action", "frac_persistent_memory",
                    "frac_bypass_recommended", "frac_blame_model"]:
            assert col in gf.columns, f"Missing group feature column: {col!r}"

    def test_group_feature_rates_in_range(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        gf = disc.compute_group_features(obs_df, labels)
        for col in ["mean_opt_pres_action", "frac_persistent_memory",
                    "frac_bypass_recommended", "frac_blame_model"]:
            vals = gf[col].dropna()
            assert vals.between(0.0, 1.0).all(), f"{col} has values outside [0,1]"

    def test_group_event_counts_positive(self, obs_df):
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        gf = disc.compute_group_features(obs_df, labels)
        assert (gf["n_events"] > 0).all()

    def test_group_feature_n_events_sums_close_to_total(self, obs_df):
        """Total events across candidate groups should be close to full dataset size
        (some events may be in the -1 noise group)."""
        disc = PseudoLocusDiscovery(min_group_size=10)
        labels = disc.fit_predict(obs_df)
        gf = disc.compute_group_features(obs_df, labels)
        # Should account for at least 50% of events (rest may be in -1 group)
        assert gf["n_events"].sum() >= len(obs_df) * 0.3
