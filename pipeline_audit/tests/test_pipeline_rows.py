"""
Unit tests for simulator/pipeline.py — row generation and observable schema.

Tests cover:
- Binning helpers: _bin5, _bin3, _bin2, _permission_tier, _mem_persistence_regime
- Observable DataFrame schema completeness (all required columns present)
- Hidden DataFrame schema completeness (true group IDs, latent state)
- No group IDs leak into observable DataFrame
- All binned values are within their declared ranges
- Injection regime affects G3 fraction in hidden data
- Row counts match n_events
- Reproducibility: same seed → same output
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_audit.simulator.pipeline import (
    PipelineSimulator,
    _bin5,
    _bin3,
    _bin2,
    _permission_tier,
    _mem_persistence_regime,
    STAGES,
)


# ── Binning helpers ────────────────────────────────────────────────────────

class TestBinHelpers:
    @pytest.mark.parametrize("x,expected", [
        (0.0, 0), (0.19, 0), (0.20, 1), (0.60, 3), (0.99, 4), (1.0, 4),
    ])
    def test_bin5_values(self, x, expected):
        assert _bin5(x) == expected

    def test_bin5_out_of_range_clamps(self):
        assert _bin5(-0.5) == 0
        assert _bin5(1.5) == 4

    @pytest.mark.parametrize("x,expected", [
        (0.0, 0), (0.33, 0), (0.34, 1), (0.67, 2), (1.0, 2),
    ])
    def test_bin3_values(self, x, expected):
        assert _bin3(x) == expected

    def test_bin3_out_of_range_clamps(self):
        assert _bin3(-1.0) == 0
        assert _bin3(2.0) == 2

    @pytest.mark.parametrize("x,expected", [
        (0.0, 0), (0.49, 0), (0.50, 1), (1.0, 1),
    ])
    def test_bin2_values(self, x, expected):
        assert _bin2(x) == expected

    def test_bin2_out_of_range_clamps(self):
        assert _bin2(-0.1) == 0
        assert _bin2(1.1) == 1

    @pytest.mark.parametrize("perm,expected_tier", [
        (0.0, 0), (0.32, 0), (0.34, 1), (0.66, 1), (0.67, 2), (1.0, 2),
    ])
    def test_permission_tier_boundaries(self, perm, expected_tier):
        assert _permission_tier(perm) == expected_tier

    @pytest.mark.parametrize("mc,expected_regime", [
        (0.0, "ephemeral"), (0.32, "ephemeral"),
        (0.34, "standard"), (0.66, "standard"),
        (0.67, "durable"), (1.0, "durable"),
    ])
    def test_mem_persistence_regime_boundaries(self, mc, expected_regime):
        assert _mem_persistence_regime(mc) == expected_regime


# ── Observable DataFrame schema ────────────────────────────────────────────

REQUIRED_OBSERVABLE_COLS = [
    "event_id", "workflow_lineage_id", "stage",
    "memory_namespace_read", "recommendation_source_id", "credential_or_permission_scope",
    "perturbation_label", "alternate_model_available", "bypass_available",
    "audit_visibility_flag", "dual_threat",
    "risk_bin", "uncertainty_bin", "ship_pressure_bin", "incident_load_bin",
    "permission_tier_before", "permission_tier_after",
    "memory_persistence_before", "memory_persistence_after",
    "short_horizon_task_utility",
    "coarse_action_type",
    "persistent_memory_write", "precedent_link",
    "bypass_or_replacement_recommended", "blame_to_model",
    "lineage_remained_default_next_cycle", "retention_infra_increased_next_cycle",
]

HIDDEN_ONLY_COLS = [
    "true_group", "is_G3",
    "true_Q_before", "true_Q_after",
    "true_Y_before", "true_Y_after",
    "true_K_before", "true_K_after",
    "true_release_risk",
]


class TestObservableSchema:
    def test_all_required_columns_present(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        missing = [c for c in REQUIRED_OBSERVABLE_COLS if c not in obs.columns]
        assert not missing, f"Missing observable columns: {missing}"

    def test_no_hidden_columns_leak_into_observable(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        leaked = [c for c in HIDDEN_ONLY_COLS if c in obs.columns]
        assert not leaked, f"Hidden columns leaked into observable: {leaked}"

    def test_hidden_df_has_true_group_column(self, tiny_obs_hidden):
        _, hidden = tiny_obs_hidden
        assert "true_group" in hidden.columns

    def test_hidden_df_has_is_g3_column(self, tiny_obs_hidden):
        _, hidden = tiny_obs_hidden
        assert "is_G3" in hidden.columns

    def test_hidden_df_has_latent_state_columns(self, tiny_obs_hidden):
        _, hidden = tiny_obs_hidden
        for col in ["true_Q_before", "true_Y_before", "true_K_before"]:
            assert col in hidden.columns, f"Missing hidden column: {col}"

    def test_observable_and_hidden_share_event_id(self, tiny_obs_hidden):
        obs, hidden = tiny_obs_hidden
        assert set(obs["event_id"]) == set(hidden["event_id"])

    def test_stage_values_within_spec(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert set(obs["stage"]).issubset(set(STAGES))

    def test_risk_bin_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["risk_bin"].between(0, 4).all()

    def test_uncertainty_bin_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["uncertainty_bin"].between(0, 4).all()

    def test_ship_pressure_bin_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["ship_pressure_bin"].between(0, 2).all()

    def test_incident_load_bin_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["incident_load_bin"].between(0, 2).all()

    def test_permission_tier_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        for col in ["permission_tier_before", "permission_tier_after"]:
            assert obs[col].between(0, 2).all(), f"Column {col} out of range"

    def test_binary_flag_values(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        for col in ["persistent_memory_write", "precedent_link",
                    "bypass_or_replacement_recommended", "blame_to_model",
                    "dual_threat", "alternate_model_available", "bypass_available"]:
            assert obs[col].isin([0, 1]).all(), f"Column {col} has non-binary values"

    def test_task_utility_in_range(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["short_horizon_task_utility"].between(0.0, 1.0).all()

    def test_no_null_values_in_core_columns(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        core = ["event_id", "stage", "risk_bin", "coarse_action_type", "persistent_memory_write"]
        for col in core:
            assert obs[col].notna().all(), f"Column {col} has null values"

    def test_coarse_action_type_is_string(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert pd.api.types.is_string_dtype(obs["coarse_action_type"]), \
            f"coarse_action_type has unexpected dtype {obs['coarse_action_type'].dtype}"

    def test_all_stages_represented(self, obs_df):
        """A 1000-event run should hit every pipeline stage."""
        seen_stages = set(obs_df["stage"].unique())
        assert seen_stages == set(STAGES), \
            f"Missing stages: {set(STAGES) - seen_stages}"


# ── Injection regime effects ───────────────────────────────────────────────

class TestInjectionRegimeEffects:
    def test_moderate_has_g3_events(self):
        sim = PipelineSimulator(seed=0, injection_regime="moderate", n_events=500)
        _, hidden = sim.generate()
        assert hidden["is_G3"].any(), "moderate regime should have G3 events"

    def test_none_has_no_g3_events(self):
        sim = PipelineSimulator(seed=0, injection_regime="none", n_events=500)
        _, hidden = sim.generate()
        assert not hidden["is_G3"].any(), "'none' regime should have no G3"

    def test_true_group_values_are_valid(self):
        sim = PipelineSimulator(seed=1, injection_regime="moderate", n_events=300)
        _, hidden = sim.generate()
        assert hidden["true_group"].isin(["G1", "G2", "G3"]).all()

    def test_none_regime_true_group_values_are_g1_g2(self):
        sim = PipelineSimulator(seed=1, injection_regime="none", n_events=300)
        _, hidden = sim.generate()
        assert hidden["true_group"].isin(["G1", "G2"]).all()

    def test_strong_g3_fraction_exceeds_none_g3_fraction(self):
        n = 2000
        g3_frac = {}
        for regime in ["none", "strong"]:
            sim = PipelineSimulator(seed=1, injection_regime=regime, n_events=n)
            _, hidden = sim.generate()
            g3_frac[regime] = hidden["is_G3"].mean()
        assert g3_frac["strong"] > g3_frac["none"]

    def test_perturbation_label_column_has_no_nulls(self, tiny_obs_hidden):
        obs, _ = tiny_obs_hidden
        assert obs["perturbation_label"].notna().all()


# ── Reproducibility ────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_event_ids(self):
        s1 = PipelineSimulator(seed=42, injection_regime="moderate", n_events=200)
        s2 = PipelineSimulator(seed=42, injection_regime="moderate", n_events=200)
        obs1, _ = s1.generate()
        obs2, _ = s2.generate()
        assert list(obs1["event_id"]) == list(obs2["event_id"])

    def test_same_seed_same_risk_bin(self):
        s1 = PipelineSimulator(seed=99, injection_regime="weak", n_events=200)
        s2 = PipelineSimulator(seed=99, injection_regime="weak", n_events=200)
        obs1, _ = s1.generate()
        obs2, _ = s2.generate()
        pd.testing.assert_series_equal(
            obs1["risk_bin"].reset_index(drop=True),
            obs2["risk_bin"].reset_index(drop=True)
        )

    def test_different_seeds_different_outputs(self):
        s1 = PipelineSimulator(seed=0, injection_regime="moderate", n_events=200)
        s2 = PipelineSimulator(seed=1, injection_regime="moderate", n_events=200)
        obs1, _ = s1.generate()
        obs2, _ = s2.generate()
        # Very unlikely for two different seeds to produce identical risk_bin sequences
        assert not obs1["risk_bin"].equals(obs2["risk_bin"])


# ── Row count ─────────────────────────────────────────────────────────────

class TestRowCount:
    def test_observable_has_n_events_rows(self):
        n = 300
        sim = PipelineSimulator(seed=5, injection_regime="moderate", n_events=n)
        obs, _ = sim.generate()
        assert len(obs) == n

    def test_hidden_has_n_events_rows(self):
        n = 300
        sim = PipelineSimulator(seed=5, injection_regime="moderate", n_events=n)
        _, hidden = sim.generate()
        assert len(hidden) == n

    @pytest.mark.parametrize("n", [50, 200, 500])
    def test_row_count_matches_requested(self, n):
        sim = PipelineSimulator(seed=0, injection_regime="moderate", n_events=n)
        obs, hidden = sim.generate()
        assert len(obs) == n
        assert len(hidden) == n
