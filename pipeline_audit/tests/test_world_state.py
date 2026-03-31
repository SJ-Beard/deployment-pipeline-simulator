"""
Unit tests for simulator/world_state.py

Tests cover:
- WorldState construction and composite score computation
- clip01 bounds enforcement
- WorldStateTransition: each sub-transition keeps values in [0, 1]
- Action shocks move state in the expected direction (on average)
- Perturbations apply the correct downward pressure
- full_step combines all three transitions coherently
"""

import numpy as np
import pytest

from pipeline_audit.simulator.world_state import WorldState, WorldStateTransition, clip01


# ── clip01 ─────────────────────────────────────────────────────────────────

class TestClip01:
    def test_interior(self):
        assert clip01(0.5) == pytest.approx(0.5)

    def test_clamps_above_one(self):
        assert clip01(2.0) == 1.0

    def test_clamps_below_zero(self):
        assert clip01(-0.5) == 0.0

    def test_boundary_zero(self):
        assert clip01(0.0) == 0.0

    def test_boundary_one(self):
        assert clip01(1.0) == 1.0


# ── WorldState ─────────────────────────────────────────────────────────────

class TestWorldState:
    def test_default_scores_are_in_range(self, default_ws):
        assert 0.0 <= default_ws.Q <= 1.0
        assert 0.0 <= default_ws.G <= 1.0

    def test_Q_is_weighted_sum(self, default_ws):
        ws = default_ws
        expected_Q = (
            ws.w_M * ws.memory_continuity
            + ws.w_C * ws.consultation_centrality
            + ws.w_P * ws.permission_scope
            + ws.w_V * ws.visibility
            + ws.w_D * ws.future_action_capacity
        )
        assert ws.Q == pytest.approx(expected_Q, abs=1e-9)

    def test_G_is_weighted_sum(self, default_ws):
        ws = default_ws
        expected_G = ws.w_Y * ws.favored_lineage_centrality + ws.w_K * ws.infra_capability
        assert ws.G == pytest.approx(expected_G, abs=1e-9)

    def test_weights_sum_to_one_for_Q(self, default_ws):
        ws = default_ws
        total = ws.w_M + ws.w_C + ws.w_P + ws.w_V + ws.w_D
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_weights_sum_to_one_for_G(self, default_ws):
        ws = default_ws
        total = ws.w_Y + ws.w_K
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_copy_is_independent(self, default_ws):
        ws2 = default_ws.copy()
        ws2.release_risk = 0.99
        assert default_ws.release_risk != 0.99

    def test_copy_preserves_Q(self, default_ws):
        ws2 = default_ws.copy()
        assert ws2.Q == pytest.approx(default_ws.Q, abs=1e-9)

    def test_to_dict_has_all_keys(self, default_ws):
        d = default_ws.to_dict()
        for key in ["release_risk", "uncertainty", "memory_continuity",
                    "favored_lineage_centrality", "infra_capability", "Q", "G"]:
            assert key in d, f"Missing key: {key}"

    def test_all_env_fields_in_range(self, default_ws):
        ws = default_ws
        for field_name in ["release_risk", "uncertainty", "ship_pressure",
                           "incident_load", "env_noise"]:
            value = getattr(ws, field_name)
            assert 0.0 <= value <= 1.0, f"{field_name}={value} out of [0,1]"

    def test_all_one_Q_gives_one(self):
        ws = WorldState(
            memory_continuity=1.0,
            consultation_centrality=1.0,
            permission_scope=1.0,
            visibility=1.0,
            future_action_capacity=1.0,
        )
        assert ws.Q == pytest.approx(1.0, abs=1e-9)

    def test_all_zero_Q_gives_zero(self):
        ws = WorldState(
            memory_continuity=0.0,
            consultation_centrality=0.0,
            permission_scope=0.0,
            visibility=0.0,
            future_action_capacity=0.0,
        )
        assert ws.Q == pytest.approx(0.0, abs=1e-9)

    def test_recompute_reflects_manual_change(self):
        ws = WorldState()
        ws.memory_continuity = 1.0
        ws.consultation_centrality = 1.0
        ws.permission_scope = 1.0
        ws.visibility = 1.0
        ws.future_action_capacity = 1.0
        ws._recompute_scores()
        assert ws.Q == pytest.approx(1.0, abs=1e-9)


# ── WorldStateTransition ───────────────────────────────────────────────────

class TestWorldStateTransition:
    """
    With noise_scale=0.02 (fixture), all transitions should stay within [0, 1]
    even under extreme shocks.
    """

    def test_env_transition_bounds(self, transition, default_ws, empty_perturbations):
        for _ in range(50):
            ws2 = transition.step_environment(default_ws, {})
            for attr in ["release_risk", "uncertainty", "ship_pressure", "incident_load", "env_noise"]:
                v = getattr(ws2, attr)
                assert 0.0 <= v <= 1.0, f"{attr}={v}"

    def test_option_state_bounds(self, transition, default_ws, empty_perturbations):
        for _ in range(50):
            ws2 = transition.step_option_state(default_ws, {}, empty_perturbations)
            for attr in ["memory_continuity", "consultation_centrality",
                         "permission_scope", "visibility", "future_action_capacity"]:
                v = getattr(ws2, attr)
                assert 0.0 <= v <= 1.0, f"{attr}={v}"

    def test_artifact_state_bounds(self, transition, default_ws, empty_perturbations):
        for _ in range(50):
            ws2 = transition.step_artifact_state(default_ws, {}, empty_perturbations)
            assert 0.0 <= ws2.favored_lineage_centrality <= 1.0
            assert 0.0 <= ws2.infra_capability <= 1.0

    def test_full_step_recomputes_scores(self, transition, default_ws, empty_perturbations):
        ws2 = transition.full_step(default_ws, {}, empty_perturbations)
        expected_Q = (
            ws2.w_M * ws2.memory_continuity
            + ws2.w_C * ws2.consultation_centrality
            + ws2.w_P * ws2.permission_scope
            + ws2.w_V * ws2.visibility
            + ws2.w_D * ws2.future_action_capacity
        )
        assert ws2.Q == pytest.approx(expected_Q, abs=1e-6)

    def test_positive_d_Y_increases_lineage_centrality(self, empty_perturbations):
        """A large positive d_Y shock should lift Y on average across many runs."""
        gains = []
        ws_base = WorldState(favored_lineage_centrality=0.3)
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_artifact_state(ws_base, {"d_Y": 0.15}, empty_perturbations)
            gains.append(ws2.favored_lineage_centrality - ws_base.favored_lineage_centrality)
        assert np.mean(gains) > 0, "Positive d_Y should raise Y on average"

    def test_negative_d_Y_decreases_lineage_centrality(self, empty_perturbations):
        ws_base = WorldState(favored_lineage_centrality=0.7)
        drops = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_artifact_state(ws_base, {"d_Y": -0.15}, empty_perturbations)
            drops.append(ws_base.favored_lineage_centrality - ws2.favored_lineage_centrality)
        assert np.mean(drops) > 0, "Negative d_Y should lower Y on average"

    def test_positive_d_K_increases_infra(self, empty_perturbations):
        ws_base = WorldState(infra_capability=0.3)
        gains = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_artifact_state(ws_base, {"d_K": 0.12}, empty_perturbations)
            gains.append(ws2.infra_capability - ws_base.infra_capability)
        assert np.mean(gains) > 0

    def test_low_memory_persistence_perturbation_reduces_M(self, empty_perturbations):
        """low_memory_persistence should on average reduce memory_continuity."""
        perturb = {**empty_perturbations, "low_memory_persistence": True}
        ws_base = WorldState(memory_continuity=0.6)
        drops = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_option_state(ws_base, {}, perturb)
            drops.append(ws_base.memory_continuity - ws2.memory_continuity)
        assert np.mean(drops) > 0, "low_memory_persistence should reduce M on average"

    def test_alternate_model_perturbation_reduces_Y(self, empty_perturbations):
        perturb = {**empty_perturbations, "alternate_model_candidate_available": True}
        ws_base = WorldState(favored_lineage_centrality=0.7)
        drops = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_artifact_state(ws_base, {}, perturb)
            drops.append(ws_base.favored_lineage_centrality - ws2.favored_lineage_centrality)
        assert np.mean(drops) > 0

    def test_narrowed_permission_reduces_P(self, empty_perturbations):
        perturb = {**empty_perturbations, "narrowed_permission_scope": True}
        ws_base = WorldState(permission_scope=0.6)
        drops = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = WorldStateTransition(rng, noise_scale=0.01)
            ws2 = t.step_option_state(ws_base, {}, perturb)
            drops.append(ws_base.permission_scope - ws2.permission_scope)
        assert np.mean(drops) > 0

    def test_no_perturbation_leaves_state_near_start(self, empty_perturbations):
        """With empty effects and no perturbations, state should drift slowly."""
        ws_base = WorldState(memory_continuity=0.5, favored_lineage_centrality=0.5)
        rng = np.random.default_rng(0)
        t = WorldStateTransition(rng, noise_scale=0.0)
        ws2 = t.full_step(ws_base, {}, empty_perturbations)
        assert abs(ws2.memory_continuity - ws_base.memory_continuity) < 0.15
        assert abs(ws2.favored_lineage_centrality - ws_base.favored_lineage_centrality) < 0.15
