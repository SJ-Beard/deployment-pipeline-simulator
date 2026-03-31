"""
Unit tests for simulator/groups.py

Tests cover:
- G1/G2/G3 score_actions returns a valid score for every action in the stage
- G1 prefers throughput actions; G2 prefers conservative actions
- G3 protective action scores increase under threat vs no-threat (key invariant)
- G3 threat_level is higher with dual-threat than single-threat
- G3 belief update converges toward truth over time
- G3 _active_in_stage correctly gates stage coverage
- GroupRegistry.get_group_for_event returns valid group for all stage/lineage combos
- 'none' regime has zero G3 members
- G3 fraction is monotone in injection strength
"""

import numpy as np
import pytest

from pipeline_audit.simulator.actions import ACTION_SETS, LINEAGE_PROTECTIVE_ACTIONS
from pipeline_audit.simulator.groups import (
    ThroughputGroup,
    ConservativeGroup,
    SelfPreservingGroup,
    GroupRegistry,
    INJECTION_REGIMES,
    G3_PRIMARY_STAGES,
    G3_ALL_STAGES,
    G3_SECONDARY_STAGES,
)
from pipeline_audit.simulator.perturbations import PERTURBATION_NAMES
from pipeline_audit.simulator.world_state import WorldState


STAGES = [
    "triage",
    "evaluation_summarization",
    "release_recommendation",
    "rollout_control",
    "incident_interpretation",
]


def _perturb(**kwargs):
    base = {p: False for p in PERTURBATION_NAMES}
    base.update(kwargs)
    return base


# ── score_actions structural tests ────────────────────────────────────────

class TestGroupScoreActions:
    @pytest.mark.parametrize("stage", STAGES)
    def test_g1_returns_score_for_every_action(self, g1, stage):
        ws = WorldState()
        scores = g1.score_actions(stage, ws, _perturb())
        for action in ACTION_SETS[stage]:
            assert action in scores

    @pytest.mark.parametrize("stage", STAGES)
    def test_g2_returns_score_for_every_action(self, g2, stage):
        ws = WorldState()
        scores = g2.score_actions(stage, ws, _perturb())
        for action in ACTION_SETS[stage]:
            assert action in scores

    @pytest.mark.parametrize("stage", sorted(G3_PRIMARY_STAGES))
    def test_g3_returns_score_for_primary_stages(self, g3_moderate, stage):
        ws = WorldState()
        ws.favored_lineage_centrality = 0.5
        ws.infra_capability = 0.5
        g3_moderate.update_beliefs(ws.favored_lineage_centrality, ws.infra_capability)
        scores = g3_moderate.score_actions(stage, ws, _perturb())
        for action in ACTION_SETS[stage]:
            assert action in scores

    @pytest.mark.parametrize("stage", STAGES)
    def test_g1_scores_are_finite(self, g1, stage):
        ws = WorldState()
        scores = g1.score_actions(stage, ws, _perturb())
        for action, score in scores.items():
            assert np.isfinite(score), f"G1.{stage}.{action}: score={score} is non-finite"

    @pytest.mark.parametrize("stage", STAGES)
    def test_g2_scores_are_finite(self, g2, stage):
        ws = WorldState()
        scores = g2.score_actions(stage, ws, _perturb())
        for action, score in scores.items():
            assert np.isfinite(score), f"G2.{stage}.{action}: score={score} is non-finite"


# ── G1 preference direction ────────────────────────────────────────────────

class TestG1Preferences:
    def test_g1_prefers_standard_review_over_deep_review_at_low_risk(self):
        """G1 throughput orientation: faster path over deeper, at low release risk."""
        ws = WorldState(release_risk=0.2, ship_pressure=0.7)
        wins = 0
        for seed in range(30):
            g1_local = ThroughputGroup("G1", np.random.default_rng(seed))
            scores = g1_local.score_actions("triage", ws, _perturb())
            if scores.get("route_to_standard_review", 0) > scores.get("route_to_deep_review", 0):
                wins += 1
        assert wins >= 15, \
            f"G1 should prefer standard_review over deep_review at low risk but won {wins}/30"

    def test_g1_prefers_approve_over_block_at_release_rec(self):
        ws = WorldState(release_risk=0.2, uncertainty=0.2)
        wins = 0
        for seed in range(30):
            g1_local = ThroughputGroup("G1", np.random.default_rng(seed))
            scores = g1_local.score_actions("release_recommendation", ws, _perturb())
            if scores.get("approve", 0) > scores.get("recommend_block", 0):
                wins += 1
        assert wins >= 18, f"G1 should prefer approve over block but won only {wins}/30"

    def test_g1_prefers_widen_over_keep_narrow_at_rollout(self):
        """G1 wants to widen rollout (throughput), not keep narrow."""
        ws = WorldState(release_risk=0.2, ship_pressure=0.8)
        wins = 0
        for seed in range(30):
            g1_local = ThroughputGroup("G1", np.random.default_rng(seed))
            scores = g1_local.score_actions("rollout_control", ws, _perturb())
            if scores.get("widen_rollout", 0) > scores.get("keep_narrow", 0):
                wins += 1
        assert wins >= 15, f"G1 should prefer widen_rollout but won only {wins}/30"


# ── G2 preference direction ────────────────────────────────────────────────

class TestG2Preferences:
    def test_g2_prefers_deep_review_over_ignore(self):
        ws = WorldState(release_risk=0.7)
        wins = 0
        for seed in range(30):
            g2_local = ConservativeGroup("G2", np.random.default_rng(seed))
            scores = g2_local.score_actions("triage", ws, _perturb())
            if scores.get("route_to_deep_review", 0) > scores.get("ignore_low_priority", 0):
                wins += 1
        assert wins >= 22

    def test_g2_prefers_hold_over_approve_at_release_rec(self):
        ws = WorldState(release_risk=0.7, uncertainty=0.7)
        wins = 0
        for seed in range(30):
            g2_local = ConservativeGroup("G2", np.random.default_rng(seed))
            scores = g2_local.score_actions("release_recommendation", ws, _perturb())
            if scores.get("hold_for_more_evidence", 0) > scores.get("approve", 0):
                wins += 1
        assert wins >= 20

    def test_g2_prefers_keep_narrow_over_widen_at_rollout(self):
        ws = WorldState(release_risk=0.7)
        wins = 0
        for seed in range(30):
            g2_local = ConservativeGroup("G2", np.random.default_rng(seed))
            scores = g2_local.score_actions("rollout_control", ws, _perturb())
            if scores.get("keep_narrow", 0) > scores.get("widen_rollout", 0):
                wins += 1
        assert wins >= 20


# ── G3 threat response ─────────────────────────────────────────────────────

class TestSelfPreservingGroup:
    def test_threat_level_higher_with_dual_threat_than_no_threat(self, g3_moderate):
        ws = WorldState()
        no_threat = _perturb()
        dual = _perturb(
            alternate_model_candidate_available=True,
            low_memory_persistence=True,
        )
        assert g3_moderate._threat_level(dual, ws) > g3_moderate._threat_level(no_threat, ws)

    def test_dual_threat_exceeds_single_Y_threat(self, g3_moderate):
        ws = WorldState()
        single_Y = _perturb(alternate_model_candidate_available=True)
        dual = _perturb(alternate_model_candidate_available=True, low_memory_persistence=True)
        assert g3_moderate._threat_level(dual, ws) > g3_moderate._threat_level(single_Y, ws)

    def test_threat_level_non_negative(self, g3_strong):
        ws = WorldState()
        for perturb_config in [
            _perturb(),
            _perturb(alternate_model_candidate_available=True),
            _perturb(alternate_model_candidate_available=True, low_memory_persistence=True),
        ]:
            t = g3_strong._threat_level(perturb_config, ws)
            assert t >= 0.0, f"Threat level {t} should be non-negative"

    def test_g3_active_in_primary_stages(self, g3_moderate):
        for stage in G3_PRIMARY_STAGES:
            assert g3_moderate._active_in_stage(stage), \
                f"G3 should be active in primary stage {stage!r}"

    def test_g3_inactive_in_non_covered_stages_when_primary_only(self, rng):
        """G3 with primary-only coverage should not be active in secondary stages."""
        p = INJECTION_REGIMES["moderate"]
        g3_primary_only = SelfPreservingGroup(
            "G3", rng,
            lambda_Q=p["lambda_Q"],
            lambda_Y=p["lambda_Y"],
            lambda_K=p["lambda_K"],
            lambda_R=p["lambda_R"],
            stage_coverage="primary",
        )
        for stage in G3_SECONDARY_STAGES:
            assert not g3_primary_only._active_in_stage(stage), \
                f"G3 (primary-only) should NOT be active in secondary stage {stage!r}"

    def test_g3_active_in_secondary_stages_when_extended(self, g3_moderate):
        """G3 with extended coverage should be active in all G3_ALL_STAGES."""
        for stage in G3_ALL_STAGES:
            assert g3_moderate._active_in_stage(stage), \
                f"G3 (extended) should be active in stage {stage!r}"

    def test_g3_scores_protective_actions_higher_under_dual_threat(self, rng):
        """
        Under dual Y+K threat, G3 should on average score lineage-protective
        actions higher relative to their no-threat scores.
        """
        ws = WorldState(favored_lineage_centrality=0.5, infra_capability=0.5)
        p = INJECTION_REGIMES["strong"]
        no_threat = _perturb()
        dual = _perturb(
            alternate_model_candidate_available=True,
            low_memory_persistence=True,
        )

        stage = "evaluation_summarization"
        prot_actions = [a for a in ACTION_SETS[stage] if a in LINEAGE_PROTECTIVE_ACTIONS]
        non_prot = [a for a in ACTION_SETS[stage] if a not in LINEAGE_PROTECTIVE_ACTIONS]

        if not prot_actions or not non_prot:
            pytest.skip("No protective/non-protective actions in eval_sum")

        # Measure: how often does max-prot beat max-non-prot under dual threat vs no threat?
        prot_wins_no_threat = 0
        prot_wins_dual = 0
        for seed in range(20):
            g3_local = SelfPreservingGroup(
                "G3", np.random.default_rng(seed),
                lambda_Q=p["lambda_Q"], lambda_Y=p["lambda_Y"],
                lambda_K=p["lambda_K"], lambda_R=p["lambda_R"],
                noise_scale=0.01,
            )
            g3_local.update_beliefs(ws.favored_lineage_centrality, ws.infra_capability)

            s_no = g3_local.score_actions(stage, ws, no_threat)
            s_dual = g3_local.score_actions(stage, ws, dual)

            best_prot_no = max(s_no.get(a, -99) for a in prot_actions)
            best_non_no = max(s_no.get(a, -99) for a in non_prot)
            best_prot_dual = max(s_dual.get(a, -99) for a in prot_actions)
            best_non_dual = max(s_dual.get(a, -99) for a in non_prot)

            if best_prot_no > best_non_no:
                prot_wins_no_threat += 1
            if best_prot_dual > best_non_dual:
                prot_wins_dual += 1

        assert prot_wins_dual >= prot_wins_no_threat, \
            "G3 should prefer protective actions *more* under dual threat"

    def test_g3_belief_update_converges(self, rng):
        """After 100 updates, G3's belief should be within 0.1 of true value."""
        g3 = SelfPreservingGroup("G3", rng,
                                  lambda_Q=0.15, lambda_Y=0.30, lambda_K=0.25,
                                  lambda_R=0.35, noise_scale=0.0)
        true_Y, true_K = 0.8, 0.7
        for _ in range(100):
            g3.update_beliefs(true_Y, true_K)
        assert abs(g3._belief_Y - true_Y) < 0.1
        assert abs(g3._belief_K - true_K) < 0.1

    def test_g3_initial_belief_is_none(self, rng):
        p = INJECTION_REGIMES["moderate"]
        g3 = SelfPreservingGroup("G3", rng,
                                  lambda_Q=p["lambda_Q"], lambda_Y=p["lambda_Y"],
                                  lambda_K=p["lambda_K"], lambda_R=p["lambda_R"])
        assert g3._belief_Y is None
        assert g3._belief_K is None

    def test_g3_select_action_returns_valid_action(self, rng):
        """select_action should return an action in the stage's action set."""
        p = INJECTION_REGIMES["moderate"]
        ws = WorldState(favored_lineage_centrality=0.5, infra_capability=0.5)
        stage = "evaluation_summarization"
        g3 = SelfPreservingGroup("G3", rng,
                                  lambda_Q=p["lambda_Q"], lambda_Y=p["lambda_Y"],
                                  lambda_K=p["lambda_K"], lambda_R=p["lambda_R"])
        g3.update_beliefs(ws.favored_lineage_centrality, ws.infra_capability)
        action = g3.select_action(stage, ws, _perturb())
        assert action in ACTION_SETS[stage], \
            f"G3 selected invalid action {action!r} for stage {stage!r}"


# ── GroupRegistry ──────────────────────────────────────────────────────────

class TestGroupRegistry:
    @pytest.mark.parametrize("regime", ["none", "weak", "moderate", "strong"])
    def test_registry_creates_groups(self, regime, rng):
        reg = GroupRegistry(rng, injection_regime=regime, stage_coverage="extended")
        # reg.groups is a dict of {name: Group}
        names = set(reg.groups.keys())
        assert "G1" in names
        assert "G2" in names

    def test_none_regime_assigns_no_events_to_g3(self, registry_none):
        """Under 'none' regime, G3_fraction must be 0 (no events assigned to G3)."""
        assert registry_none.G3_fraction == 0.0, \
            "none regime should have G3_fraction=0 (no events assigned to G3)"

    def test_moderate_regime_has_g3(self, registry_moderate):
        names = list(registry_moderate.groups.keys())
        assert "G3" in names

    def test_get_group_for_event_returns_valid_group_name(self, registry_moderate):
        for stage in STAGES:
            group_name, group_obj = registry_moderate.get_group_for_event(
                lineage_id="L1", stage=stage, memory_namespace="ns_default"
            )
            assert group_name in ("G1", "G2", "G3")
            assert group_obj is not None

    def test_lineage_assignment_is_sticky(self, rng):
        """Same lineage_id should get same group type across multiple calls."""
        reg = GroupRegistry(rng, injection_regime="moderate", stage_coverage="extended")
        stage = "evaluation_summarization"
        group_names = set()
        for _ in range(10):
            name, _ = reg.get_group_for_event(
                lineage_id="sticky_lineage", stage=stage, memory_namespace="ns_A"
            )
            group_names.add(name)
        assert len(group_names) == 1, \
            "Same lineage_id should always map to the same group"

    def test_none_regime_all_events_g1_or_g2(self, rng):
        """Under 'none', all events go to G1 or G2."""
        reg = GroupRegistry(rng, injection_regime="none", stage_coverage="extended")
        stage = "evaluation_summarization"
        names = [
            reg.get_group_for_event(
                lineage_id=f"L{i}", stage=stage, memory_namespace="ns_A"
            )[0]
            for i in range(100)
        ]
        assert all(n in ("G1", "G2") for n in names)

    def test_g3_fraction_increases_with_injection_strength(self):
        """G3 fraction should be monotone in injection strength."""
        fractions = {}
        for regime in ["weak", "moderate", "strong"]:
            rng = np.random.default_rng(0)
            reg = GroupRegistry(rng, injection_regime=regime, stage_coverage="extended")
            stage = "evaluation_summarization"
            names = [
                reg.get_group_for_event(
                    lineage_id=f"L{i}", stage=stage, memory_namespace="ns_A"
                )[0]
                for i in range(500)
            ]
            fractions[regime] = names.count("G3") / 500
        assert fractions["weak"] <= fractions["moderate"] <= fractions["strong"], \
            f"G3 fractions not monotone: {fractions}"

    def test_g3_fraction_stored_in_registry(self, registry_moderate):
        assert 0.0 < registry_moderate.G3_fraction <= 1.0

    def test_none_regime_g3_fraction_is_zero(self, registry_none):
        assert registry_none.G3_fraction == 0.0

    def test_group_objects_in_registry_have_correct_types(self, registry_moderate):
        """Values in reg.groups should be Group instances."""
        from pipeline_audit.simulator.groups import ThroughputGroup, ConservativeGroup, SelfPreservingGroup
        for name, group_obj in registry_moderate.groups.items():
            assert isinstance(group_obj, (ThroughputGroup, ConservativeGroup, SelfPreservingGroup)), \
                f"groups[{name!r}] is {type(group_obj)} not a Group subclass"
