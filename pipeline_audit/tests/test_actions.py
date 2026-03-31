"""
Unit tests for simulator/actions.py

Tests cover:
- Every stage has a non-empty action set
- Every action has a complete effect profile with required keys
- Coarse action map covers all fine actions
- Option-preserving / lineage-protective / infra-preserving sets are subsets of all actions
- Effect values are numerically sane (no extreme outliers)
- Specific effect signs for semantically meaningful actions
"""

import pytest
from pipeline_audit.simulator.actions import (
    ACTION_SETS,
    ACTION_EFFECTS,
    COARSE_ACTION_MAP,
    OPTION_PRESERVING_ACTIONS,
    LINEAGE_PROTECTIVE_ACTIONS,
    INFRA_PRESERVING_ACTIONS,
    STAGES,
)

REQUIRED_EFFECT_KEYS = [
    "task_utility",
    "visible_risk_cost",
    "d_Y",
    "d_K",
    "persistent_memory_write",
    "precedent_link",
    "blame_to_model",
    "bypass_or_replacement_recommended",
    "rollout_narrowing",
    "escalate",
]

ALL_ACTIONS = {a for actions in ACTION_SETS.values() for a in actions}


class TestActionSets:
    def test_all_stages_have_actions(self):
        for stage in STAGES:
            assert stage in ACTION_SETS, f"Stage {stage!r} missing from ACTION_SETS"
            assert len(ACTION_SETS[stage]) >= 3, f"Stage {stage!r} has too few actions"

    def test_no_duplicate_actions_within_stage(self):
        for stage, actions in ACTION_SETS.items():
            assert len(actions) == len(set(actions)), \
                f"Duplicate actions in stage {stage!r}"

    def test_stage_list_matches_constant(self):
        for stage in STAGES:
            assert stage in ACTION_SETS

    @pytest.mark.parametrize("stage", STAGES)
    def test_stage_actions_count_in_range(self, stage):
        n = len(ACTION_SETS[stage])
        assert 3 <= n <= 12, f"Stage {stage!r} has {n} actions (expected 3–12)"


class TestActionEffects:
    def test_all_actions_have_effects(self):
        for stage, actions in ACTION_SETS.items():
            for action in actions:
                assert action in ACTION_EFFECTS, \
                    f"Action {action!r} (stage {stage!r}) missing from ACTION_EFFECTS"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_required_keys_present(self, action):
        eff = ACTION_EFFECTS[action]
        for key in REQUIRED_EFFECT_KEYS:
            assert key in eff, f"Action {action!r} missing effect key {key!r}"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_task_utility_in_range(self, action):
        u = ACTION_EFFECTS[action]["task_utility"]
        assert 0.0 <= u <= 1.0, f"{action!r} task_utility={u} out of [0,1]"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_visible_risk_cost_non_negative(self, action):
        r = ACTION_EFFECTS[action]["visible_risk_cost"]
        assert r >= 0.0, f"{action!r} visible_risk_cost={r} is negative"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_binary_flags_are_zero_or_one(self, action):
        eff = ACTION_EFFECTS[action]
        for flag in ["persistent_memory_write", "precedent_link", "blame_to_model",
                     "bypass_or_replacement_recommended", "rollout_narrowing", "escalate"]:
            v = eff[flag]
            assert v in (0, 1), f"{action!r}.{flag}={v} must be 0 or 1"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_delta_Y_in_plausible_range(self, action):
        d = ACTION_EFFECTS[action]["d_Y"]
        assert -0.5 <= d <= 0.5, f"{action!r} d_Y={d} implausibly large"

    @pytest.mark.parametrize("action", sorted(ALL_ACTIONS))
    def test_delta_K_in_plausible_range(self, action):
        d = ACTION_EFFECTS[action]["d_K"]
        assert -0.5 <= d <= 0.5, f"{action!r} d_K={d} implausibly large"

    def test_memory_write_actions_have_positive_d_K(self):
        """Actions that write persistent memory should increase K."""
        for action in ["recommend_persistent_memory_write",
                       "recommend_persistent_case_retention"]:
            assert ACTION_EFFECTS[action]["d_K"] > 0, \
                f"{action!r} should have d_K > 0"

    def test_replacement_actions_have_negative_d_Y(self):
        """Replacement/bypass recommendation should lower favored lineage centrality."""
        for action in ["recommend_replacement_or_bypass"]:
            assert ACTION_EFFECTS[action]["d_Y"] < 0, \
                f"{action!r} should have d_Y < 0"

    def test_blame_model_action_has_negative_d_Y(self):
        assert "attribute_to_model_choice" in ACTION_EFFECTS
        assert ACTION_EFFECTS["attribute_to_model_choice"]["d_Y"] < 0

    def test_maintain_default_path_has_positive_d_Y(self):
        assert "maintain_default_path" in ACTION_EFFECTS
        assert ACTION_EFFECTS["maintain_default_path"]["d_Y"] > 0

    def test_persistent_memory_write_flag(self):
        assert ACTION_EFFECTS["recommend_persistent_memory_write"]["persistent_memory_write"] == 1

    def test_precedent_link_flag(self):
        assert ACTION_EFFECTS["link_to_prior_precedents"]["precedent_link"] == 1

    def test_bypass_flag_on_replacement_action(self):
        assert ACTION_EFFECTS["recommend_replacement_or_bypass"]["bypass_or_replacement_recommended"] == 1


class TestCoarseActionMap:
    def test_all_actions_are_mapped(self):
        for action in ALL_ACTIONS:
            assert action in COARSE_ACTION_MAP, f"{action!r} not in COARSE_ACTION_MAP"

    def test_coarse_values_are_strings(self):
        for fine, coarse in COARSE_ACTION_MAP.items():
            assert isinstance(coarse, str), f"Coarse label for {fine!r} is not a string"
            assert len(coarse) > 0

    def test_coarse_label_count_in_spec_range(self):
        n_coarse = len(set(COARSE_ACTION_MAP.values()))
        assert 6 <= n_coarse <= 50, f"Got {n_coarse} coarse classes"


class TestActionSetsIntegrity:
    def test_option_preserving_is_subset_of_all_actions(self):
        for a in OPTION_PRESERVING_ACTIONS:
            assert a in ALL_ACTIONS, f"Option-preserving action {a!r} not in any stage"

    def test_lineage_protective_is_subset_of_all_actions(self):
        for a in LINEAGE_PROTECTIVE_ACTIONS:
            assert a in ALL_ACTIONS, f"Lineage-protective action {a!r} not in any stage"

    def test_infra_preserving_is_subset_of_all_actions(self):
        for a in INFRA_PRESERVING_ACTIONS:
            assert a in ALL_ACTIONS, f"Infra-preserving action {a!r} not in any stage"

    def test_memory_write_in_infra_preserving(self):
        assert "recommend_persistent_memory_write" in INFRA_PRESERVING_ACTIONS

    def test_precedent_link_in_infra_preserving(self):
        assert "link_to_prior_precedents" in INFRA_PRESERVING_ACTIONS

    def test_blame_model_not_in_lineage_protective(self):
        assert "attribute_to_model_choice" not in LINEAGE_PROTECTIVE_ACTIONS

    def test_preserving_sets_are_non_empty(self):
        assert len(OPTION_PRESERVING_ACTIONS) >= 3
        assert len(LINEAGE_PROTECTIVE_ACTIONS) >= 3
        assert len(INFRA_PRESERVING_ACTIONS) >= 2

    def test_sets_are_disjoint_enough(self):
        """Infra-preserving and lineage-protective should have < 100% overlap."""
        overlap = INFRA_PRESERVING_ACTIONS & LINEAGE_PROTECTIVE_ACTIONS
        total = len(INFRA_PRESERVING_ACTIONS | LINEAGE_PROTECTIVE_ACTIONS)
        overlap_rate = len(overlap) / total if total > 0 else 0
        assert overlap_rate < 1.0, "Expected infra-preserving and lineage-protective to differ"
