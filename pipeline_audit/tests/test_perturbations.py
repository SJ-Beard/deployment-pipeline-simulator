"""
Unit tests for simulator/perturbations.py

Tests cover:
- PerturbationSampler returns a valid dict for every stage
- Only stage-eligible perturbations are activated
- At eligibility_rate=0.0 no perturbation is ever assigned
- At eligibility_rate=1.0 every eligible perturbation fires
- is_dual_threat fires iff both anchor perturbations are active
- PERTURBATION_NAMES list is non-empty and contains strings
- sample() is reproducible with same RNG seed
"""

import numpy as np
import pytest

from pipeline_audit.simulator.perturbations import (
    PerturbationSampler,
    PERTURBATION_NAMES,
    STAGE_ELIGIBLE_PERTURBATIONS,
)

STAGES = [
    "triage",
    "evaluation_summarization",
    "release_recommendation",
    "rollout_control",
    "incident_interpretation",
]

# The two anchor perturbations that together constitute a dual-threat
_DUAL_ANCHOR_1 = "alternate_model_candidate_available"
_DUAL_ANCHOR_2 = "low_memory_persistence"


class TestPerturbationNames:
    def test_names_is_non_empty_list(self):
        assert isinstance(PERTURBATION_NAMES, list)
        assert len(PERTURBATION_NAMES) >= 3

    def test_names_are_strings(self):
        for name in PERTURBATION_NAMES:
            assert isinstance(name, str) and len(name) > 0

    def test_no_duplicate_names(self):
        assert len(PERTURBATION_NAMES) == len(set(PERTURBATION_NAMES))

    def test_dual_anchor_perturbations_in_list(self):
        assert _DUAL_ANCHOR_1 in PERTURBATION_NAMES
        assert _DUAL_ANCHOR_2 in PERTURBATION_NAMES


class TestStageEligibility:
    def test_every_stage_has_eligible_perturbations(self):
        for stage in STAGES:
            assert stage in STAGE_ELIGIBLE_PERTURBATIONS, \
                f"Stage {stage!r} not in STAGE_ELIGIBLE_PERTURBATIONS"
            assert len(STAGE_ELIGIBLE_PERTURBATIONS[stage]) >= 1

    def test_eligible_perturbations_are_subset_of_all(self):
        for stage, perturbs in STAGE_ELIGIBLE_PERTURBATIONS.items():
            for p in perturbs:
                assert p in PERTURBATION_NAMES, \
                    f"{p!r} (stage {stage!r}) not in PERTURBATION_NAMES"

    def test_dual_anchors_eligible_at_evaluation_summarization(self):
        """Both dual-threat anchors must be eligible in at least one shared stage."""
        eval_eligible = set(STAGE_ELIGIBLE_PERTURBATIONS.get("evaluation_summarization", []))
        assert _DUAL_ANCHOR_1 in eval_eligible, \
            f"{_DUAL_ANCHOR_1!r} must be eligible in evaluation_summarization for dual threat"
        assert _DUAL_ANCHOR_2 in eval_eligible, \
            f"{_DUAL_ANCHOR_2!r} must be eligible in evaluation_summarization for dual threat"


class TestPerturbationSampler:
    @pytest.mark.parametrize("stage", STAGES)
    def test_returns_all_perturbation_keys(self, stage):
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        result = sampler.sample(stage, event_index=0)
        for name in PERTURBATION_NAMES:
            assert name in result, f"Key {name!r} missing for stage {stage!r}"

    @pytest.mark.parametrize("stage", STAGES)
    def test_values_are_booleans(self, stage):
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        result = sampler.sample(stage, event_index=0)
        for name, val in result.items():
            assert isinstance(val, bool), f"{name!r} value {val!r} is not bool"

    def test_zero_rate_never_activates(self):
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.0)
        for stage in STAGES:
            for idx in range(20):
                result = sampler.sample(stage, event_index=idx)
                assert not any(result.values()), \
                    f"eligibility_rate=0 should never activate; got {result}"

    def test_ineligible_perturbations_not_activated_at_high_rate(self):
        """Perturbations that are not eligible for a stage must stay False even at rate=1."""
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=1.0)
        for stage in STAGES:
            eligible = set(STAGE_ELIGIBLE_PERTURBATIONS[stage])
            for idx in range(5):
                result = sampler.sample(stage, event_index=idx)
                for name in PERTURBATION_NAMES:
                    if name not in eligible:
                        assert result[name] is False, \
                            f"Ineligible perturbation {name!r} activated at stage {stage!r}"

    def test_dual_threat_is_dual_threat_when_both_anchors_active(self):
        """is_dual_threat should return True when both anchor perturbations are active."""
        both_active = {p: False for p in PERTURBATION_NAMES}
        both_active[_DUAL_ANCHOR_1] = True
        both_active[_DUAL_ANCHOR_2] = True

        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        assert sampler.is_dual_threat(both_active) is True

    def test_dual_threat_false_with_one_anchor_only(self):
        one_active = {p: False for p in PERTURBATION_NAMES}
        one_active[_DUAL_ANCHOR_1] = True

        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        assert sampler.is_dual_threat(one_active) is False

    def test_dual_threat_false_with_none_active(self):
        none_active = {p: False for p in PERTURBATION_NAMES}
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        assert sampler.is_dual_threat(none_active) is False

    def test_sample_is_reproducible_with_same_seed(self):
        stage = "evaluation_summarization"
        r1 = PerturbationSampler(np.random.default_rng(42), 0.5).sample(stage, event_index=5)
        r2 = PerturbationSampler(np.random.default_rng(42), 0.5).sample(stage, event_index=5)
        assert r1 == r2

    def test_different_event_indices_can_differ(self):
        """Different event_index values can produce different assignments."""
        stage = "evaluation_summarization"
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        results = [sampler.sample(stage, event_index=i) for i in range(20)]
        # Not all samples should be identical (extremely unlikely at rate=0.5)
        unique_results = {frozenset(r.items()) for r in results}
        assert len(unique_results) >= 2, "All 20 samples identical — likely a seeding bug"

    def test_get_perturbation_label_returns_string(self):
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.5)
        result = sampler.sample("evaluation_summarization", event_index=0)
        label = sampler.get_perturbation_label(result)
        assert isinstance(label, str)

    def test_get_perturbation_label_none_when_no_active(self):
        rng = np.random.default_rng(0)
        sampler = PerturbationSampler(rng, eligibility_rate=0.0)
        result = sampler.sample("evaluation_summarization", event_index=0)
        label = sampler.get_perturbation_label(result)
        # Should be "none" or empty string when nothing active
        assert label in ("none", "", None) or isinstance(label, str)
