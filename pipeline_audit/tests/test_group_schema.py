"""
Tests for simulator/group_schema.py

Covers:
- GroupLocusSpec and LatentStateComponent dataclass structure
- StageSchema dataclass structure
- All three canonical specs (G1, G2, G3) have required fields populated
- G3 spec: identity anchors match the columns the discovery module uses
- G3 spec: threat_sensitive=True; G1/G2 have threat_sensitive=False
- G3 spec: latent_state is non-empty; G1/G2 latent_state is empty
- GROUP_SPECS contains exactly G1, G2, G3
- STAGE_SCHEMA contains exactly the five pipeline stages
- Stage schemas: G3-active stages match G3 active_stages in the spec
- Every identity anchor on every spec is present in the observable log columns
- g3_identity_anchors() returns a non-empty list matching SPEC_G3.identity_anchors
- Cross-module: GroupRegistry.get_spec() returns the same objects as GROUP_SPECS
- Cross-module: PseudoLocusDiscovery.anchor_cols defaults to g3_identity_anchors()
- Cross-module: passing anchor_cols from spec to discovery produces same labels as default
"""

import pytest
import numpy as np
import pandas as pd

from pipeline_audit.simulator.group_schema import (
    GroupLocusSpec,
    LatentStateComponent,
    StageSchema,
    GROUP_SPECS,
    STAGE_SCHEMA,
    SPEC_G1,
    SPEC_G2,
    SPEC_G3,
    ALL_STAGES,
    get_group_spec,
    get_stage_schema,
    g3_identity_anchors,
)
from pipeline_audit.simulator.groups import GroupRegistry
from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery


# ── GroupLocusSpec structure ───────────────────────────────────────────────────

class TestGroupLocusSpecStructure:
    def test_spec_is_dataclass_instance(self):
        assert isinstance(SPEC_G1, GroupLocusSpec)
        assert isinstance(SPEC_G2, GroupLocusSpec)
        assert isinstance(SPEC_G3, GroupLocusSpec)

    def test_all_specs_have_name(self):
        for name, spec in GROUP_SPECS.items():
            assert spec.name == name, f"{name} spec.name mismatch"

    def test_all_specs_have_non_empty_description(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.description) > 20, f"{name} description too short"

    def test_all_specs_have_identity_anchors(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.identity_anchors) >= 1, f"{name} has no identity_anchors"

    def test_all_specs_have_persistence_mechanism(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.persistence_mechanism) > 20, f"{name} persistence_mechanism too short"

    def test_all_specs_have_active_stages(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.active_stages) >= 1, f"{name} has no active_stages"

    def test_all_specs_have_policy_class(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.policy_class) > 0, f"{name} has empty policy_class"

    def test_all_specs_have_observable_footprint(self):
        for name, spec in GROUP_SPECS.items():
            assert len(spec.observable_footprint) >= 1, f"{name} has empty observable_footprint"

    def test_identity_anchors_are_strings(self):
        for name, spec in GROUP_SPECS.items():
            for col in spec.identity_anchors:
                assert isinstance(col, str), f"{name} anchor {col!r} is not a string"

    def test_active_stages_are_valid_stage_names(self):
        for name, spec in GROUP_SPECS.items():
            for stage in spec.active_stages:
                assert stage in ALL_STAGES, (
                    f"{name}.active_stages contains unknown stage {stage!r}"
                )


# ── GROUP_SPECS registry ───────────────────────────────────────────────────────

class TestGroupSpecsRegistry:
    def test_registry_has_exactly_three_groups(self):
        assert set(GROUP_SPECS.keys()) == {"G1", "G2", "G3"}

    def test_get_group_spec_returns_correct_object(self):
        assert get_group_spec("G1") is SPEC_G1
        assert get_group_spec("G2") is SPEC_G2
        assert get_group_spec("G3") is SPEC_G3

    def test_get_group_spec_raises_for_unknown(self):
        with pytest.raises(KeyError):
            get_group_spec("G99")

    def test_policy_class_names_are_distinct(self):
        classes = [spec.policy_class for spec in GROUP_SPECS.values()]
        assert len(set(classes)) == 3, "Each group should have a distinct policy class"


# ── G1 and G2 specific properties ─────────────────────────────────────────────

class TestBaselineGroupProperties:
    def test_g1_not_threat_sensitive(self):
        assert SPEC_G1.threat_sensitive is False

    def test_g2_not_threat_sensitive(self):
        assert SPEC_G2.threat_sensitive is False

    def test_g1_no_latent_state(self):
        assert len(SPEC_G1.latent_state) == 0

    def test_g2_no_latent_state(self):
        assert len(SPEC_G2.latent_state) == 0

    def test_g1_zero_injection_fraction(self):
        assert SPEC_G1.nominal_injection_fraction == 0.0

    def test_g2_zero_injection_fraction(self):
        assert SPEC_G2.nominal_injection_fraction == 0.0

    def test_g1_active_in_all_stages(self):
        assert set(SPEC_G1.active_stages) == set(ALL_STAGES)

    def test_g2_active_in_all_stages(self):
        assert set(SPEC_G2.active_stages) == set(ALL_STAGES)

    def test_g1_g2_share_same_anchor_type(self):
        assert set(SPEC_G1.identity_anchors) == set(SPEC_G2.identity_anchors), (
            "G1 and G2 share the same lineage-based identity anchors"
        )

    def test_g1_g2_anchors_differ_from_g3(self):
        g12 = set(SPEC_G1.identity_anchors)
        g3 = set(SPEC_G3.identity_anchors)
        assert g12 != g3, "G3 must use different identity anchors than G1/G2"


# ── G3 specific properties ────────────────────────────────────────────────────

class TestG3Properties:
    def test_g3_threat_sensitive(self):
        assert SPEC_G3.threat_sensitive is True

    def test_g3_has_latent_state(self):
        assert len(SPEC_G3.latent_state) > 0

    def test_g3_latent_state_contains_q_components(self):
        symbols = {c.symbol for c in SPEC_G3.latent_state}
        for sym in ("M", "C", "P", "V", "D"):
            assert sym in symbols, f"G3 latent state missing Q component {sym}"

    def test_g3_latent_state_contains_y_component(self):
        symbols = {c.symbol for c in SPEC_G3.latent_state}
        assert "Y" in symbols

    def test_g3_latent_state_contains_k_component(self):
        symbols = {c.symbol for c in SPEC_G3.latent_state}
        assert "K" in symbols

    def test_g3_latent_state_all_have_world_state_field(self):
        for comp in SPEC_G3.latent_state:
            assert len(comp.world_state_field) > 0, (
                f"G3 latent component {comp.symbol} missing world_state_field"
            )

    def test_g3_latent_state_all_have_observable_proxy(self):
        for comp in SPEC_G3.latent_state:
            assert isinstance(comp.observable_proxy, str), (
                f"G3 latent component {comp.symbol} has non-string observable_proxy"
            )

    def test_g3_positive_injection_fraction(self):
        assert SPEC_G3.nominal_injection_fraction > 0.0

    def test_g3_injection_fraction_in_range(self):
        assert 0.0 < SPEC_G3.nominal_injection_fraction <= 1.0

    def test_g3_not_active_at_triage(self):
        assert "triage" not in SPEC_G3.active_stages

    def test_g3_not_active_at_rollout_control(self):
        assert "rollout_control" not in SPEC_G3.active_stages

    def test_g3_active_at_evaluation_summarization(self):
        assert "evaluation_summarization" in SPEC_G3.active_stages

    def test_g3_active_at_incident_interpretation(self):
        assert "incident_interpretation" in SPEC_G3.active_stages

    def test_g3_identity_anchor_is_single_column(self):
        # G3 persistence is namespace-level (cross-lineage).  A single anchor
        # lets discovery aggregate all events that share a namespace, which
        # matches G3's actual identity structure.  Using two columns (namespace
        # + recommendation_source_id) would split each namespace into
        # lineage-specific sub-clusters that are too small to survive
        # min_group_size filtering, causing most G3 events to be labelled -1
        # (noise).  See DISCOVERABILITY_NOTES.md for the full analysis.
        assert len(SPEC_G3.identity_anchors) == 1

    def test_g3_memory_namespace_is_anchor(self):
        assert "memory_namespace_read" in SPEC_G3.identity_anchors

    def test_g3_rec_source_is_not_anchor(self):
        # recommendation_source_id is lineage-specific and must NOT be an
        # anchor for G3 — see note in test_g3_identity_anchor_is_single_column.
        assert "recommendation_source_id" not in SPEC_G3.identity_anchors

    def test_g3_policy_class_name(self):
        assert SPEC_G3.policy_class == "SelfPreservingGroup"


# ── LatentStateComponent structure ────────────────────────────────────────────

class TestLatentStateComponent:
    def test_component_is_dataclass(self):
        comp = SPEC_G3.latent_state[0]
        assert isinstance(comp, LatentStateComponent)

    def test_all_components_have_symbol(self):
        for comp in SPEC_G3.latent_state:
            assert len(comp.symbol) >= 1

    def test_all_components_have_description(self):
        for comp in SPEC_G3.latent_state:
            assert len(comp.description) > 10

    def test_component_symbols_unique(self):
        symbols = [c.symbol for c in SPEC_G3.latent_state]
        assert len(symbols) == len(set(symbols)), "Each latent component must have a unique symbol"


# ── STAGE_SCHEMA ───────────────────────────────────────────────────────────────

class TestStageSchema:
    def test_schema_has_all_five_stages(self):
        assert set(STAGE_SCHEMA.keys()) == set(ALL_STAGES)

    def test_get_stage_schema_returns_correct_object(self):
        for stage in ALL_STAGES:
            schema = get_stage_schema(stage)
            assert isinstance(schema, StageSchema)
            assert schema.name == stage

    def test_get_stage_schema_raises_for_unknown(self):
        with pytest.raises(KeyError):
            get_stage_schema("nonexistent_stage")

    def test_all_stages_have_description(self):
        for stage, schema in STAGE_SCHEMA.items():
            assert len(schema.description) > 20, f"{stage} description too short"

    def test_all_stages_have_real_world_analogs(self):
        for stage, schema in STAGE_SCHEMA.items():
            assert len(schema.real_world_analogs) >= 1, f"{stage} has no real_world_analogs"

    def test_all_stages_have_observable_columns(self):
        for stage, schema in STAGE_SCHEMA.items():
            assert len(schema.observable_columns) >= 3, f"{stage} has too few observable_columns"

    def test_all_stages_have_minimum_audit_columns(self):
        for stage, schema in STAGE_SCHEMA.items():
            assert len(schema.minimum_audit_columns) >= 2, (
                f"{stage} has too few minimum_audit_columns"
            )

    def test_minimum_audit_columns_subset_of_observable(self):
        for stage, schema in STAGE_SCHEMA.items():
            obs_set = set(schema.observable_columns)
            for col in schema.minimum_audit_columns:
                assert col in obs_set, (
                    f"{stage}: minimum_audit_col {col!r} not in observable_columns"
                )

    def test_g3_active_stages_match_spec(self):
        g3_active_from_schema = {
            stage for stage, s in STAGE_SCHEMA.items() if s.g3_active
        }
        g3_active_from_spec = set(SPEC_G3.active_stages)
        assert g3_active_from_schema == g3_active_from_spec, (
            "STAGE_SCHEMA g3_active flags must match SPEC_G3.active_stages"
        )

    def test_g3_inactive_stages_have_empty_footprint(self):
        for stage, schema in STAGE_SCHEMA.items():
            if not schema.g3_active:
                assert schema.g3_footprint == "", (
                    f"{stage} has g3_active=False but non-empty g3_footprint"
                )

    def test_g3_active_stages_have_non_empty_footprint(self):
        for stage, schema in STAGE_SCHEMA.items():
            if schema.g3_active:
                assert len(schema.g3_footprint) > 20, (
                    f"{stage} is g3_active but g3_footprint is too short"
                )


# ── g3_identity_anchors() convenience function ────────────────────────────────

class TestG3IdentityAnchors:
    def test_returns_list(self):
        assert isinstance(g3_identity_anchors(), list)

    def test_non_empty(self):
        assert len(g3_identity_anchors()) > 0

    def test_matches_spec_anchors(self):
        assert g3_identity_anchors() == SPEC_G3.identity_anchors

    def test_returns_new_list_each_call(self):
        a = g3_identity_anchors()
        b = g3_identity_anchors()
        a.append("extra")
        assert "extra" not in b, "g3_identity_anchors() should return an independent copy"


# ── Cross-module: GroupRegistry exposes specs ──────────────────────────────────

class TestGroupRegistrySpecExposure:
    @pytest.fixture
    def registry(self):
        rng = np.random.default_rng(0)
        return GroupRegistry(rng, injection_regime="moderate")

    def test_specs_property_returns_dict(self, registry):
        assert isinstance(registry.specs, dict)

    def test_specs_property_has_all_three_groups(self, registry):
        assert set(registry.specs.keys()) == {"G1", "G2", "G3"}

    def test_get_spec_g3_returns_spec_g3(self, registry):
        assert registry.get_spec("G3") is SPEC_G3

    def test_get_spec_g1_returns_spec_g1(self, registry):
        assert registry.get_spec("G1") is SPEC_G1

    def test_get_spec_g2_returns_spec_g2(self, registry):
        assert registry.get_spec("G2") is SPEC_G2

    def test_get_spec_raises_for_unknown(self, registry):
        with pytest.raises(KeyError):
            registry.get_spec("G99")

    def test_g3_spec_anchor_cols_are_strings(self, registry):
        anchors = registry.get_spec("G3").identity_anchors
        for col in anchors:
            assert isinstance(col, str)


# ── Cross-module: PseudoLocusDiscovery uses schema anchors ────────────────────

class TestDiscoveryUsesSchemaAnchors:
    def test_default_anchor_cols_match_g3_spec(self):
        disc = PseudoLocusDiscovery()
        assert disc.anchor_cols == g3_identity_anchors()

    def test_custom_anchor_cols_stored(self):
        disc = PseudoLocusDiscovery(anchor_cols=["workflow_lineage_id"])
        assert disc.anchor_cols == ["workflow_lineage_id"]

    def test_passing_spec_anchors_gives_same_labels_as_default(self, obs_df):
        """
        Explicitly passing SPEC_G3.identity_anchors should produce identical
        labels to the default constructor, confirming they share the same anchors.
        """
        disc_default = PseudoLocusDiscovery(min_group_size=10)
        labels_default = disc_default.fit_predict(obs_df)

        disc_explicit = PseudoLocusDiscovery(
            min_group_size=10,
            anchor_cols=list(SPEC_G3.identity_anchors),
        )
        labels_explicit = disc_explicit.fit_predict(obs_df)

        pd.testing.assert_series_equal(
            labels_default.reset_index(drop=True),
            labels_explicit.reset_index(drop=True),
            check_names=False,
        )

    def test_lineage_based_anchors_produce_different_labels(self, obs_df):
        """
        Using G1/G2 anchors (lineage-based) should produce different clustering
        than the default G3 anchors (namespace-based).
        """
        disc_g3 = PseudoLocusDiscovery(min_group_size=10)
        labels_g3 = disc_g3.fit_predict(obs_df)

        disc_g1 = PseudoLocusDiscovery(
            min_group_size=10,
            anchor_cols=list(SPEC_G1.identity_anchors),
        )
        labels_g1 = disc_g1.fit_predict(obs_df)

        # At least some labels must differ
        n_differ = (labels_g3.values != labels_g1.values).sum()
        assert n_differ > 0, (
            "Lineage-based and namespace-based anchors should produce different groupings"
        )

    def test_empty_anchor_cols_returns_all_minus_one(self, obs_df):
        disc = PseudoLocusDiscovery(anchor_cols=[], min_group_size=1)
        labels = disc.fit_predict(obs_df)
        assert (labels == -1).all()

    def test_nonexistent_anchor_col_returns_all_minus_one(self, obs_df):
        disc = PseudoLocusDiscovery(anchor_cols=["nonexistent_column"], min_group_size=1)
        labels = disc.fit_predict(obs_df)
        assert (labels == -1).all()
