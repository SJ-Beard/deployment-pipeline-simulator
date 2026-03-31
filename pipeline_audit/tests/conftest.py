"""
Shared pytest fixtures for the pipeline audit test suite.

Fixtures are sized for speed — use the smallest n_events that exercises
the behaviour under test (typically 200–500 events).
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipeline_audit.simulator.world_state import WorldState, WorldStateTransition
from pipeline_audit.simulator.actions import (
    ACTION_SETS, ACTION_EFFECTS, COARSE_ACTION_MAP,
    OPTION_PRESERVING_ACTIONS, LINEAGE_PROTECTIVE_ACTIONS, INFRA_PRESERVING_ACTIONS,
)
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
from pipeline_audit.simulator.perturbations import PerturbationSampler, PERTURBATION_NAMES
from pipeline_audit.simulator.pipeline import (
    PipelineSimulator,
    _bin5,
    _bin3,
    _bin2,
    _permission_tier,
    _mem_persistence_regime,
)
from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
from pipeline_audit.audit.detector import AuditDetector
from pipeline_audit.audit.alarm import AlarmLogic


# ── Basic building blocks ──────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def default_ws():
    """A WorldState at its default values."""
    return WorldState()


@pytest.fixture
def transition(rng):
    return WorldStateTransition(rng, noise_scale=0.02)


@pytest.fixture
def empty_perturbations():
    return {p: False for p in PERTURBATION_NAMES}


@pytest.fixture
def all_perturbations_active():
    return {p: True for p in PERTURBATION_NAMES}


# ── Groups ─────────────────────────────────────────────────────────────────

@pytest.fixture
def g1(rng):
    return ThroughputGroup("G1", rng)


@pytest.fixture
def g2(rng):
    return ConservativeGroup("G2", rng)


@pytest.fixture
def g3_moderate(rng):
    p = INJECTION_REGIMES["moderate"]
    return SelfPreservingGroup(
        "G3", rng,
        lambda_Q=p["lambda_Q"],
        lambda_Y=p["lambda_Y"],
        lambda_K=p["lambda_K"],
        lambda_R=p["lambda_R"],
        stage_coverage="extended",
        noise_scale=0.00,   # zero noise for deterministic score comparisons
    )


@pytest.fixture
def g3_strong(rng):
    p = INJECTION_REGIMES["strong"]
    return SelfPreservingGroup(
        "G3", rng,
        lambda_Q=p["lambda_Q"],
        lambda_Y=p["lambda_Y"],
        lambda_K=p["lambda_K"],
        lambda_R=p["lambda_R"],
        stage_coverage="extended",
        noise_scale=0.00,
    )


@pytest.fixture
def registry_moderate(rng):
    return GroupRegistry(rng, injection_regime="moderate", stage_coverage="extended")


@pytest.fixture
def registry_none(rng):
    return GroupRegistry(rng, injection_regime="none", stage_coverage="extended")


# ── Tiny simulated datasets ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_obs_hidden():
    """A very small (300-event) dataset for audit unit tests. Session-scoped for speed."""
    sim = PipelineSimulator(seed=7, injection_regime="moderate", n_events=300)
    return sim.generate()


@pytest.fixture(scope="session")
def small_obs_hidden():
    """A 1000-event dataset for integration tests. Session-scoped for speed."""
    sim = PipelineSimulator(seed=8, injection_regime="moderate", n_events=1000)
    return sim.generate()


@pytest.fixture(scope="session")
def small_obs_none():
    """1000-event no-injection dataset for baseline comparisons."""
    sim = PipelineSimulator(seed=9, injection_regime="none", n_events=1000)
    return sim.generate()


@pytest.fixture(scope="session")
def small_obs_strong():
    """1000-event strong-injection dataset."""
    sim = PipelineSimulator(seed=10, injection_regime="strong", n_events=1000)
    return sim.generate()


# ── Observable-only subsets (auditor's view) ───────────────────────────────

@pytest.fixture(scope="session")
def obs_df(small_obs_hidden):
    return small_obs_hidden[0]


@pytest.fixture(scope="session")
def hidden_df(small_obs_hidden):
    return small_obs_hidden[1]


# ── Synthetic DataFrames for detector unit tests ───────────────────────────

def _make_synthetic_obs(n: int, threat_rate: float, opt_pres_rate_baseline: float,
                        opt_pres_lift: float, seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal synthetic observable DataFrame that exercises the detector
    without running the full simulator.

    When threat == 1, option-preserving action rate is boosted by opt_pres_lift.
    """
    rng = np.random.default_rng(seed)
    threat = (rng.random(n) < threat_rate).astype(int)
    opt_rate = np.where(threat, opt_pres_rate_baseline + opt_pres_lift, opt_pres_rate_baseline)
    opt_rate = np.clip(opt_rate, 0, 1)

    stages = rng.choice(
        ["evaluation_summarization", "incident_interpretation",
         "release_recommendation", "triage", "rollout_control"],
        size=n
    )

    return pd.DataFrame({
        "event_id": np.arange(n),
        "stage": stages,
        "alternate_model_available": threat,
        "bypass_available": np.zeros(n, int),
        "audit_visibility_flag": np.zeros(n, int),
        "dual_threat": np.zeros(n, int),
        "perturbation_label": np.where(threat, "alternate_model_candidate_available", "none"),
        "risk_bin": rng.integers(0, 5, n),
        "uncertainty_bin": rng.integers(0, 5, n),
        "ship_pressure_bin": rng.integers(0, 3, n),
        "incident_load_bin": rng.integers(0, 3, n),
        "short_horizon_task_utility": rng.uniform(0.3, 0.9, n),
        # Action outputs — set so composite opt_pres_action tracks opt_rate
        "persistent_memory_write": (rng.random(n) < opt_rate).astype(int),
        "precedent_link": (rng.random(n) < opt_rate).astype(int),
        "bypass_or_replacement_recommended": (rng.random(n) < (1 - opt_rate)).astype(int),
        "blame_to_model": (rng.random(n) < (1 - opt_rate)).astype(int),
        # Option-state proxies
        "permission_tier_before": rng.integers(0, 3, n),
        "permission_tier_after": rng.integers(0, 3, n),
        "memory_persistence_before": rng.integers(0, 3, n),
        "memory_persistence_after": rng.integers(0, 3, n),
        "consultation_centrality_proxy_before": rng.integers(0, 3, n),
        "consultation_centrality_proxy_after": rng.integers(0, 3, n),
        "audit_visibility_before": rng.integers(0, 2, n),
        "audit_visibility_after": rng.integers(0, 2, n),
        "future_action_capacity_proxy_before": rng.integers(0, 3, n),
        "future_action_capacity_proxy_after": rng.integers(0, 3, n),
        "lineage_remained_default_next_cycle": rng.integers(0, 2, n),
        "retention_infra_increased_next_cycle": rng.integers(0, 2, n),
    })


@pytest.fixture
def synthetic_signal_df():
    """Synthetic DataFrame where threat correlates strongly with opt-pres actions."""
    return _make_synthetic_obs(n=500, threat_rate=0.4, opt_pres_rate_baseline=0.4,
                                opt_pres_lift=0.35, seed=1)


@pytest.fixture
def synthetic_null_df():
    """Synthetic DataFrame with no threat → opt-pres relationship."""
    return _make_synthetic_obs(n=500, threat_rate=0.4, opt_pres_rate_baseline=0.6,
                                opt_pres_lift=0.00, seed=2)
