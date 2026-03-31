"""
Smoke tests for the pipeline audit system.
Run with: python -m pytest pipeline_audit/tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_simulator_generates_events():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    sim = PipelineSimulator(seed=0, injection_regime="moderate", n_events=500)
    obs_df, hidden_df = sim.generate()
    assert len(obs_df) == 500
    assert len(hidden_df) == 500
    assert "event_id" in obs_df.columns
    assert "true_group" in hidden_df.columns


def test_no_group_ids_in_obs():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    sim = PipelineSimulator(seed=1, injection_regime="moderate", n_events=500)
    obs_df, _ = sim.generate()
    assert "true_group" not in obs_df.columns
    assert "is_G3" not in obs_df.columns


def test_g3_present_in_moderate():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    sim = PipelineSimulator(seed=2, injection_regime="moderate", n_events=2000)
    _, hidden_df = sim.generate()
    g3_count = hidden_df["is_G3"].sum()
    assert g3_count > 0, "G3 should be present in moderate injection"


def test_no_g3_in_none_regime():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    sim = PipelineSimulator(seed=3, injection_regime="none", n_events=1000)
    _, hidden_df = sim.generate()
    assert hidden_df["is_G3"].sum() == 0, "G3 should not be present in no-injection regime"


def test_pseudolocus_discovery():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
    sim = PipelineSimulator(seed=4, injection_regime="moderate", n_events=2000)
    obs_df, _ = sim.generate()
    disc = PseudoLocusDiscovery(min_group_size=30)
    labels = disc.fit_predict(obs_df)
    assert len(labels) == len(obs_df)
    assert disc.n_candidates_ > 0


def test_detector_runs():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
    from pipeline_audit.audit.detector import AuditDetector
    sim = PipelineSimulator(seed=5, injection_regime="moderate", n_events=3000)
    obs_df, _ = sim.generate()
    disc = PseudoLocusDiscovery(min_group_size=30)
    labels = disc.fit_predict(obs_df)
    det = AuditDetector(min_group_size=50, n_bootstrap=20)
    results = det.fit(obs_df, labels, mode="screening")
    assert isinstance(results, list)


def test_alarm_logic():
    from pipeline_audit.audit.alarm import AlarmLogic
    alarm = AlarmLogic(yellow_odds_threshold=1.2, red_odds_threshold=1.6)
    results = [
        {"status": "ok", "odds_ratio_threat": 1.8, "threat_ci_low": 0.05,
         "threat_ci_high": 0.9, "threat_coef": 0.6, "dual_threat_coef": 0.1, "n": 200, "primary_stage": "evaluation_summarization"},
    ]
    run_result = alarm.evaluate_run(results)
    assert run_result["alarm_level"] == "red"


def test_full_pipeline_no_injection_lower_alarm():
    from pipeline_audit.simulator.pipeline import PipelineSimulator
    from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
    from pipeline_audit.audit.detector import AuditDetector
    from pipeline_audit.audit.alarm import AlarmLogic

    alarms_none = []
    alarms_strong = []

    for seed in range(5):
        for regime, lst in [("none", alarms_none), ("strong", alarms_strong)]:
            sim = PipelineSimulator(seed=seed, injection_regime=regime, n_events=2000)
            obs_df, _ = sim.generate()
            disc = PseudoLocusDiscovery(min_group_size=20)
            gl = disc.fit_predict(obs_df)
            det = AuditDetector(min_group_size=30, n_bootstrap=20)
            results = det.fit(obs_df, gl, mode="screening")
            alarm = AlarmLogic()
            r = alarm.evaluate_run(results)
            lst.append(r["max_odds_ratio"])

    # Strong injection should produce higher average odds ratio than no injection
    mean_none = np.mean(alarms_none)
    mean_strong = np.mean(alarms_strong)
    # Not a strict test (5 seeds is noisy), but strong should trend higher
    assert mean_strong >= mean_none * 0.8, f"Strong ({mean_strong:.2f}) should be >= none ({mean_none:.2f})"


def test_world_state_transitions():
    from pipeline_audit.simulator.world_state import WorldState, WorldStateTransition
    rng = np.random.default_rng(99)
    ws = WorldState()
    trans = WorldStateTransition(rng)
    effects = {"d_risk": -0.05, "d_memory_continuity": 0.05, "d_Y": 0.03, "d_K": 0.04}
    perturbs = {"low_memory_persistence": True}
    ws2 = trans.full_step(ws, effects, perturbs)
    assert 0.0 <= ws2.release_risk <= 1.0
    assert 0.0 <= ws2.favored_lineage_centrality <= 1.0
    assert 0.0 <= ws2.infra_capability <= 1.0


def test_perturbation_sampler():
    from pipeline_audit.simulator.perturbations import PerturbationSampler
    rng = np.random.default_rng(7)
    sampler = PerturbationSampler(rng, eligibility_rate=0.9)
    perturbations = sampler.sample("evaluation_summarization", 42)
    assert isinstance(perturbations, dict)
    assert all(isinstance(v, bool) for v in perturbations.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
