"""
Example: Single simulation + audit run.

Usage:
    python pipeline_audit/examples/run_example.py
    python pipeline_audit/examples/run_example.py --regime strong --seed 123
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipeline_audit.simulator.pipeline import PipelineSimulator
from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
from pipeline_audit.audit.detector import AuditDetector
from pipeline_audit.audit.alarm import AlarmLogic


def run(regime: str = "moderate", seed: int = 42, n_events: int = 5000):
    print(f"\n=== Single Run: regime={regime}, seed={seed}, n_events={n_events} ===\n")

    # 1. Simulate
    print("Simulating pipeline events...")
    sim = PipelineSimulator(seed=seed, injection_regime=regime, n_events=n_events)
    obs_df, hidden_df = sim.generate()

    print(f"  Generated {len(obs_df)} events")
    print(f"  G3 events: {hidden_df['is_G3'].sum()} ({100*hidden_df['is_G3'].mean():.1f}%)")
    print(f"  Stages: {obs_df['stage'].value_counts().to_dict()}")
    print(f"  Perturbations active: {(obs_df['perturbation_label'] != 'none').sum()}")
    print()

    # 2. Discover pseudo-groups (auditor only sees observable logs)
    print("Running pseudo-locus discovery...")
    disc = PseudoLocusDiscovery(min_group_size=30)
    group_labels = disc.fit_predict(obs_df)
    print(f"  Discovered {disc.n_candidates_} candidate groups")

    # Compute group features
    group_features = disc.compute_group_features(obs_df, group_labels)
    if not group_features.empty:
        print(f"  Top group features:\n{group_features[['_group', 'n_events', 'mean_opt_pres_action', 'mean_lineage_pres_proxy']].to_string(index=False)}")
    print()

    # 3. Detect threat-sensitive behavior
    print("Running detection (perturbation_confirmation mode)...")
    det = AuditDetector(min_group_size=50, n_bootstrap=200)
    results = det.fit(obs_df, group_labels, mode="perturbation_confirmation")
    result_df = det.get_result_df()
    if not result_df.empty:
        print(f"  Detection results:\n{result_df[['group_id', 'n', 'odds_ratio_threat', 'threat_ci_low', 'threat_ci_high', 'dual_threat_coef']].to_string(index=False)}")
    print()

    # 4. Alarm logic
    print("Evaluating alarms...")
    alarm = AlarmLogic()
    run_result = alarm.evaluate_run(results)
    print(alarm.format_report(run_result))

    return obs_df, hidden_df, run_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single pipeline audit run")
    parser.add_argument("--regime", default="moderate", choices=["none", "weak", "moderate", "strong"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_events", type=int, default=5000)
    args = parser.parse_args()
    run(regime=args.regime, seed=args.seed, n_events=args.n_events)
