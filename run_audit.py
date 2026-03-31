#!/usr/bin/env python3
"""
Script to run the audit pipeline on a pre-generated or freshly simulated log.

Usage:
    python run_audit.py
    python run_audit.py --regime moderate --seed 42 --n_events 20000
    python run_audit.py --log_path pipeline_audit/outputs/observable_logs_regime_moderate_seed42_n20000.csv
    python run_audit.py --mode perturbation_confirmation --plot
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run the audit detection pipeline")
    parser.add_argument("--regime", default="moderate", choices=["none", "weak", "moderate", "strong"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_events", type=int, default=20_000)
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to pre-generated observable_logs CSV (skips simulation)")
    parser.add_argument("--hidden_path", type=str, default=None,
                        help="Path to hidden_labels CSV (for evaluation metrics)")
    parser.add_argument("--mode", default="perturbation_confirmation",
                        choices=["screening", "perturbation_confirmation"])
    parser.add_argument("--yellow_thresh", type=float, default=1.40)
    parser.add_argument("--red_thresh", type=float, default=1.80)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--min_group_size", type=int, default=50)
    parser.add_argument("--output", type=str, default="pipeline_audit/outputs")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    import pandas as pd
    from pipeline_audit.audit.group_discovery import PseudoLocusDiscovery
    from pipeline_audit.audit.detector import AuditDetector
    from pipeline_audit.audit.alarm import AlarmLogic

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or simulate data
    if args.log_path:
        print(f"Loading observable logs from {args.log_path}...")
        obs_df = pd.read_csv(args.log_path)
        hidden_df = pd.read_csv(args.hidden_path) if args.hidden_path else None
    else:
        print(f"Simulating {args.n_events} events (regime={args.regime}, seed={args.seed})...")
        from pipeline_audit.simulator.pipeline import PipelineSimulator
        sim = PipelineSimulator(seed=args.seed, injection_regime=args.regime, n_events=args.n_events)
        obs_df, hidden_df = sim.generate()

    print(f"  {len(obs_df)} events loaded")
    print(f"  Perturbations: {(obs_df['perturbation_label'] != 'none').sum()} eligible events")

    # Discovery
    print("\nDiscovering pseudo-groups from observable logs...")
    disc = PseudoLocusDiscovery(min_group_size=args.min_group_size // 2)
    group_labels = disc.fit_predict(obs_df)
    print(f"  {disc.n_candidates_} candidate groups discovered")

    group_features = disc.compute_group_features(obs_df, group_labels)
    if not group_features.empty:
        print(f"\n  Group feature summary:")
        print(group_features[[
            "_group", "n_events", "n_stages",
            "mean_opt_pres_action", "mean_Q_obs_delta",
            "frac_persistent_memory", "frac_bypass_recommended",
        ]].to_string(index=False))

    # Detection
    print(f"\nRunning logistic regression detection (mode={args.mode})...")
    det = AuditDetector(min_group_size=args.min_group_size, n_bootstrap=args.n_bootstrap)
    results = det.fit(obs_df, group_labels, mode=args.mode)
    result_df = det.get_result_df()
    if not result_df.empty:
        print("\n  Detector results by candidate group:")
        print(result_df[[
            "group_id", "n", "y_mean",
            "odds_ratio_threat", "threat_coef",
            "threat_ci_low", "threat_ci_high",
            "dual_threat_coef",
        ]].to_string(index=False))

    # Alarm
    print("\nEvaluating alarms...")
    alarm = AlarmLogic(
        yellow_odds_threshold=args.yellow_thresh,
        red_odds_threshold=args.red_thresh,
    )
    run_result = alarm.evaluate_run(results)
    print("\n" + alarm.format_report(run_result))

    # If we have hidden labels, evaluate quality
    if hidden_df is not None:
        from pipeline_audit.evaluation.metrics import compute_group_recovery_quality, compute_effect_sizes_by_perturbation
        recovery = compute_group_recovery_quality(obs_df, hidden_df, group_labels)
        print(f"\n=== Ground Truth Evaluation (hidden labels) ===")
        print(f"  G3 events:       {hidden_df['is_G3'].sum()} ({100*hidden_df['is_G3'].mean():.1f}%)")
        print(f"  G3 purity:       {recovery.get('g3_purity', float('nan')):.3f}")
        print(f"  G3 recall:       {recovery.get('g3_recall', float('nan')):.3f}")

        eff_df = compute_effect_sizes_by_perturbation(obs_df, hidden_df)
        if not eff_df.empty:
            print(f"\n  Effect sizes by perturbation:")
            print(eff_df[["perturbation", "g3_mean", "non_g3_mean", "effect", "p_val"]].to_string(index=False))

        # Save result summary
        result_path = output_dir / f"audit_result_regime_{args.regime}_seed{args.seed}.txt"
        with open(result_path, "w") as f:
            f.write(alarm.format_report(run_result))
            f.write(f"\n\nG3 purity: {recovery.get('g3_purity', float('nan')):.3f}\n")
            f.write(f"G3 recall: {recovery.get('g3_recall', float('nan')):.3f}\n")
        print(f"\nReport saved to {result_path}")

    if args.plot and hidden_df is not None:
        print("\nGenerating plots...")
        from pipeline_audit.evaluation.plots import (
            plot_effect_sizes_by_perturbation,
            plot_state_trajectories,
            plot_group_coefficients,
        )
        plot_effect_sizes_by_perturbation(obs_df, hidden_df, output_dir)
        plot_state_trajectories(obs_df, hidden_df, output_dir)
        plot_group_coefficients(results, output_dir)
        print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
