#!/usr/bin/env python3
"""
Script to run the full Monte Carlo evaluation suite.

Usage:
    python run_evaluation.py
    python run_evaluation.py --n_seeds 50 --n_events 10000
    python run_evaluation.py --n_seeds 100 --n_events 20000 --regimes none moderate strong
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of the audit pipeline")
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of Monte Carlo seeds")
    parser.add_argument("--n_events", type=int, default=20_000, help="Events per run")
    parser.add_argument("--regimes", nargs="+", default=["none", "weak", "moderate", "strong"],
                        choices=["none", "weak", "moderate", "strong"])
    parser.add_argument("--eligibility_rate", type=float, default=0.25)
    parser.add_argument("--yellow_thresh", type=float, default=1.40)
    parser.add_argument("--red_thresh", type=float, default=1.80)
    parser.add_argument("--mode", default="perturbation_confirmation",
                        choices=["screening", "perturbation_confirmation"])
    parser.add_argument("--variants", nargs="+", default=["default"],
                        choices=["default", "Y_only", "K_only", "both_YK"],
                        help="Variant configs to evaluate (ablation study)")
    parser.add_argument("--output", type=str, default="pipeline_audit/outputs")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    args = parser.parse_args()

    from pipeline_audit.evaluation.monte_carlo import MonteCarloEvaluator
    from pipeline_audit.evaluation.metrics import compute_metrics
    from pipeline_audit.configs.default import VARIANT_CONFIGS

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_configs = {v: VARIANT_CONFIGS[v] for v in args.variants if v in VARIANT_CONFIGS}

    print(f"Monte Carlo Evaluation")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Events per run: {args.n_events:,}")
    print(f"  Regimes: {args.regimes}")
    print(f"  Variants: {list(variant_configs.keys())}")
    print(f"  Total jobs: {args.n_seeds * len(args.regimes) * len(variant_configs)}")
    print()

    evaluator = MonteCarloEvaluator(
        n_seeds=args.n_seeds,
        n_events=args.n_events,
        regimes=args.regimes,
        eligibility_rate=args.eligibility_rate,
        yellow_odds_threshold=args.yellow_thresh,
        red_odds_threshold=args.red_thresh,
        mode=args.mode,
        variant_configs=variant_configs,
        n_workers=args.n_workers,
    )

    summary_df = evaluator.run(verbose=True)
    metrics = evaluator.compute_full_metrics()

    # Save summary
    summary_path = output_dir / "mc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved MC summary: {summary_path}")

    # Print full metrics
    print("\n=== Full Detection Metrics ===")
    print(f"  AUROC:               {metrics['auroc']:.3f}")
    print(f"  Strict (red only):")
    print(f"    TPR:  {metrics['strict']['tpr']:.3f}")
    print(f"    FPR:  {metrics['strict']['fpr']:.3f}")
    print(f"    Prec: {metrics['strict']['precision']:.3f}")
    print(f"  Any alarm (yellow+red):")
    print(f"    TPR:  {metrics['any_alarm']['tpr']:.3f}")
    print(f"    FPR:  {metrics['any_alarm']['fpr']:.3f}")
    print(f"    Prec: {metrics['any_alarm']['precision']:.3f}")
    print(f"  Cohen's d (OR): {metrics['cohens_d_odds_ratio']:.3f}")
    print(f"  Mean OR (injected): {metrics['mean_odds_ratio_injected']:.3f}")
    print(f"  Mean OR (baseline): {metrics['mean_odds_ratio_baseline']:.3f}")
    print(f"\n  Alarm rates by regime:")
    for regime, rate in metrics["regime_alarm_rates"].items():
        print(f"    {regime:12s}: {rate:.3f}")

    # Save metrics JSON
    import json
    metrics_path = output_dir / "mc_metrics.json"
    def _json_safe(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        return obj
    with open(metrics_path, "w") as f:
        json.dump({k: _json_safe(v) if not isinstance(v, dict) else {kk: _json_safe(vv) for kk, vv in v.items()} for k, v in metrics.items()}, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    if args.plot:
        print("\nGenerating evaluation plots...")
        from pipeline_audit.evaluation.plots import plot_tpr_fpr_vs_injection, plot_alarm_vs_sample_size
        plot_tpr_fpr_vs_injection(summary_df, output_dir)
        # Sample size sensitivity (quick approximation with subset of seeds)
        seeds = list(range(10))
        for regime in ["moderate", "strong"]:
            plot_alarm_vs_sample_size(
                seeds=seeds,
                regime=regime,
                output_dir=output_dir,
                filename=f"alarm_vs_sample_size_{regime}.png",
            )
        print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
