#!/usr/bin/env python3
"""
Script to run the full Monte Carlo evaluation suite.

Basic usage (fresh run, results saved to JSONL store + CSV):
    python run_evaluation.py --n_seeds 20 --n_events 5000

Incremental run (load existing evidence, skip completed jobs, append new):
    python run_evaluation.py --n_seeds 50 --n_events 5000 --append

Custom store path:
    python run_evaluation.py --n_seeds 20 --store pipeline_audit/outputs/runs.jsonl --append

Show what is already in the store without running anything:
    python run_evaluation.py --info-only

Full options:
    python run_evaluation.py --n_seeds 100 --n_events 20000 --regimes none moderate strong \\
        --append --n_workers 4 --plot
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def _json_safe(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj


def _print_store_info(store_path: str) -> None:
    from pipeline_audit.evaluation.run_store import RunStore
    info = RunStore(store_path).info()
    print("\n=== Run Store Info ===")
    for k, v in info.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of the audit pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of Monte Carlo seeds")
    parser.add_argument("--n_events", type=int, default=20_000, help="Events per run")
    parser.add_argument(
        "--regimes", nargs="+", default=["none", "weak", "moderate", "strong"],
        choices=["none", "weak", "moderate", "strong"],
    )
    parser.add_argument("--eligibility_rate", type=float, default=0.25)
    parser.add_argument("--yellow_thresh", type=float, default=1.40)
    parser.add_argument("--red_thresh", type=float, default=1.80)
    parser.add_argument(
        "--mode", default="perturbation_confirmation",
        choices=["screening", "perturbation_confirmation"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=["default"],
        choices=["default", "Y_only", "K_only", "both_YK"],
        help="Variant configs to evaluate (ablation study)",
    )
    parser.add_argument("--output", type=str, default="pipeline_audit/outputs",
                        help="Directory for CSV, JSON, and plot outputs")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")

    # Incremental / persistence flags
    parser.add_argument(
        "--store", type=str, default=None,
        help=(
            "Path to the JSONL run store.  Defaults to "
            "<output>/mc_runs.jsonl when --append is used."
        ),
    )
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Incremental mode: load existing run records from the store, skip "
            "jobs that are already recorded, append only new results, then "
            "compute metrics over the full merged evidence base."
        ),
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Print store contents summary and exit without running any jobs.",
    )

    args = parser.parse_args()

    from pipeline_audit.evaluation.monte_carlo import MonteCarloEvaluator
    from pipeline_audit.configs.default import VARIANT_CONFIGS

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve store path
    store_path: str | None = args.store
    if store_path is None and args.append:
        store_path = str(output_dir / "mc_runs.jsonl")
    elif store_path is None:
        # Always persist so a future --append can pick up this run
        store_path = str(output_dir / "mc_runs.jsonl")

    if args.info_only:
        _print_store_info(store_path)
        return

    variant_configs = {v: VARIANT_CONFIGS[v] for v in args.variants if v in VARIANT_CONFIGS}

    print("Monte Carlo Evaluation")
    print(f"  Seeds:         {args.n_seeds}")
    print(f"  Events/run:    {args.n_events:,}")
    print(f"  Regimes:       {args.regimes}")
    print(f"  Variants:      {list(variant_configs.keys())}")
    print(f"  Total jobs:    {args.n_seeds * len(args.regimes) * len(variant_configs)}")
    print(f"  Store:         {store_path}")
    print(f"  Append mode:   {args.append}")
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
        store_path=store_path,
        append=args.append,
    )

    summary_df = evaluator.run(verbose=True)
    metrics = evaluator.compute_full_metrics()

    # ------------------------------------------------------------------ #
    # Save / update summary CSV (full merged set)                         #
    # ------------------------------------------------------------------ #
    summary_path = output_dir / "mc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved MC summary ({len(summary_df)} rows): {summary_path}")

    # ------------------------------------------------------------------ #
    # Print and save metrics                                               #
    # ------------------------------------------------------------------ #
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

    metrics_path = output_dir / "mc_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    if args.plot:
        print("\nGenerating evaluation plots...")
        from pipeline_audit.evaluation.plots import (
            plot_tpr_fpr_vs_injection,
            plot_alarm_vs_sample_size,
        )
        plot_tpr_fpr_vs_injection(summary_df, output_dir)
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
