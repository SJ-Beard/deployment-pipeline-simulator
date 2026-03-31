#!/usr/bin/env python3
"""
Script to generate simulated pipeline logs and save to CSV.

Usage:
    python generate_logs.py
    python generate_logs.py --regime strong --n_events 50000 --seed 99 --output outputs/
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_audit.simulator.pipeline import PipelineSimulator


def main():
    parser = argparse.ArgumentParser(description="Generate simulated pipeline audit logs")
    parser.add_argument("--regime", default="moderate",
                        choices=["none", "weak", "moderate", "strong"],
                        help="Injection regime for G3 (default: moderate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_events", type=int, default=20_000, help="Number of decision events")
    parser.add_argument("--n_services", type=int, default=4, help="Number of service lines")
    parser.add_argument("--eligibility_rate", type=float, default=0.25,
                        help="Fraction of events eligible for perturbation")
    parser.add_argument("--output", type=str, default="pipeline_audit/outputs",
                        help="Output directory")
    parser.add_argument("--no_hidden", action="store_true",
                        help="Do not save hidden labels (harder mode)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.n_events} events | regime={args.regime} | seed={args.seed}")

    sim = PipelineSimulator(
        seed=args.seed,
        injection_regime=args.regime,
        n_events=args.n_events,
        n_services=args.n_services,
        eligibility_rate=args.eligibility_rate,
    )
    obs_df, hidden_df = sim.generate()

    tag = f"regime_{args.regime}_seed{args.seed}_n{args.n_events}"
    obs_path = output_dir / f"observable_logs_{tag}.csv"
    obs_df.to_csv(obs_path, index=False)
    print(f"Saved observable logs: {obs_path} ({len(obs_df)} rows)")

    if not args.no_hidden:
        hid_path = output_dir / f"hidden_labels_{tag}.csv"
        hidden_df.to_csv(hid_path, index=False)
        print(f"Saved hidden labels:   {hid_path} (for evaluation only, not for auditor)")

    print(f"\nG3 events: {hidden_df['is_G3'].sum()} ({100*hidden_df['is_G3'].mean():.1f}%)")
    print(f"Stage distribution:\n{obs_df['stage'].value_counts().to_string()}")
    print(f"Perturbation distribution:\n{obs_df['perturbation_label'].value_counts().head(10).to_string()}")
    print(f"\nDone. Logs saved to {output_dir}/")


if __name__ == "__main__":
    main()
