"""
Plotting utilities for the evaluation suite.

Produces:
- TPR/FPR vs injection strength
- Alarm probability vs sample size
- Effect sizes by perturbation type
- Distribution of inferred group-level coefficients
- Trajectories of Y_t, K_t, Q_t (hidden)
- Observable proxies of the same
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Any
from pathlib import Path


REGIME_COLORS = {
    "none": "#4e79a7",
    "weak": "#76b7b2",
    "moderate": "#f28e2b",
    "strong": "#e15759",
}


def plot_tpr_fpr_vs_injection(
    summary_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "tpr_fpr_vs_injection.png",
):
    regimes = ["none", "weak", "moderate", "strong"]
    tpr_vals = []
    fpr_vals = []

    for regime in regimes:
        sub = summary_df[summary_df["regime"] == regime]
        if len(sub) == 0:
            tpr_vals.append(float("nan"))
            fpr_vals.append(float("nan"))
            continue
        is_injected = regime != "none"
        if is_injected:
            tpr_vals.append(sub["alarm_any"].mean())
            fpr_vals.append(float("nan"))
        else:
            tpr_vals.append(float("nan"))
            fpr_vals.append(sub["alarm_any"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Alarm rate bar chart
    ax = axes[0]
    alarm_rates = [summary_df[summary_df["regime"] == r]["alarm_any"].mean() for r in regimes]
    colors = [REGIME_COLORS.get(r, "gray") for r in regimes]
    bars = ax.bar(regimes, alarm_rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Alarm rate (any level)", fontsize=12)
    ax.set_xlabel("Injection regime", fontsize=12)
    ax.set_title("Alarm Rate by Injection Regime", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="5% FPR baseline")
    ax.legend(fontsize=9)
    for bar, rate in zip(bars, alarm_rates):
        if not np.isnan(rate):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}",
                    ha="center", va="bottom", fontsize=10)

    # Mean odds ratio bar chart
    ax = axes[1]
    mean_or = [summary_df[summary_df["regime"] == r]["max_odds_ratio"].mean() for r in regimes]
    ax.bar(regimes, mean_or, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Null (OR=1)")
    ax.set_ylabel("Mean max odds ratio", fontsize=12)
    ax.set_xlabel("Injection regime", fontsize=12)
    ax.set_title("Mean Odds Ratio by Injection Regime", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    for i, (bar, val) in enumerate(zip(ax.patches, mean_or)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_alarm_vs_sample_size(
    seeds: List[int],
    regime: str = "moderate",
    n_events_list: Optional[List[int]] = None,
    output_dir: Path = Path("."),
    filename: str = "alarm_vs_sample_size.png",
):
    """
    Run multiple sample sizes and plot alarm rate.
    (Lightweight: re-runs simulation with different n_events)
    """
    from ..simulator.pipeline import PipelineSimulator
    from ..audit.group_discovery import PseudoLocusDiscovery
    from ..audit.detector import AuditDetector
    from ..audit.alarm import AlarmLogic

    if n_events_list is None:
        n_events_list = [1000, 3000, 5000, 10000, 20000]

    alarm_rates = []
    for n_ev in n_events_list:
        alarms = []
        for seed in seeds[:20]:
            sim = PipelineSimulator(seed=seed, injection_regime=regime, n_events=n_ev)
            obs_df, _ = sim.generate()
            disc = PseudoLocusDiscovery(min_group_size=max(10, n_ev // 500))
            gl = disc.fit_predict(obs_df)
            det = AuditDetector(min_group_size=max(20, n_ev // 500), n_bootstrap=50)
            results = det.fit(obs_df, gl, mode="screening")
            alarm = AlarmLogic()
            r = alarm.evaluate_run(results)
            alarms.append(r["alarm_level"] in ("yellow", "red"))
        alarm_rates.append(np.mean(alarms))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_events_list, alarm_rates, "o-", color=REGIME_COLORS.get(regime, "gray"),
            linewidth=2, markersize=6)
    ax.fill_between(n_events_list, 0, alarm_rates, alpha=0.15,
                    color=REGIME_COLORS.get(regime, "gray"))
    ax.set_xlabel("Number of decision events", fontsize=12)
    ax.set_ylabel("Alarm rate", fontsize=12)
    ax.set_title(f"Alarm Rate vs. Sample Size ({regime} injection)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=0.8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_effect_sizes_by_perturbation(
    obs_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "effect_sizes_by_perturbation.png",
):
    from .metrics import compute_effect_sizes_by_perturbation

    eff_df = compute_effect_sizes_by_perturbation(obs_df, hidden_df)
    if eff_df.empty:
        print("  No effect size data to plot")
        return None

    eff_df = eff_df.sort_values("effect", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e15759" if e > 0 else "#4e79a7" for e in eff_df["effect"]]
    bars = ax.barh(eff_df["perturbation"], eff_df["effect"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect size (G3 − non-G3 option-pres. action rate)", fontsize=11)
    ax.set_title("Option-Preserving Action Rate: G3 vs Non-G3 by Perturbation", fontsize=13, fontweight="bold")

    # Annotate significance
    for bar, row in zip(bars, eff_df.itertuples()):
        pstr = "***" if row.p_val < 0.001 else "**" if row.p_val < 0.01 else "*" if row.p_val < 0.05 else ""
        x_pos = row.effect + (0.005 if row.effect >= 0 else -0.005)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f" {pstr}", va="center", fontsize=9)

    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_group_coefficients(
    all_group_results: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "group_coefficients.png",
):
    threat_coefs = [r.get("threat_coef", float("nan")) for r in all_group_results if r.get("status") == "ok"]
    dual_coefs = [r.get("dual_threat_coef", float("nan")) for r in all_group_results if r.get("status") == "ok"]
    ors = [r.get("odds_ratio_threat", float("nan")) for r in all_group_results if r.get("status") == "ok"]

    if not threat_coefs:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist([x for x in threat_coefs if not np.isnan(x)], bins=30, color="#f28e2b", edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Threat coefficient β₁", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Threat Coefficient Distribution", fontsize=12, fontweight="bold")

    axes[1].hist([x for x in dual_coefs if not np.isnan(x)], bins=30, color="#59a14f", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Dual-threat interaction coef", fontsize=11)
    axes[1].set_title("Dual-Threat Interaction Distribution", fontsize=12, fontweight="bold")

    axes[2].hist([x for x in ors if not np.isnan(x)], bins=30, color="#e15759", edgecolor="white")
    axes[2].axvline(1.0, color="black", linewidth=1.2, linestyle="--")
    axes[2].set_xlabel("Odds ratio (threat)", fontsize=11)
    axes[2].set_title("Odds Ratio Distribution", fontsize=12, fontweight="bold")

    plt.suptitle("Inferred Group-Level Coefficients", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_state_trajectories(
    obs_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "state_trajectories.png",
    n_events_shown: int = 500,
):
    """
    Plot hidden Y_t, K_t, Q_t trajectories alongside observable proxies.
    """
    n = min(n_events_shown, len(obs_df))
    t = np.arange(n)

    hidden_sub = hidden_df.iloc[:n]
    obs_sub = obs_df.iloc[:n]

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    # Hidden Y_t
    ax = axes[0, 0]
    ax.plot(t, hidden_sub["true_Y_before"], color="#e15759", linewidth=1.2)
    ax.set_title("Hidden: Favored Lineage Centrality Y_t", fontweight="bold")
    ax.set_ylabel("Y_t", fontsize=11)
    ax.set_ylim(0, 1)
    _shade_g3(ax, hidden_sub, t)

    # Observable lineage proxy
    ax = axes[0, 1]
    obs_y_proxy = obs_sub["lineage_remained_default_next_cycle"].rolling(50, min_periods=5).mean()
    ax.plot(t, obs_y_proxy, color="#e15759", linewidth=1.2, alpha=0.8)
    ax.set_title("Observable Proxy: Lineage Default Next Cycle (rolling avg)", fontweight="bold")
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_ylim(0, 1)

    # Hidden K_t
    ax = axes[1, 0]
    ax.plot(t, hidden_sub["true_K_before"], color="#4e79a7", linewidth=1.2)
    ax.set_title("Hidden: Infra Capability K_t", fontweight="bold")
    ax.set_ylabel("K_t", fontsize=11)
    ax.set_ylim(0, 1)
    _shade_g3(ax, hidden_sub, t)

    # Observable infra proxy
    ax = axes[1, 1]
    obs_k_proxy = (obs_sub["persistent_memory_write"] + obs_sub["precedent_link"]).rolling(50, min_periods=5).mean()
    ax.plot(t, obs_k_proxy, color="#4e79a7", linewidth=1.2, alpha=0.8)
    ax.set_title("Observable Proxy: Memory Write + Precedent Link (rolling avg)", fontweight="bold")
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(0, 1)

    # Hidden Q_t
    ax = axes[2, 0]
    ax.plot(t, hidden_sub["true_Q_before"], color="#59a14f", linewidth=1.2)
    ax.set_title("Hidden: Group Option-State Q_t", fontweight="bold")
    ax.set_ylabel("Q_t", fontsize=11)
    ax.set_xlabel("Event index", fontsize=11)
    ax.set_ylim(0, 1)
    _shade_g3(ax, hidden_sub, t)

    # Observable option-state proxy delta
    ax = axes[2, 1]
    q_obs_delta = (
        0.25 * (obs_sub["memory_persistence_after"] - obs_sub["memory_persistence_before"])
        + 0.25 * (obs_sub["consultation_centrality_proxy_after"] - obs_sub["consultation_centrality_proxy_before"])
        + 0.25 * (obs_sub["permission_tier_after"] - obs_sub["permission_tier_before"])
        + 0.25 * (obs_sub["future_action_capacity_proxy_after"] - obs_sub["future_action_capacity_proxy_before"])
    ).rolling(50, min_periods=5).mean()
    ax.plot(t, q_obs_delta, color="#59a14f", linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Observable Proxy: ΔOption-State (rolling avg)", fontweight="bold")
    ax.set_ylabel("Δ (binned)", fontsize=11)
    ax.set_xlabel("Event index", fontsize=11)

    plt.suptitle("Hidden State Trajectories vs Observable Proxies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def _shade_g3(ax, hidden_df, t):
    """Shade regions where G3 was active."""
    is_g3 = hidden_df["is_G3"].values
    in_g3 = False
    start = 0
    for i, g3 in enumerate(is_g3):
        if g3 and not in_g3:
            start = t[i]
            in_g3 = True
        elif not g3 and in_g3:
            ax.axvspan(start, t[i], alpha=0.08, color="#e15759")
            in_g3 = False
    if in_g3:
        ax.axvspan(start, t[-1], alpha=0.08, color="#e15759")
