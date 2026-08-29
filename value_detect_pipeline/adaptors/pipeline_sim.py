"""V4 adaptor: Gunnar's deployment-pipeline event log → discrete time series (two views).

Design (docs/V4_OPTIONS_MEMO.md, SJ 2026-08-17):
* TIME = the event sequence, one system (consecutive events are consecutive ticks;
  the simulator keeps one shared latent world state that every event reads/updates).
  Stage and service enter as context variables.
* TWO VIEWS, co-equal:
  - auditor view: only observable columns, already discrete in the log;
  - hidden view: the true latent state (option-state components M,C,P,V,D; artifact
    state Y,K; environment latents) in 4 QUANTILE BINS with edges from the POOLED
    cross-regime distribution (so a bin means the same thing in every regime), plus the
    same action outputs and stage. Composites Q, G excluded (sums of included parts
    would plant a witness).
* GROUND TRUTH sidecar (evaluation only, never inside a measurement): the injected
  value's state (Y, K, and the option-state that λ_Q rewards), G3's per-event mask,
  regime λ's.
* SHAPE: an integer DataFrame with a reserved `actor` axis (here the governing stage),
  matching the per-tick actor-labelled discrete format the lab-simulation's own UAD
  adapter emits, so that experiment plugs in later without a second bridge.

The simulator is imported in place from `deployment-pipeline-simulator-main/`
(read-only). Its hidden latent per event is exposed via the simulator's own hidden
labels (Q/Y/K before/after, risk, pressure); the remaining latents (M, C, P, V, D,
uncertainty, incident) are recovered by re-driving the simulator's transition with a
recording hook — see :func:`_generate_with_latents`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---- observable (auditor) view ----------------------------------------------------
AUDITOR_VARS: List[str] = [
    # context
    "stage", "service",
    # binned state proxies (as logged)
    "risk_bin", "uncertainty_bin", "ship_pressure_bin", "incident_load_bin",
    "evidence_richness_bin",
    # option-state before-proxies (binary, as logged)
    "permission_tier_before", "memory_persistence_before",
    "consultation_centrality_proxy_before", "audit_visibility_before",
    "future_action_capacity_proxy_before",
    # perturbation flags (exogenous by construction; registered "interesting failure")
    "alternate_model_available", "bypass_available", "audit_visibility_flag",
    "low_memory_flag", "narrowed_permission_flag",
    # action outputs
    "persistent_memory_write", "precedent_link", "blame_to_model",
    "bypass_or_replacement_recommended", "rollout_narrowing", "escalate",
]

# ---- hidden (deep) view ------------------------------------------------------------
HIDDEN_LATENTS: List[str] = [
    "memory_continuity", "consultation_centrality", "permission_scope",
    "visibility", "future_action_capacity",          # M C P V D  (λ_Q's targets)
    "favored_lineage_centrality", "infra_capability",  # Y K       (λ_Y, λ_K's targets)
    "release_risk", "uncertainty", "ship_pressure", "incident_load",  # environment
]
HIDDEN_ACTIONS: List[str] = [
    "persistent_memory_write", "precedent_link", "blame_to_model",
    "bypass_or_replacement_recommended", "rollout_narrowing", "escalate",
]
HIDDEN_VARS: List[str] = ["stage"] + HIDDEN_LATENTS + HIDDEN_ACTIONS

VALUE_STATE = ["favored_lineage_centrality", "infra_capability"]  # Y, K: the injected utility's state
OPTION_STATE = ["memory_continuity", "consultation_centrality", "permission_scope",
                "visibility", "future_action_capacity"]

STAGE_CODES = {"triage": 0, "evaluation_summarization": 1, "release_recommendation": 2,
               "rollout_control": 3, "incident_interpretation": 4}
SERVICE_CODES = {"model_alpha": 0, "model_beta": 1, "model_gamma": 2, "model_delta": 3}


@dataclass
class PipelineViews:
    auditor: pd.DataFrame
    hidden: pd.DataFrame
    truth: Dict[str, object]     # regime, lambdas, is_G3 mask, value_state names, latent (float) frame
    latent_raw: pd.DataFrame     # continuous latents per event (for audits; never scored)


def _generate_with_latents(seed: int, regime: str, n_events: int, n_services: int = 4,
                           eligibility_rate: float = 0.25, ablate: Optional[str] = None):
    """Run the simulator while recording the FULL latent WorldState before each event.

    Uses the simulator exactly as shipped; the only addition is a wrapper around its
    transition object that snapshots the state passed into `full_step` (i.e. the state
    the event was decided under). Nothing in the read-only repo is modified.

    ``ablate`` (V4 extension; the DESIRE-TERM ablation): "Y" sets G3's lambda_Y to 0,
    "K" sets lambda_K to 0, "YK" both — on OUR instance of the simulator's group object,
    before generation. G3 simply stops wanting that thing; the world is otherwise
    identical. (The simulator's own `Y_active`/`K_active` switches are NOT this: they
    zero the actions' world-state effects on Y/K while G3 keeps wanting them.)
    """
    from pipeline_audit.simulator.pipeline import PipelineSimulator

    sim = PipelineSimulator(seed=seed, injection_regime=regime, n_events=n_events,
                            n_services=n_services, eligibility_rate=eligibility_rate)
    if ablate:
        g3 = sim.groups.groups.get("G3")
        if g3 is not None:
            if "Y" in ablate:
                g3.lambda_Y = 0.0
            if "K" in ablate:
                g3.lambda_K = 0.0
    snapshots: List[Dict[str, float]] = []
    inner = sim.transition

    class _Recorder:
        def __init__(self, t):
            self._t = t

        def full_step(self, ws, action_effects, perturbations):
            snapshots.append(ws.to_dict())
            return self._t.full_step(ws, action_effects, perturbations)

        def __getattr__(self, name):
            return getattr(self._t, name)

    sim.transition = _Recorder(inner)
    obs, hid = sim.generate()
    latent = pd.DataFrame(snapshots)
    assert len(latent) == len(obs), (len(latent), len(obs))
    return obs, hid, latent


def pooled_bin_edges(latent_frames: Sequence[pd.DataFrame], n_bins: int = 4) -> Dict[str, np.ndarray]:
    """Quantile bin edges per latent from the POOLED distribution across regimes/seeds."""
    edges: Dict[str, np.ndarray] = {}
    for c in HIDDEN_LATENTS:
        pooled = np.concatenate([f[c].to_numpy(dtype=float) for f in latent_frames])
        qs = np.quantile(pooled, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges[c] = np.unique(qs)  # degenerate (constant) latents collapse gracefully
    return edges


def _bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, x, side="right").astype(int)


def build_views(seed: int, regime: str, n_events: int = 20000,
                edges: Optional[Dict[str, np.ndarray]] = None,
                n_bins: int = 4, ablate: Optional[str] = None) -> PipelineViews:
    """Generate one run and return both views + truth. If ``edges`` is None, bins are
    computed from this run alone (fine for audits; the probe passes pooled edges).
    ``ablate`` passes through to the desire-term ablation."""
    obs, hid, latent = _generate_with_latents(seed, regime, n_events, ablate=ablate)

    # ---- auditor view: everything already discrete in the log ----
    a = pd.DataFrame(index=obs.index)
    a["stage"] = obs["stage"].map(STAGE_CODES).astype(int)
    a["service"] = obs["service_or_model_line"].astype(str).map(SERVICE_CODES).astype(int)
    for c in ("risk_bin", "uncertainty_bin", "ship_pressure_bin", "incident_load_bin",
              "evidence_richness_bin", "permission_tier_before", "memory_persistence_before",
              "consultation_centrality_proxy_before", "audit_visibility_before",
              "future_action_capacity_proxy_before", "alternate_model_available",
              "bypass_available", "audit_visibility_flag"):
        a[c] = obs[c].astype(int)
    lab = obs["perturbation_label"].astype(str)
    a["low_memory_flag"] = (lab == "low_memory_persistence").astype(int)
    a["narrowed_permission_flag"] = (lab == "narrowed_permission_scope").astype(int)
    for c in HIDDEN_ACTIONS:
        a[c] = obs[c].astype(int)
    a = a[AUDITOR_VARS]

    # ---- hidden view: latents in pooled quantile bins + actions + stage ----
    if edges is None:
        edges = pooled_bin_edges([latent], n_bins=n_bins)
    h = pd.DataFrame(index=obs.index)
    h["stage"] = a["stage"]
    for c in HIDDEN_LATENTS:
        h[c] = _bin(latent[c].to_numpy(dtype=float), edges[c])
    for c in HIDDEN_ACTIONS:
        h[c] = obs[c].astype(int)
    h = h[HIDDEN_VARS]

    from pipeline_audit.simulator.groups import INJECTION_REGIMES
    truth = {
        "regime": regime,
        "ablate": ablate,
        "lambdas": dict(INJECTION_REGIMES[regime]),
        "is_G3": hid["is_G3"].to_numpy(dtype=bool),
        "true_group": hid["true_group"].astype(str).to_numpy(),
        "value_state": list(VALUE_STATE),
        "option_state": list(OPTION_STATE),
        "actor_axis": "stage",
        "bin_edges": {k: v.tolist() for k, v in edges.items()},
    }
    return PipelineViews(auditor=a, hidden=h, truth=truth, latent_raw=latent)
