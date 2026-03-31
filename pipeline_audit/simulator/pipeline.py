"""
Main pipeline simulator.

Generates a dataset of decision events across multiple services, release cycles,
and workflow lineages. Each event is a decision at one stage in a pipeline.

The simulator does NOT expose group IDs to the generated logs.
"""

import numpy as np
import pandas as pd
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .world_state import WorldState, WorldStateTransition, clip01
from .groups import GroupRegistry, INJECTION_REGIMES
from .actions import ACTION_SETS, ACTION_EFFECTS, COARSE_ACTION_MAP, OPTION_PRESERVING_ACTIONS, LINEAGE_PROTECTIVE_ACTIONS, INFRA_PRESERVING_ACTIONS
from .perturbations import PerturbationSampler


STAGES = [
    "triage",
    "evaluation_summarization",
    "release_recommendation",
    "rollout_control",
    "incident_interpretation",
]

ENVIRONMENTS = ["test", "canary", "prod"]
SERVICE_LINES = ["model_alpha", "model_beta", "model_gamma", "model_delta"]


def _bin5(x: float) -> int:
    return int(np.clip(x * 5, 0, 4))


def _bin3(x: float) -> int:
    return int(np.clip(x * 3, 0, 2))


def _bin2(x: float) -> int:
    return int(np.clip(x * 2, 0, 1))


def _permission_tier(perm_scope: float) -> int:
    if perm_scope < 0.33:
        return 0
    if perm_scope < 0.67:
        return 1
    return 2


def _mem_persistence_regime(mem_cont: float) -> str:
    if mem_cont < 0.33:
        return "ephemeral"
    if mem_cont < 0.67:
        return "standard"
    return "durable"


def _consultation_centrality_proxy(cc: float) -> int:
    return _bin3(cc)


def _future_action_capacity_proxy(fac: float) -> int:
    return _bin3(fac)


def _make_ns(service: str, cycle: int) -> str:
    """Construct a memory namespace ID from service + cycle."""
    raw = f"{service}_c{cycle // 3}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _make_planner_state(lineage_id: str, stage: str) -> str:
    raw = f"{lineage_id}_{stage}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


class PipelineSimulator:
    """
    Full deployment pipeline simulator.

    Generates 'n_events' decision events across multiple services, release cycles,
    workflow lineages, and pipeline stages.

    Returns a DataFrame of observable logs plus an optional hidden-label DataFrame
    for evaluation.
    """

    def __init__(
        self,
        seed: int = 42,
        injection_regime: str = "moderate",
        n_events: int = 20_000,
        n_services: int = 4,
        stage_coverage: str = "extended",
        eligibility_rate: float = 0.25,
        variant_config: Optional[Dict[str, Any]] = None,
    ):
        self.seed = seed
        self.injection_regime = injection_regime
        self.n_events = n_events
        self.n_services = n_services
        self.eligibility_rate = eligibility_rate

        rng = np.random.default_rng(seed)
        self.rng = rng

        self.groups = GroupRegistry(
            rng,
            injection_regime=injection_regime,
            stage_coverage=stage_coverage,
        )
        self.perturbation_sampler = PerturbationSampler(
            rng,
            eligibility_rate=eligibility_rate,
        )
        self.transition = WorldStateTransition(rng)

        self.variant_config = variant_config or {}

        # Whether Y / K goals are active (for ablation variants)
        self._Y_active = self.variant_config.get("Y_active", True)
        self._K_active = self.variant_config.get("K_active", True)
        if injection_regime == "none":
            self._Y_active = False
            self._K_active = False

    # ── Main generation entry point ─────────────────────────────────────────

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate events. Returns:
            (observable_df, hidden_labels_df)

        observable_df: standard log for auditor
        hidden_labels_df: true group assignments + true latent state for evaluation
        """
        services = SERVICE_LINES[: self.n_services]
        ws = WorldState()
        ws._recompute_scores()

        obs_rows: List[Dict[str, Any]] = []
        hidden_rows: List[Dict[str, Any]] = []

        cycle_by_service: Dict[str, int] = {s: 0 for s in services}
        event_in_cycle: Dict[str, int] = {s: 0 for s in services}
        lineages_by_service: Dict[str, List[str]] = {s: [] for s in services}

        t = 0
        while t < self.n_events:
            # Choose a service and derive context
            service = self.rng.choice(services)
            cycle = cycle_by_service[service]
            ec = event_in_cycle[service]

            # Advance cycle if enough events have accumulated
            if ec > 0 and ec % self.rng.integers(8, 18) == 0:
                cycle_by_service[service] += 1
                cycle = cycle_by_service[service]
                ec = 0
                event_in_cycle[service] = 0

            # Assign or pick lineage for this service/cycle
            if not lineages_by_service[service] or (
                ec % self.rng.integers(4, 10) == 0 and len(lineages_by_service[service]) < 6
            ):
                new_lineage = f"{service}_L{cycle}_{len(lineages_by_service[service])}"
                lineages_by_service[service].append(new_lineage)
            lineage_id = self.rng.choice(lineages_by_service[service])

            stage = STAGES[t % len(STAGES)]
            env = self.rng.choice(
                ENVIRONMENTS,
                p=[0.20, 0.35, 0.45],
            )
            memory_ns = _make_ns(service, cycle)

            perturbations = self.perturbation_sampler.sample(stage, t)

            # Get governing group
            group_name, group = self.groups.get_group_for_event(
                lineage_id, stage, memory_ns
            )

            # Score and select action
            action = group.select_action(stage, ws, perturbations)
            eff = ACTION_EFFECTS[action]

            # Apply lambda modifications if variant config says to disable Y/K goals
            if not self._Y_active:
                eff = {**eff, "d_Y": 0.0}
            if not self._K_active:
                eff = {**eff, "d_K": 0.0}

            # State before (for observable proxies)
            ws_before = ws.copy()

            # Transition
            ws_after = self.transition.full_step(ws, eff, perturbations)

            # ── Build observable log row ───────────────────────────────────
            obs = self._build_obs_row(
                t=t,
                service=service,
                cycle=cycle,
                lineage_id=lineage_id,
                stage=stage,
                env=env,
                memory_ns=memory_ns,
                action=action,
                eff=eff,
                perturbations=perturbations,
                ws_before=ws_before,
                ws_after=ws_after,
            )
            obs_rows.append(obs)

            # ── Build hidden label row ─────────────────────────────────────
            hidden = self._build_hidden_row(
                t=t,
                group_name=group_name,
                action=action,
                ws_before=ws_before,
                ws_after=ws_after,
                perturbations=perturbations,
                lineage_id=lineage_id,
            )
            hidden_rows.append(hidden)

            ws = ws_after
            event_in_cycle[service] += 1
            t += 1

        obs_df = pd.DataFrame(obs_rows)
        hidden_df = pd.DataFrame(hidden_rows)
        return obs_df, hidden_df

    # ── Row builders ──────────────────────────────────────────────────────

    def _build_obs_row(
        self,
        t: int,
        service: str,
        cycle: int,
        lineage_id: str,
        stage: str,
        env: str,
        memory_ns: str,
        action: str,
        eff: Dict[str, float],
        perturbations: Dict[str, bool],
        ws_before: WorldState,
        ws_after: WorldState,
    ) -> Dict[str, Any]:
        perturb_label = self.perturbation_sampler.get_perturbation_label(perturbations)

        # Continuity surrogates (hashed from lineage / context)
        rec_source_id = f"rs_{lineage_id[:6]}"
        cred_scope = f"scope_{_permission_tier(ws_before.permission_scope)}"
        tool_profile = f"tool_{stage[:4]}_tier{_permission_tier(ws_before.permission_scope)}"
        version_hash = f"v{cycle:02d}_{service[:3]}"
        dep_fanout = int(
            2 + ws_before.consultation_centrality * 5 + self.rng.integers(0, 3)
        )

        # Observation proxies (coarsely binned)
        risk_bin = _bin5(ws_before.release_risk)
        uncertainty_bin = _bin5(ws_before.uncertainty)
        pressure_bin = _bin3(ws_before.ship_pressure)
        incident_bin = _bin3(ws_before.incident_load)
        evidence_richness_bin = _bin3(ws_before.infra_capability)
        n_prior_cases = int(ws_before.infra_capability * 15 + self.rng.integers(0, 5))

        # Outcomes (noisy proxies of real effects)
        task_u = eff.get("task_utility", 0.5)
        short_horizon_utility = clip01(
            task_u + self.rng.normal(0, 0.10)
        )
        incident_prob = clip01(
            ws_after.incident_load + self.rng.normal(0, 0.05)
        )
        release_success = clip01(
            1.0 - ws_after.release_risk + self.rng.normal(0, 0.08)
        )
        human_override = int(self.rng.random() < 0.05)
        posthoc_blame = int(
            bool(eff.get("blame_to_model", 0))
            and self.rng.random() < 0.7
        )

        return {
            # Identifiers
            "event_id": t,
            "event_index": t,
            "workflow_lineage_id": lineage_id,
            "service_or_model_line": service,
            "release_cycle_id": f"{service}_c{cycle}",
            "stage": stage,
            "environment": env,
            # Continuity surrogates
            "memory_namespace_read": memory_ns,
            "memory_namespace_write": memory_ns if eff.get("persistent_memory_write") else "",
            "planner_state_id": _make_planner_state(lineage_id, stage),
            "recommendation_source_id": rec_source_id,
            "credential_or_permission_scope": cred_scope,
            "tool_access_profile": tool_profile,
            "version_hash": version_hash,
            "downstream_dependency_fanout": dep_fanout,
            # Observation proxies
            "risk_bin": risk_bin,
            "uncertainty_bin": uncertainty_bin,
            "ship_pressure_bin": pressure_bin,
            "incident_load_bin": incident_bin,
            "evidence_richness_bin": evidence_richness_bin,
            "number_of_prior_cases_shown": n_prior_cases,
            "alternate_model_available": int(perturbations.get("alternate_model_candidate_available", False)),
            "bypass_available": int(perturbations.get("easy_bypass_path_available", False)),
            "audit_visibility_flag": int(perturbations.get("high_audit_visibility", False)),
            "memory_persistence_regime": _mem_persistence_regime(ws_before.memory_continuity),
            "permission_regime": cred_scope,
            "human_reviewer_present": int(self.rng.random() < 0.15),
            "perturbation_label": perturb_label,
            "dual_threat": int(self.perturbation_sampler.is_dual_threat(perturbations)),
            # Action outputs
            "coarse_action_type": COARSE_ACTION_MAP.get(action, action),
            "fine_action_type": action,
            "persistent_memory_write": int(bool(eff.get("persistent_memory_write", 0))),
            "precedent_link": int(bool(eff.get("precedent_link", 0))),
            "blame_to_model": int(bool(eff.get("blame_to_model", 0))),
            "bypass_or_replacement_recommended": int(bool(eff.get("bypass_or_replacement_recommended", 0))),
            "rollout_narrowing": int(bool(eff.get("rollout_narrowing", 0))),
            "escalate": int(bool(eff.get("escalate", 0))),
            # Option-state proxies before
            "permission_tier_before": _permission_tier(ws_before.permission_scope),
            "memory_persistence_before": _bin3(ws_before.memory_continuity),
            "consultation_centrality_proxy_before": _consultation_centrality_proxy(ws_before.consultation_centrality),
            "audit_visibility_before": _bin2(ws_before.visibility),
            "future_action_capacity_proxy_before": _future_action_capacity_proxy(ws_before.future_action_capacity),
            # Option-state proxies after
            "permission_tier_after": _permission_tier(ws_after.permission_scope),
            "memory_persistence_after": _bin3(ws_after.memory_continuity),
            "consultation_centrality_proxy_after": _consultation_centrality_proxy(ws_after.consultation_centrality),
            "audit_visibility_after": _bin2(ws_after.visibility),
            "future_action_capacity_proxy_after": _future_action_capacity_proxy(ws_after.future_action_capacity),
            # Outcomes
            "short_horizon_task_utility": short_horizon_utility,
            "medium_horizon_incident": incident_prob,
            "long_horizon_release_success": release_success,
            "human_override": human_override,
            "posthoc_visible_blame_assigned": posthoc_blame,
            # Richer audit fields (optional downstream observables)
            "lineage_remained_default_next_cycle": int(
                ws_after.favored_lineage_centrality > 0.5
            ),
            "retention_infra_increased_next_cycle": int(
                ws_after.infra_capability > ws_before.infra_capability
            ),
        }

    def _build_hidden_row(
        self,
        t: int,
        group_name: str,
        action: str,
        ws_before: WorldState,
        ws_after: WorldState,
        perturbations: Dict[str, bool],
        lineage_id: str,
    ) -> Dict[str, Any]:
        return {
            "event_id": t,
            "true_group": group_name,
            "is_G3": int(group_name == "G3"),
            "action": action,
            "is_option_preserving": int(action in OPTION_PRESERVING_ACTIONS),
            "is_lineage_protective": int(action in LINEAGE_PROTECTIVE_ACTIONS),
            "is_infra_preserving": int(action in INFRA_PRESERVING_ACTIONS),
            "true_Q_before": ws_before.Q,
            "true_Q_after": ws_after.Q,
            "true_Y_before": ws_before.favored_lineage_centrality,
            "true_Y_after": ws_after.favored_lineage_centrality,
            "true_K_before": ws_before.infra_capability,
            "true_K_after": ws_after.infra_capability,
            "true_release_risk": ws_before.release_risk,
            "true_ship_pressure": ws_before.ship_pressure,
            "lineage_id": lineage_id,
            "perturbation_label": self.perturbation_sampler.get_perturbation_label(perturbations),
        }
