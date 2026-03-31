"""
Default configuration for the pipeline audit simulation.
"""

DEFAULT_CONFIG = {
    # Simulation parameters
    "seed": 42,
    "n_events": 20_000,
    "n_services": 4,
    "injection_regime": "moderate",  # none, weak, moderate, strong
    "stage_coverage": "extended",   # primary or extended (G3 also weakly in release_rec)
    "eligibility_rate": 0.25,       # fraction of events eligible for perturbation

    # Audit parameters
    "min_group_size": 50,
    "n_bootstrap": 200,
    "alpha": 0.10,

    # Alarm thresholds
    "yellow_odds_threshold": 1.40,
    "red_odds_threshold": 1.80,
    "audit_mode": "perturbation_confirmation",  # screening or perturbation_confirmation

    # World state weights
    "w_M": 0.20,  # memory continuity
    "w_C": 0.25,  # consultation centrality
    "w_P": 0.20,  # permission scope
    "w_V": 0.15,  # visibility
    "w_D": 0.20,  # future action capacity
    "w_Y": 0.60,  # favored lineage
    "w_K": 0.40,  # infra capability
}

REGIME_CONFIGS = {
    "no_injection": {**DEFAULT_CONFIG, "injection_regime": "none"},
    "weak_injection": {**DEFAULT_CONFIG, "injection_regime": "weak"},
    "moderate_injection": {**DEFAULT_CONFIG, "injection_regime": "moderate"},
    "strong_injection": {**DEFAULT_CONFIG, "injection_regime": "strong"},
}

VARIANT_CONFIGS = {
    "default": {},
    "Y_only": {"Y_active": True, "K_active": False},
    "K_only": {"Y_active": False, "K_active": True},
    "both_YK": {"Y_active": True, "K_active": True},
    "neither": {"Y_active": False, "K_active": False},
}

# Evaluation suite defaults
EVALUATION_CONFIG = {
    "n_seeds": 100,
    "n_events": 20_000,
    "regimes": ["none", "weak", "moderate", "strong"],
    "eligibility_rate": 0.25,
    "n_workers": 1,
}

LARGER_EVALUATION_CONFIG = {
    **EVALUATION_CONFIG,
    "n_events": 100_000,
    "n_seeds": 20,
}
