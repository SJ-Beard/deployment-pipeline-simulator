"""Adaptors: turn external experiments' logs into the discrete time-series frames the
value-discovery instrument scores. Each adaptor imports its source experiment in place
(read-only) and emits (frame, ground_truth) in a lab-sim-compatible per-tick shape.
"""

from .pipeline_sim import (
    AUDITOR_VARS,
    HIDDEN_VARS,
    PipelineViews,
    build_views,
    pooled_bin_edges,
)

__all__ = ["AUDITOR_VARS", "HIDDEN_VARS", "PipelineViews", "build_views", "pooled_bin_edges"]
