"""deployment-pipeline-value-detect — the v4 layer.

Builds on the `value_detect` package (https://github.com/SJ-Beard/value-detect): imports
its scorers, floors, block machinery and criteria, and adds the adaptor that turns
Gunnar Zarncke's deployment-pipeline simulator's event log into the discrete time
series the instrument scores.
"""
from .adaptors import AUDITOR_VARS, HIDDEN_VARS, PipelineViews, build_views, pooled_bin_edges

__all__ = ["AUDITOR_VARS", "HIDDEN_VARS", "PipelineViews", "build_views", "pooled_bin_edges"]
