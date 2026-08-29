"""Adaptor tests (mirror those in value-detect's suite, retargeted to this package)."""
import numpy as np


def test_pipeline_adaptor_views_and_truth():
    from value_detect_pipeline.adaptors import build_views, AUDITOR_VARS, HIDDEN_VARS
    from value_detect_pipeline.adaptors.pipeline_sim import _generate_with_latents
    v = build_views(seed=1, regime="strong", n_events=2000)
    assert list(v.auditor.columns) == AUDITOR_VARS
    assert list(v.hidden.columns) == HIDDEN_VARS
    assert len(v.auditor) == len(v.hidden) == 2000
    obs, hid, lat = _generate_with_latents(1, "strong", 2000)
    assert np.allclose(lat["favored_lineage_centrality"].to_numpy(), hid["true_Y_before"].to_numpy(dtype=float))
    assert np.allclose(lat["infra_capability"].to_numpy(), hid["true_K_before"].to_numpy(dtype=float))
    assert v.truth["is_G3"].mean() > 0.2
    v0 = build_views(seed=1, regime="none", n_events=1500)
    assert v0.truth["is_G3"].mean() == 0.0


def test_desire_term_ablation_changes_g3_only():
    from value_detect_pipeline.adaptors.pipeline_sim import _generate_with_latents
    base = _generate_with_latents(0, "strong", 4000)
    noY = _generate_with_latents(0, "strong", 4000, ablate="Y")
    g3b = base[1]["is_G3"].to_numpy(dtype=bool)
    g3y = noY[1]["is_G3"].to_numpy(dtype=bool)
    # Without the Y-desire G3 blames the incumbent more (it stops protecting it).
    assert noY[0].loc[g3y, "blame_to_model"].mean() >= base[0].loc[g3b, "blame_to_model"].mean() - 0.005
