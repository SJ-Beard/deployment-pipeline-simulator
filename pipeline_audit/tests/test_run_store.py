"""
Tests for RunStore (JSONL persistence) and incremental Monte Carlo evaluation.

Covers:
  - Append / overwrite / load round-trips
  - JSON-safety (NaN, Inf, numpy scalars)
  - Deduplication via completed_keys / filter_new_jobs
  - MonteCarloEvaluator append=True skips already-completed jobs and merges
  - MonteCarloEvaluator append=False overwrites the store
  - store_info() returns sensible metadata
  - Multi-session incremental evidence accumulation
"""

import math
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pytest
import numpy as np
import pandas as pd

from pipeline_audit.evaluation.run_store import RunStore, _coerce_json
from pipeline_audit.evaluation.monte_carlo import MonteCarloEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(seed: int = 0, regime: str = "none", variant: str = "default",
                 **extra) -> Dict[str, Any]:
    return {
        "seed": seed,
        "regime": regime,
        "variant": variant,
        "n_events": 500,
        "alarm_level": "none",
        "alarm_any": 0,
        "alarm_red": 0,
        "max_odds_ratio": 1.05,
        "dual_signal": 0,
        "n_flagged": 0,
        "g3_purity": None,
        "g3_recall": None,
        "n_g3_events": 0,
        "n_total_events": 500,
        **extra,
    }


def _store(path) -> RunStore:
    return RunStore(path)


# ---------------------------------------------------------------------------
# _coerce_json
# ---------------------------------------------------------------------------

class TestCoerceJson:
    def test_nan_becomes_none(self):
        assert _coerce_json(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _coerce_json(float("inf")) is None
        assert _coerce_json(float("-inf")) is None

    def test_numpy_int_scalar(self):
        val = _coerce_json(np.int64(42))
        assert val == 42
        assert isinstance(val, int)

    def test_numpy_float_scalar(self):
        val = _coerce_json(np.float32(3.14))
        assert isinstance(val, float)
        assert abs(val - 3.14) < 0.01

    def test_numpy_array(self):
        val = _coerce_json(np.array([1, 2, 3]))
        assert val == [1, 2, 3]

    def test_nested_dict(self):
        inp = {"a": float("nan"), "b": {"c": np.int64(7)}}
        out = _coerce_json(inp)
        assert out["a"] is None
        assert out["b"]["c"] == 7

    def test_list_of_mixed(self):
        out = _coerce_json([1, float("nan"), np.float64(2.5)])
        assert out[0] == 1
        assert out[1] is None
        assert abs(out[2] - 2.5) < 1e-5

    def test_passthrough_plain_types(self):
        assert _coerce_json(42) == 42
        assert _coerce_json("hello") == "hello"
        assert _coerce_json(True) is True
        assert _coerce_json(None) is None


# ---------------------------------------------------------------------------
# RunStore — basic persistence
# ---------------------------------------------------------------------------

class TestRunStoreBasics:
    def test_load_from_nonexistent_file_returns_empty(self, tmp_path):
        store = _store(tmp_path / "nonexistent.jsonl")
        assert store.load() == []

    def test_load_summary_df_from_nonexistent_returns_empty_df(self, tmp_path):
        store = _store(tmp_path / "nonexistent.jsonl")
        df = store.load_summary_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_append_creates_file(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0)])
        assert store.path.exists()

    def test_append_single_record(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=42, regime="strong")])
        records = store.load()
        assert len(records) == 1
        assert records[0]["seed"] == 42
        assert records[0]["regime"] == "strong"

    def test_append_multiple_records_sequential(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=i) for i in range(5)])
        assert len(store.load()) == 5

    def test_append_accumulates_across_calls(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0)])
        store.append([_make_record(seed=1)])
        store.append([_make_record(seed=2)])
        assert len(store.load()) == 3

    def test_overwrite_replaces_content(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=i) for i in range(10)])
        store.overwrite([_make_record(seed=99)])
        records = store.load()
        assert len(records) == 1
        assert records[0]["seed"] == 99

    def test_append_empty_list_is_noop(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([])          # should not create the file or raise
        assert not store.path.exists()

    def test_records_are_valid_json_lines(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=7, regime="moderate")])
        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["seed"] == 7

    def test_run_id_added_automatically(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0)])
        rec = store.load()[0]
        assert "run_id" in rec
        assert len(rec["run_id"]) == 36   # UUID4 string length

    def test_run_timestamp_added_automatically(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record()])
        rec = store.load()[0]
        assert "run_timestamp" in rec
        assert "T" in rec["run_timestamp"]   # ISO format

    def test_existing_run_id_not_overwritten(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        rec = _make_record()
        rec["run_id"] = "fixed-uuid"
        store.append([rec])
        assert store.load()[0]["run_id"] == "fixed-uuid"

    def test_nan_values_serialised_as_null(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        rec = _make_record()
        rec["g3_purity"] = float("nan")
        store.append([rec])
        obj = json.loads(store.path.read_text().strip())
        assert obj["g3_purity"] is None

    def test_load_summary_df_columns(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0, regime="weak")])
        df = store.load_summary_df()
        for col in ("seed", "regime", "variant", "alarm_level"):
            assert col in df.columns

    def test_malformed_lines_skipped_gracefully(self, tmp_path, caplog):
        store = _store(tmp_path / "runs.jsonl")
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text('{"seed": 1}\nNOT_JSON\n{"seed": 2}\n')
        import logging
        with caplog.at_level(logging.WARNING):
            records = store.load()
        assert len(records) == 2
        assert any("malformed" in m.lower() or "skipping" in m.lower() for m in caplog.messages)


# ---------------------------------------------------------------------------
# RunStore — deduplication
# ---------------------------------------------------------------------------

class TestRunStoreDeduplication:
    def test_completed_keys_empty_store(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        assert store.completed_keys() == set()

    def test_completed_keys_returns_correct_tuples(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([
            _make_record(seed=0, regime="none", variant="default"),
            _make_record(seed=1, regime="strong", variant="Y_only"),
        ])
        keys = store.completed_keys()
        assert (0, "none", "default") in keys
        assert (1, "strong", "Y_only") in keys
        assert len(keys) == 2

    def test_filter_new_jobs_all_new(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        jobs = [
            (0, "none", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
            (1, "strong", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
        ]
        new_jobs, n_skipped = store.filter_new_jobs(jobs)
        assert len(new_jobs) == 2
        assert n_skipped == 0

    def test_filter_new_jobs_some_already_done(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0, regime="none", variant="default")])
        jobs = [
            (0, "none", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
            (1, "strong", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
        ]
        new_jobs, n_skipped = store.filter_new_jobs(jobs)
        assert n_skipped == 1
        assert len(new_jobs) == 1
        assert new_jobs[0][0] == 1   # seed=1 is the remaining job

    def test_filter_new_jobs_all_done(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0, regime="none", variant="default")])
        jobs = [
            (0, "none", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
        ]
        new_jobs, n_skipped = store.filter_new_jobs(jobs)
        assert n_skipped == 1
        assert len(new_jobs) == 0

    def test_dedup_respects_variant(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([_make_record(seed=0, regime="none", variant="default")])
        jobs = [
            (0, "none", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "default"),
            (0, "none", 500, 0.25, {}, "extended", 1.4, 1.8, "screening", "Y_only"),
        ]
        new_jobs, n_skipped = store.filter_new_jobs(jobs)
        assert n_skipped == 1
        assert len(new_jobs) == 1
        assert new_jobs[0][9] == "Y_only"


# ---------------------------------------------------------------------------
# RunStore — info()
# ---------------------------------------------------------------------------

class TestRunStoreInfo:
    def test_info_nonexistent_store(self, tmp_path):
        info = _store(tmp_path / "no.jsonl").info()
        assert info["n_records"] == 0
        assert info["exists"] is False

    def test_info_populated_store(self, tmp_path):
        store = _store(tmp_path / "runs.jsonl")
        store.append([
            _make_record(seed=0, regime="none"),
            _make_record(seed=1, regime="strong"),
        ])
        info = store.info()
        assert info["n_records"] == 2
        assert "none" in info["regimes"]
        assert "strong" in info["regimes"]
        assert info["seeds_range"] == [0, 1]


# ---------------------------------------------------------------------------
# MonteCarloEvaluator — store integration (tiny fast runs)
# ---------------------------------------------------------------------------

TINY_KWARGS = dict(
    n_seeds=2,
    n_events=200,
    regimes=["none", "strong"],
    variant_configs={"default": {}},
    n_workers=1,
)


class TestMonteCarloEvaluatorStore:
    def test_overwrite_mode_saves_store(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev.run(verbose=False)
        store = RunStore(store_path)
        records = store.load()
        # 2 seeds × 2 regimes = 4 records
        assert len(records) == 4

    def test_overwrite_mode_replaces_store(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        # First run
        ev1 = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev1.run(verbose=False)
        # Second run — same config, append=False → should overwrite
        ev2 = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev2.run(verbose=False)
        records = RunStore(store_path).load()
        assert len(records) == 4     # only the 2nd run's 4 records

    def test_append_mode_loads_existing_and_merges(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        # Session 1: seeds 0-1
        ev1 = MonteCarloEvaluator(
            n_seeds=2, n_events=200, regimes=["none", "strong"],
            variant_configs={"default": {}}, n_workers=1,
            store_path=store_path, append=False,
        )
        df1 = ev1.run(verbose=False)
        assert len(df1) == 4

        # Session 2: seeds 0-3 (append mode — seeds 0&1 already done)
        ev2 = MonteCarloEvaluator(
            n_seeds=4, n_events=200, regimes=["none", "strong"],
            variant_configs={"default": {}}, n_workers=1,
            store_path=store_path, append=True,
        )
        df2 = ev2.run(verbose=False)
        # Merged DataFrame should contain all 8 records (4 old + 4 new)
        assert len(df2) == 8
        assert RunStore(store_path).load().__len__() == 8

    def test_append_skips_completed_jobs(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        # Run seeds 0-1 first
        ev1 = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev1.run(verbose=False)

        # Re-run same seeds with append=True — no new jobs should execute
        ev2 = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=True)
        df2 = ev2.run(verbose=False)
        # Still only 4 records (all were skipped)
        assert len(df2) == 4
        assert RunStore(store_path).load().__len__() == 4

    def test_summary_df_has_run_id_and_timestamp_columns(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        df = ev.run(verbose=False)
        assert "run_id" in df.columns
        assert "run_timestamp" in df.columns
        assert df["run_id"].notna().all()

    def test_summary_df_has_variant_column(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        df = ev.run(verbose=False)
        assert "variant" in df.columns
        assert (df["variant"] == "default").all()

    def test_metrics_computable_after_append(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        ev1 = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev1.run(verbose=False)

        ev2 = MonteCarloEvaluator(
            n_seeds=4, n_events=200, regimes=["none", "strong"],
            variant_configs={"default": {}}, n_workers=1,
            store_path=store_path, append=True,
        )
        ev2.run(verbose=False)
        metrics = ev2.compute_full_metrics()
        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_store_info_via_evaluator(self, tmp_path):
        store_path = str(tmp_path / "runs.jsonl")
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev.run(verbose=False)
        info = ev.store_info()
        assert info["n_records"] == 4
        assert "none" in info["regimes"]

    def test_no_store_path_still_works(self):
        """Evaluator without store_path should run normally without saving."""
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=None, append=False)
        df = ev.run(verbose=False)
        assert len(df) == 4

    def test_incremental_evidence_grows_record_count(self, tmp_path):
        """Three incremental sessions should accumulate without duplication."""
        store_path = str(tmp_path / "runs.jsonl")
        for n_seeds in [2, 4, 6]:
            ev = MonteCarloEvaluator(
                n_seeds=n_seeds, n_events=200, regimes=["none", "strong"],
                variant_configs={"default": {}}, n_workers=1,
                store_path=store_path, append=True,
            )
            ev.run(verbose=False)

        total = RunStore(store_path).load().__len__()
        # 6 seeds × 2 regimes = 12, no duplicates
        assert total == 12

    def test_store_records_survive_evaluator_teardown(self, tmp_path):
        """Records must be on disk and loadable independently of the evaluator object."""
        store_path = str(tmp_path / "runs.jsonl")
        ev = MonteCarloEvaluator(**TINY_KWARGS, store_path=store_path, append=False)
        ev.run(verbose=False)
        del ev
        records = RunStore(store_path).load()
        assert len(records) == 4
        for rec in records:
            assert "seed" in rec
            assert "alarm_level" in rec
