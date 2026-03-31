"""
Persistent JSONL store for Monte Carlo run records.

Each line in the store file is a self-contained JSON object representing one
completed simulation+audit run.  Fields stored per record:

  run_id         str   UUID v4 — unique identifier for this record
  run_timestamp  str   ISO-8601 wall-clock time the run finished
  seed           int
  regime         str
  variant        str   variant config name (e.g. "default", "Y_only")
  n_events       int
  alarm_level    str
  alarm_any      int   0/1
  alarm_red      int   0/1
  max_odds_ratio float
  dual_signal    int   0/1
  n_flagged      int
  g3_purity      float | None
  g3_recall      float | None
  n_g3_events    int
  n_total_events int
  injection_true bool

The JSONL format allows atomic line-level appends without reading the whole
file, making it safe to incrementally grow the evidence base across sessions.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class RunStore:
    """
    JSONL-backed store for Monte Carlo run records.

    Parameters
    ----------
    path : str | Path
        Path to the .jsonl file.  Created on first write if absent.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, records: List[Dict]) -> None:
        """
        Append *records* to the store.  Each record receives a ``run_id``
        and ``run_timestamp`` if not already present.
        """
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as fh:
            for rec in records:
                out = dict(rec)
                out.setdefault("run_id", str(uuid.uuid4()))
                out.setdefault("run_timestamp", now)
                # Ensure JSON-serialisable
                out = _coerce_json(out)
                fh.write(json.dumps(out) + "\n")
        logger.info("Appended %d records to %s", len(records), self.path)

    def overwrite(self, records: List[Dict]) -> None:
        """Replace the store with *records* (clears existing content)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.append(records)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> List[Dict]:
        """Return all records in the store as a list of dicts."""
        if not self.path.exists():
            return []
        records: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %d in %s: %s", lineno, self.path, exc)
        return records

    def load_summary_df(self) -> pd.DataFrame:
        """Load all records into a DataFrame (empty if store is absent/empty)."""
        records = self.load()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def completed_keys(self) -> Set[Tuple[int, str, str]]:
        """
        Return the set of ``(seed, regime, variant)`` tuples already in the
        store.  Used to skip jobs that were finished in a previous session.
        """
        keys: Set[Tuple[int, str, str]] = set()
        for rec in self.load():
            seed = rec.get("seed")
            regime = rec.get("regime")
            variant = rec.get("variant", "default")
            if seed is not None and regime is not None:
                keys.add((int(seed), str(regime), str(variant)))
        return keys

    def filter_new_jobs(self, jobs: List[Tuple]) -> Tuple[List[Tuple], int]:
        """
        Given a list of job tuples ``(seed, regime, n_events, eligibility_rate,
        vcfg, stage_coverage, yellow_thresh, red_thresh, mode, variant_name)``,
        return ``(new_jobs, n_skipped)`` where *new_jobs* contains only the
        jobs whose ``(seed, regime, variant_name)`` are not yet in the store.
        """
        done = self.completed_keys()
        new_jobs = []
        n_skipped = 0
        for job in jobs:
            seed, regime = job[0], job[1]
            variant_name = job[9] if len(job) > 9 else "default"
            if (int(seed), str(regime), str(variant_name)) in done:
                n_skipped += 1
            else:
                new_jobs.append(job)
        return new_jobs, n_skipped

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        """Return a brief summary dict about the current store contents."""
        records = self.load()
        if not records:
            return {"n_records": 0, "path": str(self.path), "exists": self.path.exists()}
        df = pd.DataFrame(records)
        return {
            "n_records": len(records),
            "path": str(self.path),
            "regimes": sorted(df["regime"].unique().tolist()) if "regime" in df.columns else [],
            "variants": sorted(df["variant"].unique().tolist()) if "variant" in df.columns else [],
            "seeds_range": [int(df["seed"].min()), int(df["seed"].max())] if "seed" in df.columns else [],
            "earliest": df["run_timestamp"].min() if "run_timestamp" in df.columns else None,
            "latest": df["run_timestamp"].max() if "run_timestamp" in df.columns else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_json(obj):
    """Recursively ensure *obj* is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(v) for v in obj]
    if isinstance(obj, float):
        # NaN/Inf are not valid JSON
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if hasattr(obj, "tolist"):        # numpy array or matrix → list
        return _coerce_json(obj.tolist())
    if hasattr(obj, "item"):          # numpy 0-d scalar → python scalar
        return obj.item()
    return obj
