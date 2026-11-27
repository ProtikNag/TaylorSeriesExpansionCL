"""CLI: summary tables of one experiment -> runs/<exp>/summary.{csv,json}.

    python -m htcl.aggregate configs/e1_main.yaml [--samples-per-class 50] [--orders 3] [--epochs 1]

Per (dataset, method, variant): mean and sample std (ddof=1) across orders of ACC, forgetting and
BWT; forgetting/BWT on HTCL's group basis (`forgetting_g{k}`, `bwt_g{k}` for each group-schedule HTCL k in
the experiment; use these when HTCL and baselines share a table); per-task final accuracy mean/std by
task id; mean wall-clock; n_orders (rows with results, aliased HTCL duplicates included with the
canonical run's metrics) and n_unique_runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import group_k
from .metrics import mean_std
from .run import parse_args, plan_jobs, run_dir


def _read(path: Path) -> dict | None:
    """JSON file or None."""
    return json.loads(path.read_text()) if path.exists() else None


def collect(jobs: list[dict], root: Path) -> dict[tuple[str, str, str], list[dict]]:
    """(dataset, method, variant) -> per-order records {metrics, seconds, unique}."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    pending: dict[tuple[str, str], int] = {}
    for job in jobs:
        source = root / (job.get("alias_of") or job["job_id"])
        metrics, timing = _read(source / "metrics.json"), _read(source / "timing.json")
        key = (job["dataset_label"], job["method_label"])
        if metrics is None or not (source / "DONE").exists():
            pending[key] = pending.get(key, 0) + 1
            continue
        variants = metrics.get("variants", {"": metrics})
        for variant, m in variants.items():
            groups.setdefault((*key, variant), []).append(
                {"metrics": m, "seconds": timing["total_seconds"], "unique": not job.get("alias_of")})
    return groups, pending


def summarize(groups: dict, pending: dict, ks: list[int] = ()) -> list[dict]:
    """One summary row per group; ks: HTCL group sizes for the group-basis forgetting/BWT columns."""
    rows = []
    for (dataset, method, variant), records in groups.items():
        row = {"dataset": dataset, "method": method, "variant": variant, "n_orders": len(records),
               "n_unique_runs": sum(r["unique"] for r in records), "n_missing": pending.get((dataset, method), 0)}
        for metric in ("acc", "forgetting", "bwt"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std([r["metrics"][metric] for r in records])
        for k in ks:  # NaN for rows without that basis (HTCL with another k, runs from older code)
            for metric in ("forgetting", "bwt"):
                row[f"{metric}_g{k}_mean"], row[f"{metric}_g{k}_std"] = mean_std(
                    [r["metrics"].get("group_basis", {}).get(str(k), {}).get(metric, np.nan) for r in records])
        row["time_mean"], row["time_std"] = mean_std([r["seconds"] for r in records])
        per_task = np.array([r["metrics"]["final_per_task"] for r in records])
        for t in range(per_task.shape[1]):
            row[f"task{t}_mean"], row[f"task{t}_std"] = mean_std(per_task[:, t].tolist())
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    _, cfg, overrides = parse_args(argv, commands=("aggregate",))
    root = run_dir(cfg, overrides)
    jobs = plan_jobs(cfg, overrides)
    groups, pending = collect(jobs, root)
    ks = sorted({k for k in map(group_k, jobs) if k})  # group-schedule HTCL only
    rows = summarize(groups, pending, ks)
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary.csv", index=False)
    clean = [{k: None if isinstance(v, float) and np.isnan(v) else v for k, v in r.items()} for r in rows]
    (root / "summary.json").write_text(json.dumps(clean, indent=2))
    cols = [c for c in ("dataset", "method", "variant", "n_orders", "n_unique_runs", "n_missing", "acc_mean",
                        "acc_std", "forgetting_mean", "forgetting_std", "bwt_mean", "bwt_std",
                        *[f"{m}_g{k}_mean" for k in ks for m in ("forgetting", "bwt")], "time_mean")
            if c in frame]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        print(frame[cols].round(2).to_string(index=False) if len(frame) else "no finished jobs")
    print(f"wrote {root}/summary.csv and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
