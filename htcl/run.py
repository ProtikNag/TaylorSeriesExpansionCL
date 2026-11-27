"""CLI: plan / run / status for one experiment config.

    python -m htcl.run plan   configs/e1_main.yaml [--samples-per-class 50] [--orders 3] [--epochs 1]
    python -m htcl.run run    configs/e1_main.yaml (--job N | --all) [--filter method=ser ...] [overrides]
    python -m htcl.run status configs/e1_main.yaml [overrides]
    (--runs-root DIR places the experiment directories under DIR instead of runs/;
     --max-steps N stops each job after N steps, leaving a checkpoint to resume from)

Config schema (YAML): experiment, orders, orders_seed (default 0), seed (default PAPER_SEED), and either `datasets` + `methods`
(their cartesian product) or `blocks: [{datasets, methods}, ...]`. A dataset or method entry may
carry `sweep: {dotted.key: [values]}`, expanded as a cartesian product.
Overrides change the run directory suffix (runs/<experiment>__spc50_o3_ep1/).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import yaml

from .data import DEFAULT_TASKS
from .hierarchy import NO_BUFFER_METHODS
from .permutations import canonical, sample_orders
from .utils import stable_hash

PAPER_SEED = 0  # the single training seed of every job in the paper (task orders carry the variation)
DATASET_DEFAULTS = {"scenario": "domain", "seed": 0, "momentum": 0.9, "weight_decay": 1e-4, "data_root": "data"}
REQUIRED_DATASET_KEYS = ("lr", "batch_size", "epochs", "buffer_size")


def _set_dotted(spec: dict, key: str, value) -> None:
    """spec['a']['b'] = value for key 'a.b'."""
    *parents, leaf = key.split(".")
    for p in parents:
        spec = spec.setdefault(p, {})
    spec[leaf] = value


def flatten(spec: dict, prefix: str = "") -> dict:
    """{'htcl': {'levels': 2}} -> {'htcl.levels': 2}."""
    flat = {}
    for k, v in spec.items():
        key = f"{prefix}{k}"
        flat.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return flat


def expand(entry: dict) -> list[tuple[dict, dict]]:
    """Expand an entry's `sweep` into (spec, swept values) pairs."""
    entry = copy.deepcopy(entry)
    sweep = entry.pop("sweep", {}) or {}
    out = []
    for combo in itertools.product(*sweep.values()):
        spec, tags = copy.deepcopy(entry), dict(zip(sweep, combo))
        for k, v in tags.items():
            _set_dotted(spec, k, v)
        out.append((spec, tags))
    return out


def resolve_dataset(spec: dict, overrides: dict) -> dict:
    """Dataset spec with defaults and CLI overrides applied."""
    spec = {**DATASET_DEFAULTS, **spec}
    spec.setdefault("num_tasks", DEFAULT_TASKS[spec["name"]])
    if overrides.get("samples_per_class") is not None:
        spec["samples_per_class"] = overrides["samples_per_class"]
    if overrides.get("epochs") is not None:
        spec["epochs"] = overrides["epochs"]
    missing = [k for k in REQUIRED_DATASET_KEYS if k not in spec]
    if missing:
        raise ValueError(f"dataset {spec['name']} lacks {missing}")
    return spec


def resolve_method(spec: dict, dataset: dict) -> dict:
    """Method spec with the method's own defaults (for the dataset) and HTCL defaults filled in, so every
    hyperparameter enters the job id and job.json. HTCL's lam None resolves to 1/lr of the dataset."""
    from .methods import get_method
    spec = {**get_method(spec["name"]).defaults(dataset["name"]), **copy.deepcopy(spec)}
    if "htcl" in spec:
        from .hierarchy import resolve
        spec["htcl"] = resolve(spec["htcl"] or {}, dataset["lr"])
    return spec


def label(name: str, written: dict) -> str:
    """Readable label: name plus the as-written non-name keys."""
    extras = [f"{k}={v}" for k, v in sorted(flatten(written).items()) if k != "name"]
    return "|".join([name, *extras])


def run_dir(cfg: dict, overrides: dict) -> Path:
    """<runs_root>/<experiment>[__spcN_oN_epN]."""
    tags = [f"{short}{overrides[key]}" for key, short in (("samples_per_class", "spc"), ("orders", "o"),
                                                           ("epochs", "ep")) if overrides.get(key) is not None]
    return Path(overrides.get("runs_root") or "runs") / (cfg["experiment"] + ("__" + "_".join(tags) if tags else ""))


def plan_jobs(cfg: dict, overrides: dict) -> list[dict]:
    """All jobs of an experiment; canonical duplicates of group-schedule HTCL jobs carry alias_of."""
    experiment = cfg["experiment"]
    n_orders = overrides.get("orders") or cfg["orders"]
    seed = int(cfg.get("seed", PAPER_SEED))
    blocks = cfg.get("blocks") or [{"datasets": cfg["datasets"], "methods": cfg["methods"]}]
    jobs, canon_owner = [], {}
    for block in blocks:
        for ds_entry in block["datasets"]:
            for ds_written, ds_tags in expand(ds_entry):
                dataset = resolve_dataset(ds_written, overrides)
                ds_tags = {**ds_tags, **({"scenario": "class"} if dataset["scenario"] == "class" else {})}
                if dataset.get("imagenet_root") and (Path(dataset["imagenet_root"]) / "PROXY.txt").exists():
                    ds_tags["proxy"] = True  # Tiny-ImageNet stand-in (scripts/make_imagenet100_proxy.py)
                ds_label = label(dataset["name"], ds_tags)
                orders = sample_orders(dataset["num_tasks"], n_orders, cfg.get("orders_seed", 0))
                for m_entry in block["methods"]:
                    for m_written, _ in expand(m_entry):
                        method = resolve_method(m_written, dataset)
                        for order_index, order in enumerate(orders):
                            # One seed for every job of the paper: results vary only with the order.
                            job = {"experiment": experiment, "dataset": dataset, "method": method,
                                   "order": list(order), "order_index": order_index, "seed": seed}
                            job["job_id"] = stable_hash(job)
                            htcl = method.get("htcl")
                            m_tags = m_written
                            if htcl and method["name"] in NO_BUFFER_METHODS:  # HTCL adds an N-sample reservoir
                                m_tags = {**m_written, "+replay": dataset["buffer_size"]}
                            job["dataset_label"], job["method_label"] = ds_label, label(method["name"], m_tags)
                            # Only the group schedule with search best is invariant to the order inside a
                            # group; online rows depend on the arrival order, so online jobs never alias.
                            if htcl and htcl["schedule"] == "group" and htcl["search"] == "best":
                                key = stable_hash([dataset, method, canonical(order, htcl["k"])])
                                job["alias_of"] = canon_owner.setdefault(key, job["job_id"])
                                if job["alias_of"] == job["job_id"]:
                                    job["alias_of"] = None
                            jobs.append(job)
    return jobs


def matches(job: dict, filters: list[str]) -> bool:
    """--filter key=value: key matches a dotted field of dataset/method by full name or suffix;
    'method' and 'dataset' mean their names, 'htcl' is True/False."""
    flat = {**flatten(job["dataset"], "dataset."), **flatten(job["method"], "method."),
            "method": job["method"]["name"], "dataset": job["dataset"]["name"], "htcl": "htcl" in job["method"],
            "order_index": job["order_index"], "job_id": job["job_id"]}
    for f in filters:
        key, value = f.split("=", 1)
        if not any((k == key or k.endswith("." + key)) and str(v) == value for k, v in flat.items()):
            return False
    return True


def job_state(job: dict, root: Path) -> str:
    """done | partial | pending | alias."""
    if job.get("alias_of"):
        return "alias"
    out = root / job["job_id"]
    return "done" if (out / "DONE").exists() else "partial" if (out / "checkpoint.pt").exists() else "pending"


def write_plan(cfg: dict, overrides: dict) -> tuple[Path, list[dict]]:
    """Plan and write runs/<exp>/jobs.jsonl (deterministic, rewritten each time)."""
    root = run_dir(cfg, overrides)
    jobs = plan_jobs(cfg, overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return root, jobs


def _print_counts(jobs: list[dict], root: Path, with_state: bool) -> None:
    """One line per (dataset, method) with job counts."""
    groups: dict[tuple[str, str], Counter] = {}
    for job in jobs:
        c = groups.setdefault((job["dataset_label"], job["method_label"]), Counter())
        c["jobs"] += 1
        c[job_state(job, root) if with_state else ("alias" if job.get("alias_of") else "runnable")] += 1
    keys = ("jobs", "done", "partial", "pending", "alias") if with_state else ("jobs", "runnable", "alias")
    for (ds, method), c in groups.items():
        print(f"  {ds:<28} {method:<48} " + "  ".join(f"{k}={c[k]}" for k in keys))
    total = Counter()
    for c in groups.values():
        total.update(c)
    print("  TOTAL " + "  ".join(f"{k}={total[k]}" for k in keys))


def parse_args(argv: list[str] | None = None, commands: tuple[str, ...] = ("plan", "run", "status")):
    """Shared argument parser (also used by htcl.aggregate)."""
    parser = argparse.ArgumentParser(prog="htcl.run")
    if len(commands) > 1:
        parser.add_argument("command", choices=commands)
    parser.add_argument("config")
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--orders", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--job", type=int, help="index into jobs.jsonl (0-based)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--filter", action="append", default=[], help="key=value, repeatable")
    parser.add_argument("--device", help="cpu | mps | cuda (default: best available)")
    parser.add_argument("--runs-root", default="runs", help="parent of the experiment run directories")
    parser.add_argument("--max-steps", type=int, help="stop each job after this many steps (tests resume)")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    overrides = {"samples_per_class": args.samples_per_class, "orders": args.orders, "epochs": args.epochs,
                 "runs_root": args.runs_root}
    return args, cfg, overrides


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args, cfg, overrides = parse_args(argv)
    root, jobs = write_plan(cfg, overrides)
    if args.command == "plan":
        n_alias = sum(1 for j in jobs if j.get("alias_of"))
        print(f"{root}/jobs.jsonl: {len(jobs)} jobs, {len(jobs) - n_alias} to run, "
              f"{n_alias} aliased canonical duplicates")
        _print_counts(jobs, root, with_state=False)
        return 0
    if args.command == "status":
        print(f"{root}")
        _print_counts([j for j in jobs if matches(j, args.filter)], root, with_state=True)
        return 0

    from .engine import run_job
    from .utils import get_device
    if args.job is not None:
        selected = [jobs[args.job]]
    elif args.all:
        selected = [j for j in jobs if matches(j, args.filter)]
    else:
        print("run needs --job N or --all", file=sys.stderr)
        return 2
    device = get_device(args.device)
    failures = 0
    for i, job in enumerate(selected):
        if job.get("alias_of"):
            print(f"[{i + 1}/{len(selected)}] {job['job_id']} alias of {job['alias_of']}, skipped")
            continue
        print(f"[{i + 1}/{len(selected)}] {job['job_id']} {job['dataset_label']} {job['method_label']} "
              f"order#{job['order_index']} {job['order']}", flush=True)
        try:
            run_job(job, root / job["job_id"], device, max_steps=args.max_steps)
        except Exception:  # keep going under --all; the failed job resumes from its checkpoint
            failures += 1
            traceback.print_exc()
            if args.job is not None:
                return 1
    print(f"finished: {len(selected) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
