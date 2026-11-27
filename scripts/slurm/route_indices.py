#!/usr/bin/env python
"""Split a planned experiment's job indices into CPU-class and GPU-class array lists.

One `--job N` index equals line N (0-based) of the plan, so the lists printed here feed
straight into `sbatch --array=...`. GPU-class = CIFAR-100 / ImageNet datasets (heavy);
everything else (SplitMNIST, 20 Newsgroups, Cora) is light enough for CPU nodes. Aliased
canonical duplicates (group-schedule HTCL only) are dropped, since the runner skips them.

Usage:
    python scripts/slurm/route_indices.py configs/e1_main.yaml --samples-per-class 50 --orders 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from htcl.run import write_plan

GPU_KEYS = ("cifar100", "imagenet")


def compress(indices: list[int]) -> str:
    """Render a sorted index list as a Slurm --array string (ranges where contiguous)."""
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--samples-per-class", type=int)
    p.add_argument("--orders", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--runs-root", default="runs")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    overrides = {"samples_per_class": a.samples_per_class, "orders": a.orders,
                 "epochs": a.epochs, "runs_root": a.runs_root}
    root, jobs = write_plan(cfg, overrides)

    cpu, gpu, alias = [], [], 0
    for idx, j in enumerate(jobs):
        if j.get("alias_of"):
            alias += 1
            continue
        (gpu if any(k in j["dataset_label"] for k in GPU_KEYS) else cpu).append(idx)

    print(f"# plan: {root}")
    print(f"# total={len(jobs)} runnable={len(cpu) + len(gpu)} aliased(skipped)={alias}")
    print(f"# CPU-class jobs (SplitMNIST/20NG/Cora): {len(cpu)}")
    print(f"CPU_ARRAY={compress(cpu)}")
    print(f"# GPU-class jobs (CIFAR-100/ImageNet): {len(gpu)}")
    print(f"GPU_ARRAY={compress(gpu)}")


if __name__ == "__main__":
    main()
