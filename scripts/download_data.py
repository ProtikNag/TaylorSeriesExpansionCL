#!/usr/bin/env python
"""Download the four core datasets through the engine's own loaders, so paths match exactly.

Run on a COMPUTE node with internet (never the login node). build_data downloads the full
dataset before any subsampling, so the tiny samples_per_class here only keeps the in-memory
build cheap; the on-disk download is complete and reused by every real run.

    MNIST       -> data/MNIST
    CIFAR-100   -> data/cifar-100-python
    20 Newsgroups -> scikit-learn cache (~/scikit_learn_data)
    Cora        -> data/Planetoid
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from htcl.data import build_data

SPECS = [
    {"name": "splitmnist", "num_tasks": 5, "classes_per_task": 2, "samples_per_class": 1},
    {"name": "cifar100", "num_tasks": 10, "samples_per_class": 1},
    {"name": "news20", "num_tasks": 5, "samples_per_class": 1},
    {"name": "cora", "num_tasks": 3, "samples_per_class": 1},
]

for spec in SPECS:
    print(f"[download] {spec['name']} ...", flush=True)
    data = build_data(spec)
    print(f"[ok] {spec['name']}: {len(data.tasks)} tasks built", flush=True)

print("ALL_DATA_DOWNLOADED")
