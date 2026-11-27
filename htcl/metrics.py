"""ACC, forgetting, BWT from an accuracy matrix, and aggregation across orders."""

from __future__ import annotations

import numpy as np

from .permutations import groups


def compute_metrics(matrix: list[list[float]], order: list[int], learned_pos: dict[int, int] | None = None) -> dict:
    """Metrics of one run.

    Args:
        matrix: A[l][j], row l = after training position l, column j = task id (percent).
        order: task ids in learning order.
        learned_pos: row at which task j counts as learned. Defaults to its position in the order;
            for HTCL it is the last position of the task's group (the deployed model only sees
            the task once the group is consolidated).

    Returns:
        dict with acc, forgetting, bwt (NaN when no task can be forgotten) and final_per_task.
    """
    acc_matrix = np.asarray(matrix, dtype=float)
    last = len(order) - 1
    learned = learned_pos or {t: p for p, t in enumerate(order)}
    # Only tasks learned before the final row can be forgotten.
    forgettable = [t for t in order if learned[t] < last]
    final = acc_matrix[last]
    if forgettable:
        forgetting = float(np.mean([acc_matrix[learned[t]:last, t].max() - final[t] for t in forgettable]))
        bwt = float(np.mean([final[t] - acc_matrix[learned[t], t] for t in forgettable]))
    else:
        forgetting = bwt = float("nan")
    return {"acc": float(final.mean()), "forgetting": forgetting, "bwt": bwt, "final_per_task": final.tolist()}


def learned_positions(steps: list[list[int]]) -> dict[int, int]:
    """Row at which each task counts as learned: the last position of its step (HTCL: its group)."""
    learned, pos = {}, -1
    for step in steps:
        pos += len(step)
        learned.update({t: pos for t in step})
    return learned


def group_basis(matrix: list[list[float]], order: list[int], ks) -> dict[str, dict]:
    """Forgetting and BWT on HTCL's basis for each group size k: learned at the end of the task's group
    of k, F without the last group. Baselines report every k, so their columns match HTCL-k's."""
    out = {}
    for k in ks:
        m = compute_metrics(matrix, order, learned_positions(groups(order, k)))
        out[str(k)] = {"forgetting": m["forgetting"], "bwt": m["bwt"]}
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1); std is NaN for fewer than two values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
