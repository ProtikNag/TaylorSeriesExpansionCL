"""Task-order sampling, grouping and canonicalisation."""

from __future__ import annotations

import itertools
import math
import random


def sample_orders(num_tasks: int, n: int, seed: int = 0) -> list[tuple[int, ...]]:
    """Draw min(n, T!) distinct orders uniformly at random; all T! (lexicographic) when T! <= n.

    Draws are sequential with rejection, so the first m orders for a given seed do not depend on n
    (as long as n < T!), which keeps reduced runs a prefix of full runs.
    """
    if math.factorial(num_tasks) <= n:
        return list(itertools.permutations(range(num_tasks)))
    rng = random.Random(seed)
    orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(orders) < n:
        perm = list(range(num_tasks))
        rng.shuffle(perm)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            orders.append(tuple(perm))
    return orders


def groups(order: tuple[int, ...] | list[int], k: int) -> list[list[int]]:
    """Split an order into consecutive groups of size k (the last group may be smaller)."""
    return [list(order[i:i + k]) for i in range(0, len(order), k)]


def canonical(order: tuple[int, ...] | list[int], k: int) -> tuple[int, ...]:
    """Sort task ids within each consecutive group of size k."""
    return tuple(t for group in groups(order, k) for t in sorted(group))
