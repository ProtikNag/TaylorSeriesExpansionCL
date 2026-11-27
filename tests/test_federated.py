"""Checks for FedAvg / FedProx consolidation: `.venv/bin/python -m pytest -q tests/test_federated.py`."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from htcl.methods.federated import fedavg_update, fedprox_penalty


def test_fedavg_running_average_equals_sample_weighted_mean() -> None:
    """Three clients aggregated one by one give sum_c n_c w_c / sum_c n_c; integer buffers are copied."""
    torch.manual_seed(0)
    clients = [nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2)) for _ in range(3)]
    for i, client in enumerate(clients):
        client[1].running_mean.fill_(float(i))
        client[1].num_batches_tracked.fill_(10 * i)
    counts = [5, 2, 3]
    global_model = copy.deepcopy(clients[0])
    seen = counts[0]
    for client, n in zip(clients[1:], counts[1:]):
        fedavg_update(global_model, client, seen, n)
        seen += n
    for name, value in global_model.state_dict().items():
        if value.is_floating_point():
            exact = sum(n * c.state_dict()[name] for c, n in zip(clients, counts)) / sum(counts)
            assert torch.allclose(value, exact, atol=1e-6), name
    assert int(global_model[1].num_batches_tracked) == 20
    assert torch.allclose(global_model[1].running_mean, torch.full((2,), (0 * 5 + 1 * 2 + 2 * 3) / 10))


def test_fedprox_penalty_value_and_gradient() -> None:
    """(mu/2)||w - a||^2 with gradient mu (w - a)."""
    model = nn.Linear(2, 1)
    anchor = [torch.zeros_like(p) for p in model.parameters()]
    mu = 0.3
    penalty = fedprox_penalty(model, anchor, mu)
    expected = 0.5 * mu * sum(p.detach().pow(2).sum() for p in model.parameters())
    assert torch.allclose(penalty, expected)
    penalty.backward()
    for p in model.parameters():
        assert torch.allclose(p.grad, mu * p.detach())
