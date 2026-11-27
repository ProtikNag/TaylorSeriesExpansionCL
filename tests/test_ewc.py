"""EWC checks: `.venv/bin/python -m pytest -q tests/test_ewc.py`."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from htcl.consolidation import recalibrate_bn
from htcl.methods.ewc import EWC


def _method(lam: float = 2.0) -> EWC:
    """EWC on a tiny BatchNorm model."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
    return EWC(model, {"lr": 0.1, "lam": lam, "fisher_samples": 16}, torch.device("cpu"))


def test_fisher_is_per_sample_and_train_mode_restored() -> None:
    """F = mean_i grad_i^2 (not the squared batch-mean gradient), model back in train mode with BN updating."""
    method = _method()
    x, y = torch.randn(8, 3), torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    method.model.train()
    stats_before = copy.deepcopy(method.model[1].state_dict())
    torch.manual_seed(1)
    method.end_task(0, loader)
    assert method.model.training and all(m.training for m in method.model.modules())
    # The Fisher's BN recalibration runs on a copy: the model's own statistics are untouched.
    assert all(torch.equal(v, method.model[1].state_dict()[k]) for k, v in stats_before.items())

    # Reference: eval mode at recalibrated BN statistics (same shuffle seed), as the Fisher is computed.
    reference = copy.deepcopy(method.model)
    torch.manual_seed(1)
    recalibrate_bn(reference, loader.dataset, torch.device("cpu"), loader.batch_size)
    reference.eval()
    params = list(reference.parameters())
    expected = [torch.zeros_like(p) for p in params]
    for i in range(8):
        grads = torch.autograd.grad(F.cross_entropy(reference(x[i:i + 1]), y[i:i + 1]), params)
        for e, g in zip(expected, grads):
            e += g.pow(2) / 8
    for got, e in zip(method.fishers[0], expected):
        assert torch.allclose(got, e, atol=1e-6)

    method.model.train()
    bn = method.model[1]
    before = bn.running_mean.clone()
    method.begin_task(1, loader)
    method.observe(x[:4], y[:4], 1)
    assert not torch.equal(before, bn.running_mean)  # BatchNorm is not frozen after the Fisher pass


def test_penalty_sums_tasks() -> None:
    """penalty = sum_t lam/2 * sum F_t (w - w_t*)^2, and survives a state round trip."""
    method = _method(lam=3.0)
    params = list(method.model.parameters())
    method.anchors = [[torch.zeros_like(p) for p in params], [torch.ones_like(p) for p in params]]
    method.fishers = [[torch.full_like(p, 2.0) for p in params], [torch.full_like(p, 0.5) for p in params]]
    expected = sum(1.5 * ((2.0 * p.pow(2)).sum() + (0.5 * (p - 1).pow(2)).sum()) for p in params)
    assert torch.allclose(method.penalty(), expected)
    clone = _method(lam=3.0)
    clone.load_state_dict(copy.deepcopy(method.state_dict()))
    assert torch.allclose(clone.penalty(), expected)


def test_penalty_stability_cap() -> None:
    """Where lr * lam * sum_t F_t > 1 the coordinate's penalty is scaled so its curvature is 1 / lr."""
    method = _method(lam=10.0)  # lr 0.1: lr * lam = 1
    params = list(method.model.parameters())
    method.anchors = [[torch.zeros_like(p) for p in params]] * 2
    method.fishers = [[torch.full_like(p, 3.0) for p in params], [torch.full_like(p, 1.0) for p in params]]
    # sum F = 4 -> s = 1/4: penalty = lam/2 * (1/4) * 4 * w^2 = lam/2 * w^2
    expected = sum(5.0 * p.pow(2).sum() for p in params)
    assert torch.allclose(method.penalty(), expected)
    method.fishers = [[torch.full_like(p, 0.25) for p in params]] * 2  # sum F = 0.5: below the cap
    assert torch.allclose(method.penalty(), sum(5.0 * 0.5 * p.pow(2).sum() for p in params))
