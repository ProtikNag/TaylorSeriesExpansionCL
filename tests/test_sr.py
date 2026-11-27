"""Spectral regularization checks: `.venv/bin/python -m pytest -q tests/test_sr.py`."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from htcl.methods.sr import SR


def _net() -> nn.Module:
    """Conv + BN + Linear: covers reshaped kernels, norm scales and biases."""
    return nn.Sequential(nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4 * 4 * 4, 3))


def test_regularizer_matches_exact_formula() -> None:
    """After enough persistent power-iteration steps, R equals the formula with exact sigma_1, and
    the state round trip keeps u."""
    torch.manual_seed(0)
    method = SR(_net(), {"lr": 0.1, "lam": 1.0, "k": 2}, torch.device("cpu"))
    for _ in range(200):  # u persists across calls, so repeated single iterations converge
        r = method.regularizer()
    conv, bn, lin = method.model[0], method.model[1], method.model[3]
    sigma = [torch.linalg.matrix_norm(w.reshape(w.shape[0], -1), ord=2) for w in (conv.weight, lin.weight)]
    exact = (sum((s ** 2 - 1) ** 2 for s in sigma) + (bn.weight ** 2 - 1).pow(2).sum()
             + sum(b.norm() ** 4 for b in (conv.bias, bn.bias, lin.bias)))
    assert torch.allclose(r, exact, rtol=1e-4)

    clone = SR(_net(), {"lr": 0.1}, torch.device("cpu"))
    clone.load_state_dict(copy.deepcopy(method.state_dict()))
    assert torch.allclose(clone.regularizer(), r, rtol=1e-4)


def test_gradient_of_sigma_is_u_v_transpose() -> None:
    """d sigma / d W = u v^T for the converged singular pair (sign-invariant check via SVD)."""
    torch.manual_seed(1)
    method = SR(nn.Linear(5, 4, bias=False), {"lr": 0.1}, torch.device("cpu"))
    for _ in range(200):
        method.spectral_norm("weight", method.model.weight)
    sigma = method.spectral_norm("weight", method.model.weight)
    sigma.backward()
    u, _, vh = torch.linalg.svd(method.model.weight.detach())
    assert torch.allclose(method.model.weight.grad, torch.outer(u[:, 0], vh[0]), atol=1e-4)
