"""Consolidation steps: Taylor update (eq. hier_update), diagonal curvature, DiLoCo, BN recalibration.

All curvature and gradient computations run in eval mode (no BatchNorm statistic updates, no dropout).
Parameter lists follow `model.parameters()` order restricted to requires_grad.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from .data import make_loader

PER_SAMPLE_FLOATS = 1e8  # per-sample-gradient memory budget per vmap chunk (floats, ~400 MB)


def trainable(model: nn.Module) -> list[torch.Tensor]:
    """Trainable parameters in a fixed order."""
    return [p for p in model.parameters() if p.requires_grad]


def _subset(dataset: Dataset, n: int) -> Dataset:
    """Random subset of at most n items (torch global RNG, so it is reproducible per job)."""
    if len(dataset) <= n:
        return dataset
    return Subset(dataset, torch.randperm(len(dataset))[:n].tolist())


class _EvalMode:
    """Context manager: model.eval() inside, previous mode restored on exit."""

    def __init__(self, model: nn.Module):
        """Remember the model."""
        self.model, self.was_training = model, model.training

    def __enter__(self) -> nn.Module:
        """Switch to eval mode."""
        self.model.eval()
        return self.model

    def __exit__(self, *exc) -> None:
        """Restore the previous mode."""
        self.model.train(self.was_training)


def mean_gradient(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int | str) -> list[torch.Tensor]:
    """g = gradient of the mean cross-entropy over `dataset` at the current weights (eval mode)."""
    params = trainable(model)
    total = [torch.zeros_like(p) for p in params]
    with _EvalMode(model):
        for x, y in make_loader(dataset, batch_size, shuffle=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, reduction="sum") / len(dataset)
            for acc, grad in zip(total, torch.autograd.grad(loss, params)):
                acc += grad
    return total


def fisher_diag(model: nn.Module, dataset: Dataset, device: torch.device, n_samples: int) -> list[torch.Tensor]:
    """Diagonal empirical Fisher: mean over up to n_samples of squared per-sample gradients.

    Uses torch.func.vmap(grad) in memory-bounded chunks; falls back to a per-sample loop for models
    whose ops vmap does not support.
    """
    data = _subset(dataset, n_samples)
    params = trainable(model)
    chunk = max(1, int(PER_SAMPLE_FLOATS // sum(p.numel() for p in params)))
    total = [torch.zeros_like(p) for p in params]
    with _EvalMode(model):
        try:
            _fisher_vmap(model, data, device, chunk, total)
        except (RuntimeError, NotImplementedError) as err:
            warnings.warn(f"vmap per-sample gradients failed ({err}); using a per-sample loop")
            total = [torch.zeros_like(p) for p in params]
            for x, y in make_loader(data, 1, shuffle=False):
                loss = F.cross_entropy(model(x.to(device)), y.to(device))
                for acc, grad in zip(total, torch.autograd.grad(loss, params)):
                    acc += grad.pow(2)
    return [acc / len(data) for acc in total]


def _fisher_vmap(model: nn.Module, data: Dataset, device: torch.device, chunk: int, total: list[torch.Tensor]) -> None:
    """Accumulate summed squared per-sample gradients into `total` with torch.func."""
    from torch.func import functional_call, grad, vmap
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())

    def sample_loss(p: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    per_sample_grad = vmap(grad(sample_loss), in_dims=(None, 0, 0))
    for x, y in make_loader(data, chunk, shuffle=False):
        grads = per_sample_grad(params, x.to(device), y.to(device))
        for acc, name in zip(total, names):
            acc += grads[name].pow(2).sum(0)


def hutchinson_diag(model: nn.Module, dataset: Dataset, device: torch.device, n_samples: int,
                    batch_size: int | str, probes: int) -> list[torch.Tensor]:
    """Diagonal Hessian of the mean loss, E[z * Hz] with Rademacher z, on up to n_samples items.

    Hz is a forward-over-reverse product (torch.func.jvp of grad), which also works on MPS where
    double backward through max-pooling is unsupported. Negative entries are clipped to 0 so that
    H + lam*I stays positive (the theorem needs lam > -mu_min).
    """
    from torch.func import functional_call, grad, jvp
    data = _subset(dataset, n_samples)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    total = {n: torch.zeros_like(p) for n, p in params.items()}
    with _EvalMode(model):
        for _ in range(probes):
            for x, y in make_loader(data, batch_size, shuffle=False):
                x, y = x.to(device), y.to(device)

                def loss_fn(p: dict) -> torch.Tensor:
                    out = functional_call(model, (p, buffers), (x,))
                    return F.cross_entropy(out, y, reduction="sum") / len(data)

                z = {n: torch.randint(0, 2, p.shape, device=p.device).to(p.dtype) * 2 - 1 for n, p in params.items()}
                _, hz = jvp(grad(loss_fn), (params,), (z,))
                for n in names:
                    total[n] += z[n] * hz[n]
    return [(total[n] / probes).clamp_min(0) for n in names]


def taylor_step(w_prev: list[torch.Tensor], target: list[torch.Tensor], g: list[torch.Tensor], h: list[torch.Tensor],
                lam: float, step_scale: float = 1.0, clip_norm: float | None = None) -> list[torch.Tensor]:
    """eq. hier_update with diagonal H: w_prev + eta * (H + lam I)^{-1} (lam * (target - w_prev) - g).

    `clip_norm` caps the global L2 norm of the update (off by default).
    """
    update = [(lam * (t - w) - gi) / (hi + lam) for w, t, gi, hi in zip(w_prev, target, g, h)]
    scale = step_scale
    if clip_norm is not None:
        norm = torch.sqrt(sum(u.pow(2).sum() for u in update))
        scale *= min(1.0, clip_norm / (float(norm) + 1e-12))
    return [w + scale * u for w, u in zip(w_prev, update)]


def diloco_step(w_prev: list[torch.Tensor], target: list[torch.Tensor], momentum: list[torch.Tensor],
                lr: float = 1.0, mu: float = 0.0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Outer Nesterov step on the pseudo-gradient w_prev - target (DiLoCo). Returns (weights, momentum)."""
    pseudo = [w - t for w, t in zip(w_prev, target)]
    momentum = [mu * m + d for m, d in zip(momentum, pseudo)]
    return [w - lr * (d + mu * m) for w, d, m in zip(w_prev, pseudo, momentum)], momentum


def assign(model: nn.Module, values: list[torch.Tensor]) -> None:
    """Copy values into the model's trainable parameters."""
    with torch.no_grad():
        for p, v in zip(trainable(model), values):
            p.copy_(v)


def average_states(states: list[dict]) -> dict:
    """Elementwise mean of state dicts (integer buffers are taken from the first)."""
    return {k: torch.stack([s[k] for s in states]).mean(0) if states[0][k].is_floating_point() else states[0][k]
            for k in states[0]}


@torch.no_grad()
def recalibrate_bn(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int | str) -> None:
    """Reset BatchNorm running stats and re-estimate them (cumulative average) in train mode.

    The loader is shuffled (torch global RNG, restored by the checkpoint): the consolidation data is a
    concatenation of task blocks, and task-homogeneous batches would leave the between-task spread
    of the channel means out of running_var. make_loader drops the incomplete trailing batch, so
    every batch gets the same weight in the cumulative average.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average = exact mean over the passes
    was_training = model.training
    model.train()
    for x, _ in make_loader(dataset, batch_size, shuffle=True):
        if len(x) > 1:  # BN cannot normalise a single sample in train mode
            model(x.to(device))
    model.train(was_training)
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
