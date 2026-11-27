"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

After task t the model is anchored at w_t* with the diagonal Fisher F_t, and every later step
minimises  CE + sum_t (lam / 2) * sum_i F_{t,i} (w_i - w*_{t,i})^2.

F_t is the empirical Fisher: the mean of squared per-sample gradients of the cross-entropy with the
true labels, over a seeded random subset of at most `fisher_samples` training items of task t, in eval
mode. It reuses `consolidation.fisher_diag`, the same estimator HTCL uses for its curvature. The eval
mode BatchNorm statistics are first recalibrated on the task's training data, on a copy of the model
(the model itself is untouched): the training EMA is far from converged after a short task, and the
per-sample gradients at those stale statistics are huge (CIFAR-100, task 1, 1 epoch: max F ~ 7e8 at
spc 20 / 50, against 65 / 24 after recalibration; spc 50, 3 epochs: 7.5 against 2.1).

Stability cap. The penalty's curvature on coordinate i is c_i = lam * sum_t F_{t,i}; SGD with momentum
0.9 diverges on it once lr * c_i > 2 (1 + 0.9) = 3.8, which a single large Fisher entry reaches at
reduced scale (CIFAR-100, lam 10, lr 0.1: lr * c_i = 65 after one task at spc 20). Each coordinate's
penalty is therefore scaled by s_i = min(1, 1 / (lr * c_i)), so lr * s_i * c_i <= 1. Coordinates
below that bound (the regime the method was tuned in) are unchanged.

lam is per dataset. The old code's 400 was calibrated for its Fisher (B times the squared batch-mean
gradient), which is orders of magnitude smaller than this per-sample estimate. At CIFAR-100's lr 0.1
SGD with momentum diverges at lam >= 50 (runs/_smoke/ewc_probe__spc50, cifar_probe__spc50), while
lam in {1, 10} trains (spc50, 10 epochs, one order: ACC 12.1 / 11.1, forgetting 19.9 / 10.6); we take
10, the strongest stable value of that sweep. The other datasets keep 400 (smaller lr; their smoke
runs train stably).

Under HTCL, `rebase` moves every anchor to w_1 when the local model is reset to it, so the local
search is anchored to the hierarchical model (Algorithm 1, TrainSeq(sigma, w_1)) and not to discarded
local optima; the Fishers are kept.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..consolidation import fisher_diag, recalibrate_bn, trainable
from .base import Method

DEFAULT_LAM = 400.0           # old code's ewc_lambda (htcl/methods/ewc.py on wip/imagenet-vit)
DATASET_LAM = {"cifar100": 10.0}  # see the module docstring
DEFAULT_FISHER_SAMPLES = 512  # same cap as HTCL's curvature_samples


class EWC(Method):
    """Cross-entropy plus one quadratic penalty per finished task (cfg keys: lam, fisher_samples)."""

    @classmethod
    def defaults(cls, dataset: str | None) -> dict:
        """Per-dataset lam (module docstring) and the Fisher sample cap."""
        return {"lam": DATASET_LAM.get(dataset, DEFAULT_LAM), "fisher_samples": DEFAULT_FISHER_SAMPLES}

    def __init__(self, model, cfg: dict, device: torch.device):
        """Create the method with empty anchor and Fisher lists."""
        super().__init__(model, cfg, device)
        self.lam = float(self.cfg["lam"])
        self.fisher_samples = int(self.cfg["fisher_samples"])
        # One entry per finished task, each a list aligned with trainable(self.model).
        # ponytail: 2 parameter copies per task (ResNet-18, 10 tasks ~ 0.9 GB); merge into
        # sum_t F_t and sum_t F_t w_t* (same gradient) if memory bites.
        self.anchors: list[list[torch.Tensor]] = []
        self.fishers: list[list[torch.Tensor]] = []

    def penalty(self) -> torch.Tensor:
        """sum_t (lam / 2) * sum_i s_i F_{t,i} (w_i - w*_{t,i})^2, s_i the stability cap (module docstring)."""
        params = trainable(self.model)
        lr_lam = float(self.cfg["lr"]) * self.lam
        caps = [(1.0 / (lr_lam * sum(fs))).clamp(max=1.0) for fs in zip(*self.fishers)]  # inf -> 1 where F = 0
        total = torch.zeros((), device=self.device)
        for anchor, fisher in zip(self.anchors, self.fishers):
            total = total + sum((s * f * (p - a).pow(2)).sum() for p, a, f, s in zip(params, anchor, fisher, caps))
        return 0.5 * self.lam * total

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One SGD step on CE + EWC penalty (the model stays in train mode)."""
        loss = F.cross_entropy(self.model(x), y)
        if self.anchors:
            loss = loss + self.penalty()
        return self.optimize(loss)

    def rebase(self) -> None:
        """Move every stored anchor to the current weights (HTCL: w_1); the Fishers are kept."""
        self.anchors = [[p.detach().clone() for p in trainable(self.model)] for _ in self.anchors]

    def end_task(self, task_id: int, train_loader) -> None:
        """Store w_t* and the per-sample empirical Fisher of this task's training data (BN recalibrated)."""
        probe = copy.deepcopy(self.model)
        recalibrate_bn(probe, train_loader.dataset, self.device, train_loader.batch_size)
        self.fishers.append(fisher_diag(probe, train_loader.dataset, self.device, self.fisher_samples))
        self.anchors.append([p.detach().clone() for p in trainable(self.model)])

    def state_dict(self) -> dict:
        """Model, anchors and Fisher diagonals."""
        return {**super().state_dict(), "anchors": self.anchors, "fishers": self.fishers}

    def load_state_dict(self, state: dict) -> None:
        """Restore model, anchors and Fisher diagonals (moved to the device)."""
        super().load_state_dict(state)
        self.anchors = [[a.to(self.device) for a in task] for task in state["anchors"]]
        self.fishers = [[f.to(self.device) for f in task] for task in state["fishers"]]
