"""Method interface shared by all continual-learning baselines.

Contract (see docs/DESIGN.md):
- The engine calls `model.train()` before `begin_task`, then `observe` per batch for every epoch,
  then `end_task`. A fresh optimizer is created in `begin_task`.
- `observe` must route its loss through `self.optimize(loss)`, which adds the FedProx term when
  HTCL's fedprox aggregator sets `self.prox`.
- `predict(x)` must depend only on the current `self.model` weights and the method state:
  HTCL loads the deployed weights into `self.model` and calls `prepare_eval()` before evaluating.
- `state_dict()` must cover everything needed to resume at a task boundary (model, buffers,
  anchors, ...); optimizers are recreated per task and are not saved.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Method:
    """Base class: plain cross-entropy SGD (subclasses override `observe` and the state hooks)."""

    DEFAULTS: dict = {}  # method hyperparameters; subclasses list theirs here

    @classmethod
    def defaults(cls, dataset: str | None) -> dict:
        """Hyperparameter defaults for a dataset. `run.resolve_method` writes them into the job spec,
        so they enter the job id and job.json."""
        return dict(cls.DEFAULTS)

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device):
        """cfg holds dataset hyperparameters (lr, momentum, weight_decay, batch_size, buffer_size,
        num_outputs, scenario, num_tasks) merged with the method spec (alpha, beta, lam, ...).
        Missing method keys fall back to `defaults(cfg["name"])` (cfg["name"] is the dataset name)."""
        self.model = model.to(device)
        self.cfg = {**self.defaults(cfg.get("name")), **cfg}
        self.device = device
        self.optimizer: torch.optim.Optimizer | None = None
        self.prox: tuple[float, list[torch.Tensor]] | None = None  # (mu, anchor params), FedProx only

    def make_optimizer(self) -> torch.optim.Optimizer:
        """SGD with the dataset's lr, momentum 0.9 and weight decay 1e-4 (unless overridden)."""
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"], momentum=self.cfg.get("momentum", 0.9),
                               weight_decay=self.cfg.get("weight_decay", 1e-4))

    def begin_task(self, task_id: int, train_loader) -> None:
        """Called before the first epoch of a task: fresh optimizer."""
        self.optimizer = self.make_optimizer()

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One training step on a batch; returns the loss value."""
        raise NotImplementedError

    def end_task(self, task_id: int, train_loader) -> None:
        """Called after the last epoch of a task."""

    def rebase(self) -> None:
        """Re-anchor weight-anchored state (e.g. EWC's w*) on the current `self.model`; HTCL calls it after
        resetting the local model to w_1. No-op by default."""

    def prepare_eval(self) -> None:
        """Called before evaluation with the current weights (e.g. recompute class means)."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (or scores) over the output space."""
        return self.model(x)

    def optimize(self, loss: torch.Tensor) -> float:
        """Backward and SGD step, adding the FedProx proximal term when set."""
        if self.prox is not None:
            from .federated import fedprox_penalty
            mu, anchor = self.prox
            loss = loss + fedprox_penalty(self.model, anchor, mu)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def state_dict(self) -> dict:
        """Model weights; subclasses add their own state."""
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore from `state_dict` output."""
        self.model.load_state_dict(state["model"])
