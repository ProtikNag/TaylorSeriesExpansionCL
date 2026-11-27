"""Strong Experience Replay (Zhuo et al., "Continual Learning with Strong Experience Replay", ICCV 2023;
arXiv:2305.13622, Eq. 8 and Algorithm 1).

For task t > 1, with x^t the current batch, (x^m, y^m, z) a buffer batch with stored logits z and
theta_{t-1} the frozen model from the start of the task:

    L = CE(f(x^t), y) + beta * CE(f(x^m), y^m) + alpha * ||f(x^m) - z||^2 + gamma * ||f(x^t) - f(x^t; theta_{t-1})||^2

i.e. classification on D_t and M, backward consistency (DER) on M and forward consistency on D_t.
The paper's Eq. 8 has weight 1 on the buffer CE (our beta) and calls the forward weight beta (our
gamma). On the first task only CE(f(x^t), y) is used (Algorithm 1, line 8), but the buffer is filled
with (x, y, f(x)) from the first batch on.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..buffer import ReservoirBuffer
from .base import Method

# Paper Sec. 4.2 balancing factors (alpha, paper-beta = our gamma) per dataset. CIFAR-100 uses
# (0.5, 0.5); CIFAR-10 and P-/R-MNIST use (0.2, 0.2), which we take for every other dataset.
PAPER_ALPHA_GAMMA = {"cifar100": (0.5, 0.5)}
DEFAULT_ALPHA_GAMMA = (0.2, 0.2)


class SER(Method):
    """DER++-style replay with stored logits plus forward consistency to the task-start model."""

    @classmethod
    def defaults(cls, dataset: str | None) -> dict:
        """Paper Sec. 4.2 alpha/gamma for the dataset; beta (buffer CE weight) is 1 in Eq. 8."""
        alpha, gamma = PAPER_ALPHA_GAMMA.get(dataset, DEFAULT_ALPHA_GAMMA)
        return {"alpha": alpha, "beta": 1.0, "gamma": gamma}

    def __init__(self, model, cfg: dict, device: torch.device):
        """Reservoir of `buffer_size` samples with logits; alpha/beta/gamma from cfg or paper defaults."""
        super().__init__(model, cfg, device)
        self.buffer = ReservoirBuffer(cfg["buffer_size"])
        self.alpha, self.beta, self.gamma = (float(self.cfg[k]) for k in ("alpha", "beta", "gamma"))
        self.tasks_seen = 0
        self.old_model: torch.nn.Module | None = None

    def begin_task(self, task_id: int, train_loader) -> None:
        """Fresh optimizer; from the second task on, freeze a copy of the current model as theta_{t-1}."""
        super().begin_task(task_id, train_loader)
        self.old_model = None
        if self.tasks_seen > 0:
            # With BatchNorm the copy stays in train mode, so f(x; theta_{t-1}) uses the statistics of the
            # same joint batch as f(x; theta_t) (see observe) and the forward term is exactly 0 at the
            # start of the task; its running stats are never used. Without BatchNorm (TextMLP, GCN, ViT)
            # it goes to eval mode, so dropout does not turn the target into a random draw.
            self.old_model = copy.deepcopy(self.model).requires_grad_(False)
            self.old_model.train(any(isinstance(m, _BatchNorm) for m in self.old_model.modules()))

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One SGD step on Eq. 8, then offer (x, y, pre-step logits) to the reservoir."""
        n = x.shape[0]
        replay = self.tasks_seen > 0 and len(self.buffer) > 0
        if replay:
            buf_x, buf_y, _, buf_z = self.buffer.sample(n, self.device)
            outputs = self.model(torch.cat([x, buf_x]))
            out_cur, out_buf = outputs[:n], outputs[n:]
        else:
            out_cur = self.model(x)
        loss = F.cross_entropy(out_cur, y)
        if replay:
            loss = loss + self.beta * F.cross_entropy(out_buf, buf_y) + self.alpha * F.mse_loss(out_buf, buf_z)
        if self.old_model is not None:
            with torch.no_grad():  # same (joint) batch as out_cur, so BatchNorm normalises identically
                old_out = self.old_model(torch.cat([x, buf_x]))[:n] if replay else self.old_model(x)
            loss = loss + self.gamma * F.mse_loss(out_cur, old_out)
        value = self.optimize(loss)
        self.buffer.add(x, y, task_id, logits=out_cur.detach())  # DER convention: logits before the step
        return value

    def end_task(self, task_id: int, train_loader) -> None:
        """Count the task and drop the frozen copy (it is rebuilt in the next begin_task)."""
        self.tasks_seen += 1
        self.old_model = None

    def state_dict(self) -> dict:
        """Model, buffer and task counter (the frozen copy is rebuilt from the model at begin_task)."""
        return {**super().state_dict(), "buffer": self.buffer.state_dict(), "tasks_seen": self.tasks_seen}

    def load_state_dict(self, state: dict) -> None:
        """Restore model, buffer and task counter."""
        super().load_state_dict(state)
        self.buffer.load_state_dict(state["buffer"])
        self.tasks_seen = int(state["tasks_seen"])
