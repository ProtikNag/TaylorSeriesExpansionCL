"""DualNet (Pham et al., NeurIPS 2021): a slow learner trained by self-supervision on the episodic
memory and a fast learner that adapts the slow representation with supervision.

Per incoming batch (x, y) (Algorithm 1 of the paper, minus the parts noted below):
 1. Slow learner, `ssl_steps` times: draw a memory batch m, two augmented views a(m), a'(m),
    z_A = f_slow(a(m)), z_B = f_slow(a'(m)), and descend the Barlow Twins loss
        L_BT = sum_i (1 - C_ii)^2 + lambda_BT sum_i sum_{j != i} C_ij^2,
        C_ij = (1/N) sum_n zhat_A[n,i] zhat_B[n,j],  zhat = per-dimension batch standardisation,
    divided by the feature dimension D (see below), with a separate SGD optimizer over the slow
    learner's parameters (head excluded) and no weight decay (the supervised step applies it).
 2. Fast learner: h = f_slow(x) * (1 + gamma) + beta with [gamma, beta] = W h_slow + b (FiLM on the
    slow features, W = b = 0 at init so the module starts as the identity); logits = head(h).
    Cross-entropy on x concatenated with a same-size memory batch, through `self.optimize`, updating
    the fast learner, the head and the slow learner (the supervised gradient also reaches the
    slow learner; our reading of the paper, to be checked against the official code).
 3. Offer (x, y) to the reservoir memory.

The slow learner, the fast learner and the head all live in `self.model` (a `DualNetModel`), so
HTCL consolidates the whole DualNet and `predict` is simply `self.model(x)`.

Augmentations (`augment`):
 - images (N, C, H, W): per-sample random resized crop (side fraction 0.6..1, random position, via
   one affine grid_sample with reflection padding), horizontal flip only for 3-channel images (not
   for digits), contrast x0.6..1.4 and brightness +-0.4 in normalised units;
 - feature vectors (20NG TF-IDF): feature dropout with p = `feature_dropout`;
 - integer inputs (Cora node indices): no input augmentation; the two views differ through the
   GCN's own dropout (p = 0.5) in train mode, as in SimCSE.
All input-augmentation randomness is drawn on the CPU from torch's global generator (restored by
the checkpoint), then moved to the device.

Simplifications against the paper (documented here, not hidden):
 - the fast learner modulates only the final (penultimate) features, not every layer's feature map,
   so one module works for every architecture (ResNet, MLP, GCN, ViT);
 - no projector head: Barlow Twins acts on the penultimate features directly;
 - the supervised loss is plain CE on current + memory samples (no logit-distillation term);
 - one SSL step per incoming batch by default (`ssl_steps`), which keeps compute close to ER;
 - L_BT is divided by D: the raw sum is ~D (512 on the ResNets) at init and, with plain SGD at the
   dataset lr, made the supervised loss diverge in a SplitMNIST check (CE rose to ~40, accuracy at
   chance); scaled by 1/D the CE converges as in ER. Barlow Twins itself uses LARS at large batch.

Defaults: bt_lambda 5e-3 (Zbontar et al. 2021, Barlow Twins); ssl_steps 1, feature_dropout 0.2 and
the augmentation ranges are our choices; lr, momentum, weight decay and buffer come from the dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..buffer import ReservoirBuffer
from .base import Method


class DualNetModel(nn.Module):
    """Slow learner (the dataset model) + fast learner (FiLM on its features) + the model's head."""

    def __init__(self, slow: nn.Module):
        """Wrap a model exposing `.features(x)` and a linear `.head`."""
        super().__init__()
        self.slow = slow
        dim = slow.head.in_features
        self.fast = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.fast.weight)  # identity modulation at init
        nn.init.zeros_(self.fast.bias)

    @property
    def head(self) -> nn.Linear:
        """The classifier (the slow model's head, registered once under `slow.head`)."""
        return self.slow.head

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Fast-adapted features h = f (1 + gamma) + beta, [gamma, beta] = fast(f)."""
        slow_features = self.slow.features(x)
        gamma, beta = self.fast(slow_features).chunk(2, dim=1)
        return slow_features * (1 + gamma) + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of the fast learner."""
        return self.head(self.features(x))


def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor, off_diag_weight: float) -> torch.Tensor:
    """(1/D) [sum_i (1 - C_ii)^2 + w sum_{i != j} C_ij^2] on batch-standardised embeddings (N x D)."""
    z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-5)
    z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-5)
    corr = z_a.T @ z_b / z_a.shape[0]
    diag = torch.diagonal(corr)
    # Divided by D so the loss is O(1) for plain SGD at the dataset lr (the raw sum is ~D at init).
    return ((1 - diag).pow(2).sum() + off_diag_weight * (corr.pow(2).sum() - diag.pow(2).sum())) / z_a.shape[1]


def augment(x: torch.Tensor, feature_dropout: float) -> torch.Tensor:
    """One random view of a batch (see the module docstring for the per-modality choices)."""
    if not x.is_floating_point():  # node indices: views come from the model's dropout
        return x
    n = x.shape[0]
    if x.dim() == 4:
        u = torch.rand(n, 6).to(x.device)
        scale = 0.6 + 0.4 * u[:, 0]
        flip = torch.where(u[:, 1] < 0.5, -1.0, 1.0) if x.shape[1] == 3 else torch.ones_like(scale)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1] = scale * flip, scale
        theta[:, 0, 2] = (1 - scale) * (2 * u[:, 2] - 1)  # crop centre stays inside the image
        theta[:, 1, 2] = (1 - scale) * (2 * u[:, 3] - 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        out = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
        contrast = (0.6 + 0.8 * u[:, 4]).view(n, 1, 1, 1)
        brightness = (0.8 * u[:, 5] - 0.4).view(n, 1, 1, 1)
        return out * contrast + brightness
    keep = (torch.rand(x.shape) >= feature_dropout).to(x.device, x.dtype)
    return x * keep / (1 - feature_dropout)


class DualNet(Method):
    """DualNet with a reservoir memory (see module docstring)."""

    DEFAULTS = {"ssl_steps": 1, "bt_lambda": 5e-3, "feature_dropout": 0.2}

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device):
        """Wrap the dataset model as the slow learner and create the memory."""
        super().__init__(DualNetModel(model), cfg, device)
        self.buffer = ReservoirBuffer(cfg["buffer_size"])
        self.ssl_steps = int(self.cfg["ssl_steps"])
        self.bt_lambda = float(self.cfg["bt_lambda"])
        self.feature_dropout = float(self.cfg["feature_dropout"])
        self.ssl_optimizer: torch.optim.Optimizer | None = None

    def begin_task(self, task_id: int, train_loader) -> None:
        """Fresh supervised optimizer (all parameters) and SSL optimizer (slow learner, no head)."""
        super().begin_task(task_id, train_loader)
        head = {id(p) for p in self.model.head.parameters()}
        slow = [p for p in self.model.slow.parameters() if id(p) not in head]
        # No weight decay here: the supervised optimizer already decays these parameters (protocol 1e-4 once).
        self.ssl_optimizer = torch.optim.SGD(slow, lr=self.cfg["lr"], momentum=self.cfg.get("momentum", 0.9),
                                             weight_decay=0.0)

    def ssl_step(self, n: int) -> None:
        """One Barlow Twins step of the slow learner on a memory batch of up to n samples."""
        if min(n, len(self.buffer)) < 2:  # batch statistics need at least two samples
            return
        mem_x = self.buffer.sample(n, self.device)[0]
        z_a = self.model.slow.features(augment(mem_x, self.feature_dropout))
        z_b = self.model.slow.features(augment(mem_x, self.feature_dropout))
        loss = barlow_twins_loss(z_a, z_b, self.bt_lambda)
        if self.prox is not None:  # FedProx term on local training, as in Method.optimize
            from .federated import fedprox_penalty
            mu, anchor = self.prox
            loss = loss + fedprox_penalty(self.model, anchor, mu)
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        self.ssl_optimizer.step()

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """SSL steps on memory, supervised step on current + memory, then update the memory."""
        for _ in range(self.ssl_steps):
            self.ssl_step(x.shape[0])
        inputs, targets = x, y
        if len(self.buffer) > 0:
            mem_x, mem_y, _, _ = self.buffer.sample(x.shape[0], self.device)
            inputs, targets = torch.cat([x, mem_x]), torch.cat([y, mem_y])
        loss = self.optimize(F.cross_entropy(self.model(inputs), targets))
        self.buffer.add(x, y, task_id)
        return loss

    def state_dict(self) -> dict:
        """Model (slow + fast + head) and memory."""
        return {**super().state_dict(), "buffer": self.buffer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore model and memory."""
        super().load_state_dict(state)
        self.buffer.load_state_dict(state["buffer"])
