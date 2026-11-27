"""Spectral regularization (Lewandowski et al., 2024, "Learning Continually by Spectral Regularization").

Every step minimises  CE + lam * R(theta)  with (paper Sec. 4 and App. A.7)

    R(theta) = sum_l (sigma_1(W_l)^k - 1)^2  +  sum_l ||b_l||_2^(2k)  +  sum_norm sum_i (gamma_i^k - 1)^2

- W_l: every module's 2-D+ `weight` (Linear, Conv, PyG Linear inside GCNConv, the head included);
  conv kernels are reshaped to (out, in * kh * kw) (paper App. A.7, after Yoshida and Miyato 2017).
- b_l: every module's `bias` (layer biases and normalization shifts) is pulled toward zero through
  its spectral norm, which for a vector is the L2 norm (paper Sec. 4).
- gamma: 1-D `weight`s (BatchNorm/LayerNorm/GroupNorm scales) are diagonal matrices; the paper
  regularizes each entry toward 1 rather than the non-differentiable max (App. A.7).
- Parameters that are none of these (ViT cls_token, pos_embed) are not regularized.

sigma_1 comes from `power_iters` power-iteration rounds per step (paper: one suffices) with the left
singular vector u kept across steps in the method state, as in spectral normalization (Miyato et al.,
2018): v = W^T u / ||W^T u||, u = W v / ||W v||, sigma = u^T W v with u and v detached, so the
gradient of sigma is u v^T.

Defaults: lam 1e-4 (paper App. B: best of {1e-2, 1e-3, 1e-4} on CIFAR-10/100 and tiny-ImageNet),
k 2 (paper Sec. 4). The optimizer is the protocol's SGD with weight decay 1e-4 (docs/DESIGN.md); the
paper used Adam without weight decay.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method

DEFAULT_LAM = 1e-4
DEFAULT_K = 2
DEFAULT_POWER_ITERS = 1


class SR(Method):
    """Cross-entropy plus the spectral regularizer (cfg keys: lam, k, power_iters)."""

    DEFAULTS = {"lam": DEFAULT_LAM, "k": DEFAULT_K, "power_iters": DEFAULT_POWER_ITERS}

    def __init__(self, model, cfg: dict, device: torch.device):
        """Create the method and one random unit vector u per weight matrix (seeded by the job seed)."""
        super().__init__(model, cfg, device)
        self.lam = float(self.cfg["lam"])
        self.k = int(self.cfg["k"])
        self.power_iters = int(self.cfg["power_iters"])
        self.u: dict[str, torch.Tensor] = {
            name: F.normalize(torch.randn(p.shape[0], device=device), dim=0)
            for name, p in self.model.named_parameters() if name.rsplit(".", 1)[-1] == "weight" and p.ndim >= 2}

    def spectral_norm(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """sigma_1 of the weight reshaped to (out, -1), differentiable in the weight; updates u[name]."""
        matrix = weight.reshape(weight.shape[0], -1)
        with torch.no_grad():
            u = self.u[name]
            for _ in range(self.power_iters):
                v = F.normalize(matrix.t() @ u, dim=0)
                u = F.normalize(matrix @ v, dim=0)
            self.u[name] = u  # rebinding (not in place) keeps HTCL/R2 state snapshots independent
        return u @ matrix @ v

    def regularizer(self) -> torch.Tensor:
        """R(theta) over all modules' direct `weight` and `bias` parameters (see module docstring)."""
        total = torch.zeros((), device=self.device)
        for module_name, module in self.model.named_modules():
            for param_name, p in module.named_parameters(recurse=False):
                if param_name == "bias":
                    total = total + p.norm().pow(2 * self.k)
                elif param_name == "weight" and p.ndim >= 2:
                    name = f"{module_name}.weight" if module_name else "weight"
                    total = total + (self.spectral_norm(name, p).pow(self.k) - 1).pow(2)
                elif param_name == "weight":
                    total = total + (p.pow(self.k) - 1).pow(2).sum()
        return total

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One SGD step on CE + lam * R."""
        return self.optimize(F.cross_entropy(self.model(x), y) + self.lam * self.regularizer())

    def state_dict(self) -> dict:
        """Model and the persistent power-iteration vectors."""
        return {**super().state_dict(), "u": dict(self.u)}

    def load_state_dict(self, state: dict) -> None:
        """Restore model and power-iteration vectors (moved to the device)."""
        super().load_state_dict(state)
        self.u = {name: u.to(self.device) for name, u in state["u"].items()}
