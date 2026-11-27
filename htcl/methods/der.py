"""Dark Experience Replay: DER and DER++ (Buzzega et al., NeurIPS 2020).

Loss per step, with (x', z') and (x'', y'') two independent draws from the reservoir:

    L = CE(f(x), y) + alpha * MSE(f(x'), z') + beta * CE(f(x''), y'')

z' are the logits the network produced for x' when it was inserted into the buffer.
beta = 0 is DER (the paper's baseline); beta > 0 is DER++.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..buffer import ReservoirBuffer
from .base import Method


class DER(Method):
    """DER / DER++ with a reservoir buffer of (x, y, logits).

    Defaults alpha = 0.5, beta = 0.0 follow the old code that produced the paper's DER baseline
    (wip/imagenet-vit:htcl/methods/der.py, registered as DER with beta = 0). Both are configurable
    through the method spec (`{name: der, alpha: ..., beta: ...}`).
    """

    DEFAULTS = {"alpha": 0.5, "beta": 0.0}

    def __init__(self, model, cfg: dict, device: torch.device):
        """Create the method and its reservoir of `buffer_size` samples."""
        super().__init__(model, cfg, device)
        self.alpha = float(self.cfg["alpha"])
        self.beta = float(self.cfg["beta"])
        self.buffer = ReservoirBuffer(cfg["buffer_size"])

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One DER(++) step, then offer the batch to the reservoir with its pre-update logits."""
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        if len(self.buffer) > 0:
            buf_x, _, _, buf_logits = self.buffer.sample(x.shape[0], self.device)
            loss = loss + self.alpha * F.mse_loss(self.model(buf_x), buf_logits)
            if self.beta > 0:
                # Independent second draw for the CE term (Buzzega et al., Alg. 1), not a reuse.
                buf_x2, buf_y2, _, _ = self.buffer.sample(x.shape[0], self.device)
                loss = loss + self.beta * F.cross_entropy(self.model(buf_x2), buf_y2)
        value = self.optimize(loss)
        self.buffer.add(x, y, task_id, logits=logits.detach())
        return value

    def state_dict(self) -> dict:
        """Model and buffer (stored logits included)."""
        return {**super().state_dict(), "buffer": self.buffer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore model and buffer."""
        super().load_state_dict(state)
        self.buffer.load_state_dict(state["buffer"])
