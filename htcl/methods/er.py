"""Experience Replay with a reservoir buffer (Chaudhry et al., 2019)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..buffer import ReservoirBuffer
from .base import Method


class ER(Method):
    """Each step concatenates a buffer batch of the same size as the current batch."""

    def __init__(self, model, cfg: dict, device: torch.device):
        """Create the method and its reservoir of `buffer_size` samples."""
        super().__init__(model, cfg, device)
        self.buffer = ReservoirBuffer(cfg["buffer_size"])

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """CE on current + replayed samples, then offer the current batch to the reservoir."""
        inputs, targets = x, y
        if len(self.buffer) > 0:
            buf_x, buf_y, _, _ = self.buffer.sample(x.shape[0], self.device)
            inputs, targets = torch.cat([x, buf_x]), torch.cat([y, buf_y])
        loss = self.optimize(F.cross_entropy(self.model(inputs), targets))
        self.buffer.add(x, y, task_id)
        return loss

    def state_dict(self) -> dict:
        """Model and buffer."""
        return {**super().state_dict(), "buffer": self.buffer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore model and buffer."""
        super().load_state_dict(state)
        self.buffer.load_state_dict(state["buffer"])
