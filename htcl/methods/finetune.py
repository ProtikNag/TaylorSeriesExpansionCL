"""Fine-tuning: plain SGD on the current task (lower reference)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method


class Finetune(Method):
    """Cross-entropy on the current batch only."""

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """One SGD step on the current batch."""
        return self.optimize(F.cross_entropy(self.model(x), y))
