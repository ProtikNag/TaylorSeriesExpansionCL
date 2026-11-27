"""Sample-level reservoir replay buffer (Vitter's algorithm R)."""

from __future__ import annotations

import torch


class ReservoirBuffer:
    """Fixed-capacity reservoir of (x, y, task_id, logits?) kept on the CPU.

    Every sample of the stream ends up in the buffer with probability capacity / seen.
    Randomness comes from torch's global generator, so it is restored with the job's RNG state.
    """

    def __init__(self, capacity: int):
        """Create an empty buffer holding at most `capacity` samples."""
        self.capacity = int(capacity)
        self.seen = 0
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.task: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None

    def __len__(self) -> int:
        """Number of stored samples."""
        return min(self.seen, self.capacity)

    def _allocate(self, x: torch.Tensor, logits: torch.Tensor | None) -> None:
        """Lazily allocate storage shaped after the first batch."""
        self.x = torch.zeros((self.capacity, *x.shape[1:]), dtype=x.dtype)
        self.y = torch.zeros(self.capacity, dtype=torch.long)
        self.task = torch.zeros(self.capacity, dtype=torch.long)
        if logits is not None:
            self.logits = torch.zeros((self.capacity, *logits.shape[1:]), dtype=logits.dtype)

    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int, logits: torch.Tensor | None = None) -> None:
        """Offer a batch to the reservoir, sample by sample."""
        if self.capacity <= 0:
            return
        x, y = x.detach().cpu(), y.detach().cpu()
        logits = logits.detach().cpu() if logits is not None else None
        if self.x is None:
            self._allocate(x, logits)
        for i in range(x.shape[0]):
            if self.seen < self.capacity:
                slot = self.seen
            else:
                slot = int(torch.randint(0, self.seen + 1, (1,)))
            self.seen += 1
            if slot < self.capacity:
                self.x[slot], self.y[slot], self.task[slot] = x[i], y[i], task_id
                if self.logits is not None and logits is not None:
                    self.logits[slot] = logits[i]

    def sample(self, n: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
        """Draw min(n, len) samples without replacement: (x, y, task, logits-or-None) on `device`."""
        if len(self) == 0:  # storage is allocated by the first add; callers check len(buffer) first
            raise ValueError("sample() on an empty ReservoirBuffer")
        idx = torch.randperm(len(self))[:n]
        logits = self.logits[idx].to(device) if self.logits is not None else None
        return self.x[idx].to(device), self.y[idx].to(device), self.task[idx].to(device), logits

    def state_dict(self) -> dict:
        """Everything needed to restore the buffer."""
        return {"capacity": self.capacity, "seen": self.seen, "x": self.x, "y": self.y,
                "task": self.task, "logits": self.logits}

    def load_state_dict(self, state: dict) -> None:
        """Restore from `state_dict` output."""
        for key, value in state.items():
            setattr(self, key, value)
