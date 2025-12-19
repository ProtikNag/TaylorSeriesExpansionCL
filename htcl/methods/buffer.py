"""
Replay buffer implementations for continual learning.
"""

import torch
import random
from typing import Optional, Tuple, List
from torch.utils.data import Dataset


class ReplayBuffer:
    """
    Experience replay buffer with reservoir sampling.
    
    Stores samples on CPU to conserve GPU memory.
    Uses reservoir sampling for representative sampling when capacity is reached.
    """
    
    def __init__(self, capacity: int = 500, device: str = "cuda"):
        self.capacity = int(capacity)
        self.device = device
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.seen = 0  # Total samples seen (for reservoir sampling)
    
    def add_sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ):
        """
        Add a sample to the buffer.
        
        Args:
            x: Input tensor
            y: Label tensor
            z: Optional auxiliary tensor (e.g., logits for distillation)
        """
        # Store as CPU tensors
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        z_cpu = z.detach().cpu() if z is not None else torch.zeros(1)
        
        self.seen += 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((x_cpu, y_cpu, z_cpu))
        else:
            # Reservoir sampling
            idx = random.randint(0, self.seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (x_cpu, y_cpu, z_cpu)
    
    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ):
        """Add a batch of samples."""
        batch_size = x.size(0)
        for i in range(batch_size):
            z_i = z[i] if z is not None else None
            self.add_sample(x[i], y[i], z_i)
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch from the buffer.
        
        Returns None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None
        
        k = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, k)
        
        x, y, z = zip(*samples)
        return (
            torch.stack(x).to(self.device),
            torch.stack(y).to(self.device),
            torch.stack(z).to(self.device)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.seen = 0
    
    def get_dataset(self) -> "ReplayDataset":
        """Get buffer contents as a Dataset."""
        return ReplayDataset(self.buffer)


class ReplayDataset(Dataset):
    """Dataset wrapper for replay buffer contents."""
    
    def __init__(self, buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.buffer = buffer
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self.buffer[idx]
        return x, y


class ClassBalancedBuffer(ReplayBuffer):
    """
    Replay buffer that maintains class balance.
    
    Stores equal numbers of samples per class.
    """
    
    def __init__(self, capacity: int = 500, device: str = "cuda"):
        super().__init__(capacity, device)
        self.class_buffers: dict = {}  # class_id -> list of samples
    
    def add_sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ):
        """Add sample with class balancing."""
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        z_cpu = z.detach().cpu() if z is not None else torch.zeros(1)
        
        label = int(y_cpu.item()) if y_cpu.numel() == 1 else int(y_cpu[0].item())
        
        if label not in self.class_buffers:
            self.class_buffers[label] = []
        
        self.class_buffers[label].append((x_cpu, y_cpu, z_cpu))
        
        # Enforce capacity with class balancing
        self._enforce_capacity()
    
    def _enforce_capacity(self):
        """Ensure total samples don't exceed capacity."""
        total = sum(len(buf) for buf in self.class_buffers.values())
        
        while total > self.capacity:
            # Remove from the class with most samples
            max_class = max(self.class_buffers.keys(), 
                          key=lambda k: len(self.class_buffers[k]))
            if self.class_buffers[max_class]:
                # Remove random sample from this class
                idx = random.randint(0, len(self.class_buffers[max_class]) - 1)
                self.class_buffers[max_class].pop(idx)
                total -= 1
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample with class balance."""
        all_samples = []
        for buf in self.class_buffers.values():
            all_samples.extend(buf)
        
        if not all_samples:
            return None
        
        k = min(batch_size, len(all_samples))
        samples = random.sample(all_samples, k)
        
        x, y, z = zip(*samples)
        return (
            torch.stack(x).to(self.device),
            torch.stack(y).to(self.device),
            torch.stack(z).to(self.device)
        )
    
    def __len__(self) -> int:
        return sum(len(buf) for buf in self.class_buffers.values())
