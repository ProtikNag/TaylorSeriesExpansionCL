"""iCaRL checks: herding, (task, label)-keyed memory rebalancing, NME scores. `.venv/bin/python -m pytest -q tests/test_icarl.py`."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from htcl.data import TaskDataset
from htcl.methods.icarl import ICaRL, herding


class TinyNet(nn.Module):
    """features = identity on 2-d inputs, linear head."""

    def __init__(self, num_outputs: int):
        """2 -> C head."""
        super().__init__()
        self.head = nn.Linear(2, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Inputs are the features."""
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits."""
        return self.head(self.features(x))


def _loader(points: torch.Tensor, labels: list[int]) -> DataLoader:
    """Loader over a TaskDataset of fixed 2-d points."""
    base = [(p, 0) for p in points]
    return DataLoader(TaskDataset(base, torch.arange(len(points)).numpy(), torch.tensor(labels).numpy()),
                      batch_size=4, shuffle=False)


def test_end_task_herds_on_every_plain_sample() -> None:
    """end_task herds over the task's plain view (`dataset.plain`) and every sample of it, not the
    augmented, shuffled, drop_last training loader."""
    torch.manual_seed(0)
    method = ICaRL(TinyNet(2), {"lr": 0.01, "buffer_size": 20, "num_outputs": 2, "scenario": "domain"},
                   torch.device("cpu"))
    plain_pts = torch.randn(10, 2)
    labels = torch.tensor([0] * 5 + [1] * 5).numpy()
    plain = TaskDataset([(p, 0) for p in plain_pts], torch.arange(10).numpy(), labels)
    train = TaskDataset([(p + 100.0, 0) for p in plain_pts], torch.arange(10).numpy(), labels, plain=plain)
    loader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)  # sees only 8 of 10, augmented
    method.begin_task(0, loader)
    method.end_task(0, loader)
    assert len(method.buffer) == 10  # m = 20 // 2 = 10 per class, capped at the 5 available: all 10 kept
    assert sorted(method.buffer.x.tolist()) == sorted(plain_pts.tolist())


def test_herding_picks_mean_first_and_no_repeats() -> None:
    """The first pick is the row closest to the mean; picks are distinct; m > n is capped."""
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    picked = herding(feats, 5)
    assert picked[0] == 2 and sorted(picked) == [0, 1, 2]


def test_domain_memory_keeps_every_task_and_nme_scores() -> None:
    """Domain scenario: task 1 reuses labels 0/1 but must not overwrite task 0's exemplars; capacity
    is split evenly over the 4 (task, label) classes; predict() max-pools NME scores per label."""
    torch.manual_seed(0)
    method = ICaRL(TinyNet(2), {"lr": 0.01, "buffer_size": 8, "num_outputs": 2, "scenario": "domain"},
                   torch.device("cpu"))
    centers = {(0, 0): [1.0, 0.0], (0, 1): [0.0, 1.0], (1, 0): [-1.0, 0.0], (1, 1): [0.0, -1.0]}
    for task in (0, 1):
        pts = torch.cat([torch.tensor(centers[(task, lab)]) + 0.05 * torch.randn(6, 2) for lab in (0, 1)])
        loader = _loader(pts, [0] * 6 + [1] * 6)
        method.begin_task(task, loader)
        method.end_task(task, loader)
        if task == 0:
            assert len(method.buffer) == 8  # 4 per class
    assert sorted(method.buffer.classes()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(method.buffer) == 8  # rebalanced to 2 per class
    method.model.eval()
    method.prepare_eval()
    probe = torch.tensor([[-1.0, 0.05], [0.05, 1.0]])  # nearest to (1, 0) -> label 0; to (0, 1) -> label 1
    assert method.predict(probe).argmax(1).tolist() == [0, 1]
