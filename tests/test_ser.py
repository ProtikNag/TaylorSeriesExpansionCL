"""Checks for SER: first-task gating, stored logits, frozen task-start model, state round trip."""

from __future__ import annotations

import torch
import torch.nn as nn

from htcl.methods.ser import SER


def test_ser_terms_and_state() -> None:
    """Task 1 is plain CE and fills the buffer with logits; task 2 freezes theta_{t-1}; state round-trips."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))
    cfg = {"name": "splitmnist", "lr": 0.1, "buffer_size": 6}
    ser = SER(model, cfg, torch.device("cpu"))
    assert (ser.alpha, ser.beta, ser.gamma) == (0.2, 1.0, 0.2)
    x, y = torch.randn(4, 4), torch.randint(0, 2, (4,))

    ser.begin_task(0, None)
    assert ser.old_model is None
    ser.observe(x, y, 0)
    assert len(ser.buffer) == 4 and ser.buffer.logits is not None and ser.buffer.logits.abs().sum() > 0
    ser.end_task(0, None)

    ser.begin_task(1, None)
    assert ser.old_model is not None and not any(p.requires_grad for p in ser.old_model.parameters())
    with torch.no_grad():  # forward consistency is exactly zero at the task start (both in train mode)
        assert torch.allclose(ser.old_model(x), ser.model(x))
    before = [p.clone() for p in ser.old_model.parameters()]
    ser.observe(x, y, 1)
    assert all(torch.equal(a, b) for a, b in zip(before, ser.old_model.parameters()))  # frozen
    ser.end_task(1, None)

    clone = SER(nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2)), cfg,
                torch.device("cpu"))
    clone.load_state_dict(ser.state_dict())
    assert clone.tasks_seen == 2 and torch.equal(clone.buffer.logits, ser.buffer.logits)
    assert SER(model, {**cfg, "name": "cifar100"}, torch.device("cpu")).alpha == 0.5


def test_forward_term_zero_at_task_start() -> None:
    """At the first step of task 2 the forward term is 0: the frozen BN model sees the same joint batch,
    and a dropout model's frozen copy is in eval mode (deterministic target)."""
    import torch.nn.functional as F
    for model in (nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2)),
                  nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))):
        torch.manual_seed(0)
        cfg = {"name": "splitmnist", "lr": 0.1, "buffer_size": 6, "alpha": 0.0, "beta": 0.0, "gamma": 1.0}
        ser = SER(model, cfg, torch.device("cpu"))
        x, y = torch.randn(4, 4), torch.randint(0, 2, (4,))
        ser.model.train()
        ser.begin_task(0, None)
        ser.observe(x, y, 0)
        ser.end_task(0, None)
        ser.begin_task(1, None)
        x = torch.randn(4, 4) + 2.0  # task 2 data differs from the buffer, so joint-batch BN stats differ from x's
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
        assert ser.old_model.training == has_bn
        if has_bn:  # loss = CE on the joint batch's current rows + gamma * forward term
            with torch.no_grad():
                ce = F.cross_entropy(ser.model(torch.cat([x, ser.buffer.x[:len(ser.buffer)]]))[:4], y)
            assert abs(ser.observe(x, y, 1) - float(ce)) < 1e-6
        else:  # the target is deterministic and equals the current model's eval-mode output
            with torch.no_grad():
                assert torch.equal(ser.old_model(x), ser.old_model(x))
                assert torch.allclose(ser.old_model(x), ser.model.eval()(x))


if __name__ == "__main__":
    test_ser_terms_and_state()
    test_forward_term_zero_at_task_start()
    print("ok")
