"""Check of DER / DER++: stored logits, the loss formula and the independent second buffer draw."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from htcl.methods.der import DER


def test_der_loss_and_buffer() -> None:
    """With lr = 0 the weights are fixed, so observe's loss must equal the hand-computed formula."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    cfg = {"lr": 0.0, "buffer_size": 8, "alpha": 0.3, "beta": 0.7}
    method = DER(model, cfg, torch.device("cpu"))
    x0, y0 = torch.randn(8, 4), torch.randint(0, 3, (8,))
    method.begin_task(0, None)
    method.observe(x0, y0, 0)  # empty buffer: plain CE, then the batch is stored with its logits
    assert torch.allclose(method.buffer.logits, model(x0).detach())

    x, y = torch.randn(5, 4), torch.randint(0, 3, (5,))
    stored = method.buffer.logits.clone()  # observe's reservoir insert may overwrite slots afterwards
    rng = torch.get_rng_state()
    loss = method.observe(x, y, 0)
    torch.set_rng_state(rng)
    idx1, idx2 = torch.randperm(8)[:5], torch.randperm(8)[:5]  # two independent draws
    with torch.no_grad():
        expected = (F.cross_entropy(model(x), y)
                    + 0.3 * F.mse_loss(model(x0[idx1]), stored[idx1])
                    + 0.7 * F.cross_entropy(model(x0[idx2]), y0[idx2]))
    assert abs(loss - float(expected)) < 1e-5


if __name__ == "__main__":
    test_der_loss_and_buffer()
    print("ok")
