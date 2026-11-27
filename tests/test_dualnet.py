"""Runnable checks for DualNet: identity init, Barlow Twins loss, augmentation, both learners train."""

import torch

from htcl.methods.dualnet import DualNet, augment, barlow_twins_loss
from htcl.models import GCN, MnistResNet, TextMLP


def test_dualnet() -> None:
    """Fast learner starts as identity; BT loss is ~0 for identical whitened views; both learners move."""
    torch.manual_seed(0)
    base = TextMLP(input_dim=30, num_outputs=3)
    method = DualNet(base, {"lr": 0.1, "buffer_size": 20}, torch.device("cpu"))
    x, y = torch.randn(8, 30), torch.randint(0, 3, (8,))
    method.model.eval()
    assert torch.allclose(method.model(x), base(x)), "fast learner must start as the identity"
    z = torch.randn(256, 4)
    assert barlow_twins_loss(z, z, 5e-3) < 0.5, "identical views: C ~ I"
    assert augment(torch.randn(4, 1, 28, 28), 0.2).shape == (4, 1, 28, 28)
    assert torch.equal(augment(torch.arange(5), 0.2), torch.arange(5))

    method.model.train()
    method.begin_task(0, None)
    method.observe(x, y, 0)  # fills the memory (no SSL yet: memory empty)
    slow_before = method.model.slow.body[0].weight.detach().clone()
    method.ssl_step(8)
    assert not torch.equal(slow_before, method.model.slow.body[0].weight), "SSL must update the slow learner"
    fast_before = method.model.fast.weight.detach().clone()
    method.observe(x, y, 0)
    assert not torch.equal(fast_before, method.model.fast.weight), "supervised step must update the fast learner"
    head = {id(p) for p in method.model.head.parameters()}
    assert all(id(p) not in head for g in method.ssl_optimizer.param_groups for p in g["params"])
    assert all(g["weight_decay"] == 0 for g in method.ssl_optimizer.param_groups)  # decay applied once

    image_method = DualNet(MnistResNet(2), {"lr": 0.01, "buffer_size": 10}, torch.device("cpu"))
    image_method.begin_task(0, None)
    xi, yi = torch.randn(4, 1, 28, 28), torch.randint(0, 2, (4,))
    image_method.observe(xi, yi, 0)
    image_method.observe(xi, yi, 0)  # SSL on a 4-sample memory batch through the ResNet

    graph = (torch.randn(12, 5), torch.randint(0, 12, (2, 30)))
    graph_method = DualNet(GCN(graph, 3), {"lr": 0.01, "buffer_size": 10}, torch.device("cpu"))
    graph_method.begin_task(0, None)
    nodes, labels = torch.arange(6), torch.randint(0, 3, (6,))
    graph_method.observe(nodes, labels, 0)
    graph_method.observe(nodes, labels, 0)  # node-index memory, views from the GCN's dropout


if __name__ == "__main__":
    test_dualnet()
    print("ok")
