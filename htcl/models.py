"""Architectures from the paper's table. Every model has `.features(x)` and a linear `.head`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .data import CLData


class _Headed(nn.Module):
    """forward(x) = head(features(x))."""

    head: nn.Linear

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits."""
        return self.head(self.features(x))


def _resnet18_trunk(cifar_stem: bool) -> nn.Module:
    """torchvision ResNet-18 without fc; optional 3x3 stride-1 stem and no maxpool."""
    net = torchvision.models.resnet18(weights=None)
    if cifar_stem:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


class MnistResNet(_Headed):
    """Conv(1->64) -> MaxPool -> Conv(64->128) -> MaxPool -> 1x1 proj(128->3) -> ResNet-18 -> Linear(512->C)."""

    def __init__(self, num_outputs: int):
        """Build the network with a C-way head."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.proj = nn.Conv2d(128, 3, 1)
        self.trunk = _resnet18_trunk(cifar_stem=False)
        self.head = nn.Linear(512, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """512-d ResNet features."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.trunk(self.proj(x))


class CifarResNet(_Headed):
    """ResNet-18 with a 3x3 stride-1 stem, no maxpool -> Linear(512->C)."""

    def __init__(self, num_outputs: int):
        """Build the network with a C-way head."""
        super().__init__()
        self.trunk = _resnet18_trunk(cifar_stem=True)
        self.head = nn.Linear(512, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """512-d ResNet features."""
        return self.trunk(x)


class TextMLP(_Headed):
    """Linear(D->1024) -> ReLU -> Drop -> Linear(1024->512) -> ReLU -> Drop -> Linear(512->C)."""

    def __init__(self, input_dim: int, num_outputs: int, dropout: float = 0.5):
        """Build the MLP."""
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(512, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """512-d hidden features."""
        return self.body(x)


class SmallMLP(_Headed):
    """784 -> 100 -> 100 -> C MLP for the MNIST Class-IL loss-landscape experiment."""

    def __init__(self, num_outputs: int, hidden: int = 100):
        """Build the MLP."""
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden features."""
        return self.body(x)


class GCN(_Headed):
    """GCNConv(F->128) -> ReLU -> Drop .5 -> GCNConv(128->128) -> ReLU -> Drop .5 -> Linear(128->C).

    The full graph lives in non-persistent buffers; inputs are node indices, so the GCN always
    propagates over the whole graph and the batch selects rows.
    """

    def __init__(self, graph: tuple[torch.Tensor, torch.Tensor], num_outputs: int, hidden: int = 128,
                 dropout: float = 0.5):
        """Build the GCN for a fixed graph (node features, edge_index)."""
        super().__init__()
        from torch_geometric.nn import GCNConv
        node_x, edge_index = graph
        self.register_buffer("node_x", node_x, persistent=False)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.conv1 = GCNConv(node_x.shape[1], hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Linear(hidden, num_outputs)

    def features(self, idx: torch.Tensor) -> torch.Tensor:
        """128-d node embeddings of the requested nodes."""
        h = F.dropout(F.relu(self.conv1(self.node_x, self.edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, self.edge_index)), self.dropout, self.training)
        return h[idx]


class ViTSmall(_Headed):
    """timm ViT-S/16 (pretrained) -> Linear(384->C)."""

    def __init__(self, num_outputs: int, pretrained: bool = True):
        """Load the backbone (weights from the HF hub cache) and add a C-way head."""
        super().__init__()
        import timm
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """384-d CLS features."""
        return self.backbone(x)


def build_model(data: CLData, pretrained: bool = True) -> _Headed:
    """Model for a dataset, with a head of `data.num_outputs` units."""
    n = data.num_outputs
    if data.name == "splitmnist":
        return MnistResNet(n)
    if data.name == "mnist_classil3":
        return SmallMLP(n)
    if data.name == "cifar100":
        return CifarResNet(n)
    if data.name == "news20":
        return TextMLP(data.input_dim, n)
    if data.name == "cora":
        return GCN(data.graph, n)
    if data.name == "imagenet100":
        return ViTSmall(n, pretrained)
    raise ValueError(f"no model for {data.name}")
