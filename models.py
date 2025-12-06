# models.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()

        # Use ResNet18 backbone, adjusted for CIFAR resolution
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Shared classifier for all tasks: always predicts 10 classes (0–9)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        # bigger conv channel counts
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)   # in=1, out=64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # in=64, out=128
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dims

        # after two pool ops: 28 -> 14 -> 7 (so spatial = 7x7)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)   # first FC (larger)
        self.fc2 = nn.Linear(256, 128)           # NEW extra hidden layer
        self.fc3 = nn.Linear(128, num_classes)   # output

        # optional regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 64, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B,128, 7, 7]
        x = x.view(x.size(0), -1)              # flatten -> [B, 128*7*7]
        x = F.relu(self.fc1(x))                # -> [B,256]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))                # -> [B,128] (extra hidden)
        x = self.fc3(x)                        # -> [B,num_classes]
        return x


class GCNNet(nn.Module):
    """
    Simple 2-layer GCN for node classification (torch_geometric required).

    - If in_channels is provided at construction, convs are created immediately.
    - If in_channels is None, convs are created lazily on the first forward using x.size(1).
    - forward expects (x, edge_index) and returns logits of shape [N, num_classes].
    """
    def __init__(self, num_classes=2, in_channels=None, hidden=128, dropout=0.5):
        super(GCNNet, self).__init__()
        if not TG_AVAILABLE:
            raise ImportError("torch_geometric is required for GCNNet but not available.")
        self.num_classes = num_classes
        self.hidden = hidden
        self.dropout = dropout

        # conv placeholders (may be created immediately or lazily)
        if in_channels is not None:
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        else:
            self.conv1 = None
            self.conv2 = None

        # final classifier maps hidden -> num_classes
        self.classifier = nn.Linear(hidden, num_classes)

    def _init_convs(self, in_channels):
        """Create convs lazily when input feature dim is known."""
        if self.conv1 is None or self.conv2 is None:
            self.conv1 = GCNConv(in_channels, self.hidden)
            self.conv2 = GCNConv(self.hidden, self.hidden)

    def forward(self, x, edge_index):
        """
        x: Tensor [N, F]
        edge_index: LongTensor [2, E]
        returns logits: [N, num_classes]
        """
        if self.conv1 is None:
            # assume x is [N, F]
            self._init_convs(x.size(1))

        # two GCN layers with ReLU + dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)  # [N, num_classes]
        return logits


class TextMLP(nn.Module):
    """
    Simple MLP for TF-IDF features / bag-of-words style inputs.

    Notes:
    - By default assumes input_dim=20000 (to match the vectorizer used in the dataset helper).
      If the actual input dimension differs, the first Linear layer will be created lazily on the
      first forward pass to match x.shape[1].
    - This design keeps compatibility with existing training loops which call get_model(num_classes, dataset="20Newsgroups")
      and then pass batches of vectors of unknown dimensionality.
    """
    def __init__(self, num_classes=4, input_dim=20000, hidden=1024, dropout=0.5):
        super(TextMLP, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        # create first FC lazily if input_dim is None or unknown
        if input_dim is not None:
            self.fc1 = nn.Linear(input_dim, hidden)
        else:
            self.fc1 = None
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.classifier = nn.Linear(hidden // 2, num_classes)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        # x expected shape: [B, D]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        device = x.device
        in_features = x.size(1)
        if self.fc1 is None or self.fc1.in_features != in_features:
            # lazily create fc1 that matches input features
            self.fc1 = nn.Linear(in_features, self.hidden).to(device)

        h = F.relu(self.fc1(x))
        h = self.dropout_layer(h)
        h = F.relu(self.fc2(h))
        h = self.dropout_layer(h)
        logits = self.classifier(h)
        return logits


def get_model(num_classes=2, freeze_backbone=False, dataset="SplitMNIST"):
    """
    Factory function for models.
    - CIFAR100 -> ResNet with num_classes outputs
    - SplitMNIST -> SmallCNN with num_classes outputs (typically 2)
    - Cora -> GCNNet with num_classes outputs (node-level)
    - 20Newsgroups -> TextMLP with num_classes outputs (document-level)

    NOTE: This function keeps the same signature as before. New datasets are supported by dataset name.
    """
    if dataset == "CIFAR100":
        model = SimpleResNet(num_classes)
    elif dataset == "SplitMNIST":
        model = SmallCNN(num_classes)  # Always 2-way
    elif dataset == "Cora":
        model = GCNNet(num_classes=num_classes, in_channels=None)
    elif dataset == "20Newsgroups":
        # default input_dim=20000 to match dataset vectorizer; TextMLP will adapt automatically if dims differ
        model = TextMLP(num_classes=num_classes, input_dim=20000)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    # Freeze backbone if requested (only applicable for models that expose a .backbone attribute)
    if freeze_backbone and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
