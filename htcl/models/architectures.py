"""
Neural network architectures for continual learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class SimpleResNet(nn.Module):
    """ResNet18 backbone adapted for CIFAR-100 resolution."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.backbone = models.resnet18(weights=None)
        # Adapt for CIFAR resolution (32x32)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        return self.backbone(x)


class SmallCNN(nn.Module):
    """Small CNN + ResNet backbone for grayscale images (MNIST)."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Project to 3 channels for ResNet
        self.to3 = nn.Conv2d(128, 3, kernel_size=1)
        
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.to3(x)
        x = self.backbone(x)
        return self.classifier(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.to3(x)
        return self.backbone(x)


class GCNNet(nn.Module):
    """
    Simple 2-layer GCN for node classification with MLP fallback.
    Uses MLP mode when edge_index is not provided.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: Optional[int] = None,
        hidden: int = 128,
        dropout: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.dropout = dropout
        
        # GCN layers (lazy init if in_channels not provided)
        self.conv1 = None
        self.conv2 = None
        if in_channels is not None:
            self._init_convs(in_channels)
        
        self.classifier = nn.Linear(hidden, num_classes)
        self.mlp_fc1 = None  # Lazy init for MLP fallback
    
    def _init_convs(self, in_channels: int):
        """Initialize GCN convolution layers."""
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.hidden)
    
    def _ensure_mlp_fc1(self, in_features: int, device: torch.device):
        """Lazily create MLP layer if needed."""
        if self.mlp_fc1 is None or self.mlp_fc1.in_features != in_features:
            self.mlp_fc1 = nn.Linear(in_features, self.hidden).to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional graph structure.
        
        Args:
            x: Node features [N, F] or batch features [B, F]
            edge_index: Graph edges [2, E] or None for MLP mode
        
        Returns:
            Logits [N, num_classes] or [B, num_classes]
        """
        if edge_index is not None:
            # Graph mode
            if self.conv1 is None:
                self._init_convs(x.size(1))
            
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.classifier(x)
        
        # MLP fallback mode
        self._ensure_mlp_fc1(x.size(1), x.device)
        h = F.relu(self.mlp_fc1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class TextMLP(nn.Module):
    """MLP for TF-IDF/bag-of-words text features."""
    
    def __init__(
        self,
        num_classes: int = 4,
        input_dim: Optional[int] = 20000,
        hidden: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.dropout_rate = dropout
        
        # Lazy init fc1 if input_dim unknown
        self.fc1 = nn.Linear(input_dim, hidden) if input_dim else None
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.classifier = nn.Linear(hidden // 2, num_classes)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Lazy init fc1 to match input features
        if self.fc1 is None or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), self.hidden).to(x.device)
        
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.classifier(h)


# Model registry
MODEL_REGISTRY = {
    "CIFAR100": SimpleResNet,
    "SplitMNIST": SmallCNN,
    "Cora": GCNNet,
    "20Newsgroups": TextMLP,
}

# Default number of classes per task for each dataset
DEFAULT_CLASSES = {
    "CIFAR100": 10,
    "SplitMNIST": 2,
    "Cora": 3,
    "20Newsgroups": 4,
}


def get_model(
    dataset: str,
    num_classes: Optional[int] = None,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        dataset: Name of the dataset (determines architecture)
        num_classes: Number of output classes (uses default if None)
        freeze_backbone: Whether to freeze backbone parameters
        **kwargs: Additional model arguments
    
    Returns:
        Initialized model
    """
    if dataset not in MODEL_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if num_classes is None:
        num_classes = DEFAULT_CLASSES.get(dataset, 10)
    
    model_cls = MODEL_REGISTRY[dataset]
    model = model_cls(num_classes=num_classes, **kwargs)
    
    if freeze_backbone and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    return model
