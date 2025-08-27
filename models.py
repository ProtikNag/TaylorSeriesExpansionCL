# models.py

import torch
import torch.nn as nn
import torchvision.models as models


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


class SimpleResNetMNIST(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNetMNIST, self).__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=2, freeze_backbone=False, dataset="SplitMNIST"):
    """
    Factory function for models.
    - CIFAR100 -> ResNet with 10 outputs
    - SplitMNIST -> ResNet with 2 outputs
    """
    if dataset == "CIFAR100":
        model = SimpleResNet(num_classes)
    elif dataset == "SplitMNIST":
        model = SimpleResNetMNIST(num_classes)  # Always 2-way
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model