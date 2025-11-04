# models.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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


# class SmallCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SmallCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)  # halves spatial dims
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 14, 14]
#         x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 7, 7]
#         x = x.view(x.size(0), -1)             # flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


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


def get_model(num_classes=2, freeze_backbone=False, dataset="SplitMNIST"):
    """
    Factory function for models.
    - CIFAR100 -> ResNet with 10 outputs
    - SplitMNIST -> ResNet with 2 outputs
    """
    if dataset == "CIFAR100":
        model = SimpleResNet(num_classes)
    elif dataset == "SplitMNIST":
        model = SmallCNN(num_classes)  # Always 2-way
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
