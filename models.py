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


def get_model(num_classes=10, freeze_backbone=False):
    model = SimpleResNet(num_classes=num_classes)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model
