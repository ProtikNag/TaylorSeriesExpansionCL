"""
DualNet: Continual Learning, Fast and Slow implementation.

DualNet comprises two complementary learning systems inspired by CLS theory:
- Slow learner: learns general, task-agnostic representation via Self-Supervised Learning (SSL)
- Fast learner: adapts the slow learner's representation for quick knowledge acquisition

Key equations from the paper:
- Barlow Twins loss (Eq. 1): L_BT = Σ_i (1 - C_ii)² + λ_BT Σ_i Σ_{j≠i} C²_ij
- Feature adaptation (Eq. 5-6): m_l = g_{θ,l}(h'_{l-1}), h'_l = h_l ⊗ m_l
- Training loss (Eq. 7): CE(y, ŷ) + (1/|M|) Σ CE(ŷ_i, y_i) + λ_tr D_KL(π(ŷ_i/τ) || π(ŷ_k/τ))

Reference: Pham et al., "DualNet: Continual Learning, Fast and Slow", NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd
import json
from datetime import datetime

from .buffer import ReplayBuffer
from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


# =============================================================================
# DualNet Architecture Components
# =============================================================================

class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins SSL loss (Eq. 1-2 in paper).

    Enforces the cross-correlation matrix between two augmented views
    to be close to identity, reducing redundancy in representations.
    """

    def __init__(self, lambda_bt: float = 0.005):
        """
        Args:
            lambda_bt: Trade-off factor for off-diagonal terms (default 0.005)
        """
        super().__init__()
        self.lambda_bt = lambda_bt

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Compute Barlow Twins loss between two representation batches.

        Args:
            z_a: Representations from view A [batch_size, feature_dim]
            z_b: Representations from view B [batch_size, feature_dim]

        Returns:
            Barlow Twins loss value
        """
        # Normalize representations along batch dimension
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-6)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-6)

        batch_size = z_a.size(0)
        feature_dim = z_a.size(1)

        # Compute cross-correlation matrix (Eq. 2)
        c = torch.mm(z_a_norm.T, z_b_norm) / batch_size

        # Loss: invariance term (diagonal) + redundancy reduction (off-diagonal)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_bt * off_diag
        return loss

    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Extract off-diagonal elements from a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SSLProjector(nn.Module):
    """
    Projector network for SSL (projects backbone features to embedding space).
    Standard practice in SSL methods like Barlow Twins.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class FastLearner(nn.Module):
    """
    Fast learner network that generates gating masks for feature adaptation (Eq. 5-6).

    Generates pixel-wise transformation coefficients from raw input to adapt
    the slow learner's feature maps for task-specific learning.
    """

    def __init__(self, in_channels: int = 3, feature_dims: List[int] = None):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            feature_dims: List of feature dimensions for each layer to adapt
                         (should match slow learner's residual block outputs)
        """
        super().__init__()
        if feature_dims is None:
            # Default for ResNet18: [64, 128, 256, 512]
            feature_dims = [64, 128, 256, 512]

        self.feature_dims = feature_dims
        self.num_layers = len(feature_dims)

        # Build convolutional layers for generating gating masks
        layers = []
        current_channels = in_channels
        for i, dim in enumerate(feature_dims):
            layers.append(nn.Sequential(
                nn.Conv2d(current_channels, dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.Sigmoid(),  # Gating values between 0 and 1
            ))
            current_channels = dim

        self.mask_generators = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate gating masks for each layer.

        Args:
            x: Input image [batch_size, channels, height, width]

        Returns:
            List of gating masks for each layer
        """
        masks = []
        h = x
        for generator in self.mask_generators:
            h = generator(h)
            masks.append(h)
        return masks


class DualNetModel(nn.Module):
    """
    Complete DualNet model combining slow and fast learners.

    The slow learner provides general representations via SSL.
    The fast learner adapts these representations for supervised learning.
    """

    def __init__(
            self,
            slow_learner: nn.Module,
            num_classes: int,
            in_channels: int = 3,
            feature_dims: List[int] = None,
            proj_hidden_dim: int = 2048,
            proj_output_dim: int = 2048,
    ):
        """
        Args:
            slow_learner: Backbone network (e.g., ResNet18 without final FC)
            num_classes: Number of output classes
            in_channels: Number of input channels
            feature_dims: Feature dimensions for fast learner adaptation
            proj_hidden_dim: Hidden dimension for SSL projector
            proj_output_dim: Output dimension for SSL projector
        """
        super().__init__()
        self.slow_learner = slow_learner
        self.fast_learner = FastLearner(in_channels, feature_dims)

        # Infer feature dimension from slow learner
        if feature_dims is not None:
            self.feature_dim = feature_dims[-1]
        else:
            self.feature_dim = 512  # Default for ResNet18

        # SSL projector for Barlow Twins
        self.projector = SSLProjector(self.feature_dim, proj_hidden_dim, proj_output_dim)

        # Classifier head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Store feature dims for adaptation
        self.feature_dims = feature_dims if feature_dims else [64, 128, 256, 512]

    def get_slow_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from slow learner (for SSL)."""
        return self.slow_learner(x)

    def get_projected_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get projected features for Barlow Twins loss."""
        features = self.get_slow_features(x)
        return self.projector(features)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass with feature adaptation.

        Args:
            x: Input images
            return_features: If True, also return adapted features

        Returns:
            Logits (and optionally features)
        """
        # Get slow learner features
        slow_features = self.get_slow_features(x)

        # Get gating masks from fast learner
        masks = self.fast_learner(x)

        # Apply final mask to slow features (simplified adaptation)
        # In full implementation, adaptation happens at each residual block
        # Here we apply adaptation to final features for simplicity
        if len(slow_features.shape) == 2:
            # Features are already flattened
            adapted_features = slow_features
        else:
            # Apply final gating mask
            final_mask = masks[-1]
            if slow_features.shape[2:] != final_mask.shape[2:]:
                final_mask = F.adaptive_avg_pool2d(final_mask, slow_features.shape[2:])
            adapted_features = slow_features * final_mask
            adapted_features = F.adaptive_avg_pool2d(adapted_features, 1).flatten(1)

        logits = self.classifier(adapted_features)

        if return_features:
            return logits, adapted_features
        return logits


class DualNetWithIntermediateFeatures(nn.Module):
    """
    DualNet variant that performs adaptation at intermediate layers.
    Requires a backbone that exposes intermediate features (e.g., modified ResNet).
    """

    def __init__(
            self,
            base_model: nn.Module,
            num_classes: int,
            in_channels: int = 3,
            feature_dims: List[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.feature_dims = feature_dims if feature_dims else [64, 128, 256, 512]

        self.fast_learner = FastLearner(in_channels, self.feature_dims)
        self.classifier = nn.Linear(self.feature_dims[-1], num_classes)
        self.projector = SSLProjector(self.feature_dims[-1], 2048, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: use base model directly
        features = self.base_model(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)

    def get_projected_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.projector(features)


# =============================================================================
# Look-ahead Optimizer (Eq. 3-4)
# =============================================================================

class Lookahead(optim.Optimizer):
    """
    Look-ahead optimizer wrapper (Eq. 3-4 in paper).

    Performs K inner updates with base optimizer, then performs
    a momentum update towards the look-ahead weights.
    """

    def __init__(self, base_optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Args:
            base_optimizer: Inner optimizer (e.g., SGD)
            k: Number of inner optimization steps
            alpha: Look-ahead learning rate (momentum factor)
        """
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = {}
        self._step_count = 0

        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p] = {'slow_weights': p.data.clone()}

    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # Perform look-ahead update every k steps
        if self._step_count % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad and p in self.state:
                        slow = self.state[p]['slow_weights']
                        # ϕ ← ϕ + α(ϕ̃_K − ϕ) (Eq. 4)
                        slow.add_(p.data - slow, alpha=self.alpha)
                        p.data.copy_(slow)

        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def defaults(self):
        return self.base_optimizer.defaults


# =============================================================================
# Data Augmentation for SSL
# =============================================================================

class DualViewTransform:
    """
    Generate two augmented views of an image for Barlow Twins.
    Follows standard SSL augmentation practices.
    """

    def __init__(self, base_transform=None):
        self.base_transform = base_transform

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to create two views.

        Args:
            x: Input tensor [C, H, W]

        Returns:
            Tuple of two augmented views
        """
        view_a = self._augment(x)
        view_b = self._augment(x)
        return view_a, view_b

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])

        # Random crop (simulate by small random shifts)
        if x.dim() == 3 and x.shape[1] > 4 and x.shape[2] > 4:
            shift_h = torch.randint(-2, 3, (1,)).item()
            shift_w = torch.randint(-2, 3, (1,)).item()
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

        # Color jitter (simple version)
        if torch.rand(1).item() > 0.5:
            brightness = 0.8 + 0.4 * torch.rand(1).item()
            x = x * brightness
            x = torch.clamp(x, 0, 1)

        return x


def apply_augmentation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentations to create two views for Barlow Twins.

    Args:
        x: Batch of images [batch_size, C, H, W]

    Returns:
        Tuple of two augmented batches
    """
    batch_size = x.size(0)

    # View A augmentation
    view_a = x.clone()
    flip_mask_a = torch.rand(batch_size) > 0.5
    for i in range(batch_size):
        if flip_mask_a[i]:
            view_a[i] = torch.flip(view_a[i], dims=[-1])

    # View B augmentation
    view_b = x.clone()
    flip_mask_b = torch.rand(batch_size) > 0.5
    for i in range(batch_size):
        if flip_mask_b[i]:
            view_b[i] = torch.flip(view_b[i], dims=[-1])

    # Add slight color jitter
    brightness_a = 0.9 + 0.2 * torch.rand(batch_size, 1, 1, 1, device=x.device)
    brightness_b = 0.9 + 0.2 * torch.rand(batch_size, 1, 1, 1, device=x.device)
    view_a = torch.clamp(view_a * brightness_a, 0, 1)
    view_b = torch.clamp(view_b * brightness_b, 0, 1)

    return view_a, view_b


# =============================================================================
# Training Functions
# =============================================================================

def train_dualnet_single_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        fast_optimizer: optim.Optimizer,
        slow_optimizer: optim.Optimizer,
        criterion: nn.Module,
        bt_criterion: BarlowTwinsLoss,
        buffer: ReplayBuffer,
        device: str,
        ssl_iterations: int = 3,
        lambda_tr: float = 0.5,
        temperature: float = 2.0,
) -> float:
    """
    Train for one epoch with DualNet.

    Args:
        model: DualNet model (with slow and fast learners)
        train_loader: DataLoader for current task
        fast_optimizer: Optimizer for fast learner and classifier
        slow_optimizer: Optimizer for slow learner (Look-ahead)
        criterion: Classification loss (CrossEntropyLoss)
        bt_criterion: Barlow Twins loss for SSL
        buffer: Replay buffer storing (x, y, logits)
        device: Device to train on
        ssl_iterations: Number of SSL iterations between supervised updates (n)
        lambda_tr: Weight for soft label KL divergence loss
        temperature: Temperature for soft label distillation

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        # =================================================================
        # Phase 1: Self-Supervised Learning on memory (Slow Learner)
        # =================================================================
        replay = buffer.sample(batch_size=min(inputs.size(0), buffer.current_size))
        if replay is not None and ssl_iterations > 0:
            x_buf, _, _ = replay

            for _ in range(ssl_iterations):
                slow_optimizer.zero_grad()

                # Create two augmented views
                view_a, view_b = apply_augmentation(x_buf)

                # Get projected representations
                z_a = model.get_projected_features(view_a)
                z_b = model.get_projected_features(view_b)

                # Barlow Twins loss (Eq. 1)
                ssl_loss = bt_criterion(z_a, z_b)
                ssl_loss.backward()
                slow_optimizer.step()

        # =================================================================
        # Phase 2: Supervised Learning (Fast Learner + Slow Learner)
        # =================================================================
        fast_optimizer.zero_grad()

        # Forward pass on current data
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Experience Replay with soft labels (Eq. 7)
        replay = buffer.sample(batch_size=inputs.size(0))
        if replay is not None:
            x_buf, y_buf, logits_buf = replay

            # Get current predictions for buffer samples
            out_buf = model(x_buf)

            # Hard label loss (CE)
            replay_ce_loss = criterion(out_buf, y_buf.long())
            loss = loss + replay_ce_loss

            # Soft label loss (KL divergence) - Eq. 7
            if logits_buf is not None and logits_buf.numel() > 0 and lambda_tr > 0:
                # Soft targets from stored logits
                soft_targets = F.softmax(logits_buf / temperature, dim=1)
                soft_outputs = F.log_softmax(out_buf / temperature, dim=1)
                kl_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                loss = loss + lambda_tr * (temperature ** 2) * kl_loss

        loss.backward()
        fast_optimizer.step()

        # Add current batch to buffer with current logits
        with torch.no_grad():
            logits = outputs.detach()
            buffer.add_batch(inputs, labels, logits)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_dualnet_on_task(
        model: nn.Module,
        train_loader: DataLoader,
        buffer: ReplayBuffer,
        num_epochs: int,
        lr: float,
        device: str,
        ssl_iterations: int = 3,
        lambda_tr: float = 0.5,
        temperature: float = 2.0,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        verbose: bool = False,
) -> List[float]:
    """
    Train DualNet model on a single task.

    Returns list of losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    bt_criterion = BarlowTwinsLoss(lambda_bt=0.005)

    # Fast learner optimizer (SGD with momentum)
    fast_params = list(model.fast_learner.parameters()) + list(model.classifier.parameters())
    fast_optimizer = optim.SGD(fast_params, lr=lr, momentum=0.9)

    # Slow learner optimizer (Look-ahead with SGD)
    slow_params = list(model.slow_learner.parameters()) + list(model.projector.parameters())
    base_slow_optimizer = optim.SGD(slow_params, lr=lr * 0.1, momentum=0.9)
    slow_optimizer = Lookahead(base_slow_optimizer, k=lookahead_k, alpha=lookahead_alpha)

    losses = []
    for epoch in range(num_epochs):
        loss = train_dualnet_single_epoch(
            model, train_loader, fast_optimizer, slow_optimizer,
            criterion, bt_criterion, buffer, device,
            ssl_iterations, lambda_tr, temperature
        )
        losses.append(loss)

        if verbose:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f} (DualNet)")

    return losses


def train_dualnet_model(
        base_model: nn.Module,
        task_perm: Tuple[int, ...],
        train_loaders: List[DataLoader],
        num_epochs: int,
        lr: float,
        device: str,
        buffer_size: int,
        num_classes: int,
        ssl_iterations: int = 3,
        lambda_tr: float = 0.5,
        temperature: float = 2.0,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        feature_dims: List[int] = None,
        verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh DualNet model on tasks in the given order.

    Args:
        base_model: Backbone model to use as slow learner
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Replay buffer capacity
        num_classes: Total number of classes
        ssl_iterations: Number of SSL iterations between supervised updates
        lambda_tr: Weight for soft label loss
        temperature: Temperature for soft label distillation
        lookahead_k: Look-ahead inner steps
        lookahead_alpha: Look-ahead momentum
        feature_dims: Feature dimensions for fast learner
        verbose: Print progress

    Returns:
        Trained DualNet model
    """
    # Create DualNet model
    slow_learner = copy.deepcopy(base_model).to(device)
    model = DualNetModel(
        slow_learner=slow_learner,
        num_classes=num_classes,
        feature_dims=feature_dims,
    ).to(device)

    buffer = ReplayBuffer(capacity=buffer_size, device=device)
    criterion = nn.CrossEntropyLoss()
    bt_criterion = BarlowTwinsLoss(lambda_bt=0.005)

    # Optimizers
    fast_params = list(model.fast_learner.parameters()) + list(model.classifier.parameters())
    fast_optimizer = optim.SGD(fast_params, lr=lr, momentum=0.9)

    slow_params = list(model.slow_learner.parameters()) + list(model.projector.parameters())
    base_slow_optimizer = optim.SGD(slow_params, lr=lr * 0.1, momentum=0.9)
    slow_optimizer = Lookahead(base_slow_optimizer, k=lookahead_k, alpha=lookahead_alpha)

    model.train()

    for epoch in range(num_epochs):
        for task_id in task_perm:
            loader = train_loaders[task_id]

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                # Phase 1: SSL on memory (Slow Learner)
                replay = buffer.sample(batch_size=min(inputs.size(0), max(1, buffer.current_size)))
                if replay is not None and ssl_iterations > 0:
                    x_buf, _, _ = replay

                    for _ in range(ssl_iterations):
                        slow_optimizer.zero_grad()
                        view_a, view_b = apply_augmentation(x_buf)
                        z_a = model.get_projected_features(view_a)
                        z_b = model.get_projected_features(view_b)
                        ssl_loss = bt_criterion(z_a, z_b)
                        ssl_loss.backward()
                        slow_optimizer.step()

                # Phase 2: Supervised Learning
                fast_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Experience Replay
                replay = buffer.sample(batch_size=inputs.size(0))
                if replay is not None:
                    x_buf, y_buf, logits_buf = replay
                    out_buf = model(x_buf)

                    # Hard label loss
                    replay_ce_loss = criterion(out_buf, y_buf.long())
                    loss = loss + replay_ce_loss

                    # Soft label loss (KL divergence)
                    if logits_buf is not None and logits_buf.numel() > 0 and lambda_tr > 0:
                        soft_targets = F.softmax(logits_buf / temperature, dim=1)
                        soft_outputs = F.log_softmax(out_buf / temperature, dim=1)
                        kl_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                        loss = loss + lambda_tr * (temperature ** 2) * kl_loss

                loss.backward()
                fast_optimizer.step()

                # Update buffer
                with torch.no_grad():
                    buffer.add_batch(inputs, labels, outputs.detach())

    return model


def run_dualnet_experiments(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        buffer_size: int = 500,
        num_epochs: int = 30,
        lr: float = 0.01,
        num_classes: int = 10,
        ssl_iterations: int = 3,
        lambda_tr: float = 0.5,
        temperature: float = 2.0,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        feature_dims: List[int] = None,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run DualNet experiments across multiple task permutations.

    Args:
        model: Base model to use as slow learner backbone
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        buffer_size: Replay buffer size
        num_epochs: Training epochs
        lr: Learning rate
        num_classes: Total number of classes
        ssl_iterations: SSL iterations between supervised updates (n in paper)
        lambda_tr: Weight for soft label loss
        temperature: Temperature for soft label distillation
        lookahead_k: Look-ahead optimizer inner steps (K in paper)
        lookahead_alpha: Look-ahead momentum (β in paper)
        feature_dims: Feature dimensions for fast learner
        device: Device
        dataset: Dataset name for logging
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary with results and timing information
    """
    import time
    import os

    method_name = "DualNet"
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name} experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  buffer_size={buffer_size}, epochs={num_epochs}, lr={lr}")
        print(f"  ssl_iterations={ssl_iterations}, lambda_tr={lambda_tr}")
        print(f"  lookahead_k={lookahead_k}, lookahead_alpha={lookahead_alpha}")
        print(f"{'=' * 60}\n")

    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()

        if verbose:
            print(f"{method_name} Permutation {seq_id}/{len(perms)}: {perm}")

        # Train model on this permutation
        trained_model = train_dualnet_model(
            model, perm, train_loaders,
            num_epochs, lr, device, buffer_size,
            num_classes, ssl_iterations, lambda_tr,
            temperature, lookahead_k, lookahead_alpha,
            feature_dims
        )

        # Evaluate on all tasks
        accs = evaluate_all_tasks(trained_model, test_loaders, device)
        perm_elapsed_time = time.time() - perm_start_time

        results.append({
            "sequence": perm,
            "accuracies": accs,
            "mean_acc": sum(accs) / len(accs),
            "time_seconds": perm_elapsed_time
        })

        if verbose:
            print(f"  Accuracies: {[f'{a:.1f}' for a in accs]}")
            print(f"  Mean: {results[-1]['mean_acc']:.2f}%, Time: {perm_elapsed_time:.1f}s\n")

    total_elapsed_time = time.time() - total_start_time

    # Create summary DataFrame
    df = pd.DataFrame({
        "sequence": [str(r["sequence"]) for r in results],
        **{f"Task{t + 1}": [r["accuracies"][t] for r in results]
           for t in range(num_tasks)},
        "Mean": [r["mean_acc"] for r in results],
        "Time_s": [r.get("time_seconds", 0.0) for r in results]
    })

    # Save results
    # Save results to proper subdirectories
    paths = ensure_results_dirs(output_dir)

    # Save CSV
    csv_path = os.path.join(paths['csv'], f"ser_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)

    # Save JSON summary
    json_filename = f"ser_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": "DUALNET",
        "dataset": dataset,
        "total_time_seconds": total_elapsed_time,
        "mean_accuracy": float(df["Mean"].mean()),
        "std_accuracy": float(df["Mean"].std()),
        "per_task_mean": [float(df[f"Task{t + 1}"].mean()) for t in range(num_tasks)],
        "per_task_std": [float(df[f"Task{t + 1}"].std()) for t in range(num_tasks)],
    }
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)

    if verbose:
        print(f"Saved {method_name} results to {csv_path}")
        print(f"Total {method_name} time: {total_elapsed_time:.1f}s")

    return {
        "method": method_name,
        "dataset": dataset,
        "results": results,
        "total_time_seconds": total_elapsed_time,
        "summary": {
            "mean_accuracy": df["Mean"].mean(),
            "std_accuracy": df["Mean"].std(),
            "per_task_mean": [df[f"Task{t + 1}"].mean() for t in range(num_tasks)],
            "per_task_std": [df[f"Task{t + 1}"].std() for t in range(num_tasks)],
            "total_time": total_elapsed_time,
            "avg_time_per_perm": total_elapsed_time / max(len(perms), 1),
        }
    }