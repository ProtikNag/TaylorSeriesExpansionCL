"""
Spectral Regularization for Continual Learning.

This implements spectral regularization to maintain trainability in continual learning
by keeping the maximum singular value of each layer close to one.

Key equation from the paper:
    R(θ) = Σ_l [(σ₁(W_l)^k - 1)² + ||b_l||^{2k}]

where σ₁(W) is the spectral norm (largest singular value) computed via power iteration.

Key insights:
- Regularize spectral norm towards 1 (not 0) to prevent collapse
- Include bias terms in regularization
- Use k=2 for balanced stability and effectiveness
- Single power iteration is sufficient for effective regularization

Reference: Lewandowski et al., "Learning Continually by Spectral Regularization", 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from typing import List, Tuple, Optional, Dict, Any, Union
from torch.utils.data import DataLoader
import pandas as pd
import json
from datetime import datetime

from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


def power_iteration(
        weight: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        num_iters: int = 1,
        eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the spectral norm (largest singular value) using power iteration.

    Args:
        weight: Weight matrix of shape (out_features, in_features) or reshaped conv
        u: Initial vector for power iteration (will be created if None)
        num_iters: Number of power iterations (1 is sufficient per paper)
        eps: Small constant for numerical stability

    Returns:
        sigma: Spectral norm (largest singular value)
        u: Updated u vector for next iteration
    """
    # Reshape to 2D if necessary
    height = weight.size(0)
    width = weight.view(height, -1).size(1)
    weight_mat = weight.view(height, width)

    # Initialize u vector if not provided
    if u is None:
        u = torch.randn(height, device=weight.device, dtype=weight.dtype)
        u = F.normalize(u, dim=0, eps=eps)

    # Power iteration
    with torch.no_grad():
        for _ in range(num_iters):
            # v = W^T u / ||W^T u||
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps)
            # u = W v / ||W v||
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=eps)

    # Compute spectral norm: σ = u^T W v
    # We need gradients here, so recompute without no_grad
    v = F.normalize(torch.mv(weight_mat.t(), u.detach()), dim=0, eps=eps)
    sigma = torch.dot(u.detach(), torch.mv(weight_mat, v))

    return sigma, u.detach()


def reshape_weight_for_spectral_norm(weight: torch.Tensor) -> torch.Tensor:
    """
    Reshape weight tensor for spectral norm computation.

    For conv layers: (out_channels, in_channels, kH, kW) -> (out_channels, in_channels*kH*kW)
    For linear layers: already 2D

    This provides an efficient upper bound on the true spectral norm of the
    Toeplitz matrix defining the convolution (Tsuzuku et al., 2018).
    """
    if weight.dim() > 2:
        # Convolutional layer: reshape to (out_channels, in_channels * kernel_size)
        return weight.view(weight.size(0), -1)
    return weight


class SpectralRegularizer:
    """
    Manages spectral regularization state (u vectors) for all layers.

    Stores the u vectors from power iteration for each layer to enable
    faster convergence across training iterations.
    """

    def __init__(self, model: nn.Module, k: int = 2):
        """
        Initialize spectral regularizer for a model.

        Args:
            model: Neural network to regularize
            k: Exponent for spectral norm penalty (default 2 as in paper)
        """
        self.k = k
        self.u_vectors: Dict[str, torch.Tensor] = {}
        self._register_layers(model)

    def _register_layers(self, model: nn.Module):
        """Register all layers that need spectral regularization."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Initialize u vector for weight matrix
                weight = module.weight
                height = weight.size(0)
                self.u_vectors[f"{name}.weight"] = torch.randn(
                    height, device=weight.device, dtype=weight.dtype
                )
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LayerNorm, nn.GroupNorm)):
                # Normalization layers have scaling parameters
                pass  # These are handled separately (element-wise towards 1)

    def compute_regularization(
            self,
            model: nn.Module,
            num_power_iters: int = 1,
    ) -> torch.Tensor:
        """
        Compute total spectral regularization loss for the model.

        R(θ) = Σ_l [(σ₁(W_l)^k - 1)² + ||b_l||^{2k}]

        For normalization layers, we regularize each scaling parameter towards 1.

        Args:
            model: Neural network
            num_power_iters: Number of power iterations (1 is sufficient)

        Returns:
            Total regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, module in model.named_modules():
            # Linear and Convolutional layers
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weight = module.weight

                # Compute spectral norm of weight matrix
                weight_reshaped = reshape_weight_for_spectral_norm(weight)
                u_key = f"{name}.weight"

                # Get or initialize u vector
                if u_key in self.u_vectors:
                    u = self.u_vectors[u_key].to(weight.device)
                else:
                    u = None

                sigma, u_new = power_iteration(weight_reshaped, u, num_power_iters)
                self.u_vectors[u_key] = u_new

                # Spectral norm regularization: (σ^k - 1)²
                # Regularize towards 1, not 0 (key insight from paper)
                reg_loss = reg_loss + (sigma.pow(self.k) - 1).pow(2)

                # Bias regularization: ||b||^{2k}
                if module.bias is not None:
                    bias_norm = torch.norm(module.bias, p=2)
                    reg_loss = reg_loss + bias_norm.pow(2 * self.k)

            # Batch Normalization layers
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.weight is not None:  # gamma (scaling)
                    # For diagonal matrices, spectral norm = max|γ_i|
                    # But max is not differentiable, so regularize each towards 1
                    # As per Appendix A.7 of the paper
                    reg_loss = reg_loss + ((module.weight.pow(self.k) - 1).pow(2)).sum()
                if module.bias is not None:  # beta (shift)
                    reg_loss = reg_loss + module.bias.pow(2 * self.k).sum()

            # Layer Normalization
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    reg_loss = reg_loss + ((module.weight.pow(self.k) - 1).pow(2)).sum()
                if module.bias is not None:
                    reg_loss = reg_loss + module.bias.pow(2 * self.k).sum()

            # Group Normalization
            elif isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    reg_loss = reg_loss + ((module.weight.pow(self.k) - 1).pow(2)).sum()
                if module.bias is not None:
                    reg_loss = reg_loss + module.bias.pow(2 * self.k).sum()

        return reg_loss


def compute_spectral_regularization(
        model: nn.Module,
        k: int = 2,
        num_power_iters: int = 1,
) -> torch.Tensor:
    """
    Compute spectral regularization loss for a model (stateless version).

    This is a convenience function that doesn't maintain u vectors across calls.
    For better efficiency during training, use the SpectralRegularizer class.

    Args:
        model: Neural network
        k: Exponent for spectral norm penalty
        num_power_iters: Number of power iterations

    Returns:
        Total regularization loss
    """
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            weight = module.weight
            weight_reshaped = reshape_weight_for_spectral_norm(weight)
            sigma, _ = power_iteration(weight_reshaped, num_iters=num_power_iters)

            # (σ^k - 1)²
            reg_loss = reg_loss + (sigma.pow(k) - 1).pow(2)

            if module.bias is not None:
                bias_norm = torch.norm(module.bias, p=2)
                reg_loss = reg_loss + bias_norm.pow(2 * k)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                 nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                reg_loss = reg_loss + ((module.weight.pow(k) - 1).pow(2)).sum()
            if module.bias is not None:
                reg_loss = reg_loss + module.bias.pow(2 * k).sum()

    return reg_loss


def train_spectral_single_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        spectral_reg: SpectralRegularizer,
        device: str,
        lambda_reg: float = 0.0001,
        num_power_iters: int = 1,
) -> float:
    """
    Train for one epoch with Spectral Regularization.

    Args:
        model: Neural network
        train_loader: DataLoader for current task
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss)
        spectral_reg: SpectralRegularizer instance
        device: Device to train on
        lambda_reg: Weight for spectral regularization (default 0.0001 as in paper)
        num_power_iters: Number of power iterations for spectral norm

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        # Forward pass on current data
        outputs = model(inputs)
        task_loss = criterion(outputs, labels)

        # Compute spectral regularization
        # R(θ) = Σ_l [(σ₁(W_l)^k - 1)² + ||b_l||^{2k}]
        reg_loss = spectral_reg.compute_regularization(model, num_power_iters)

        # Total loss: J_τ^λ(θ) = J_τ(θ) + λR(θ)
        loss = task_loss + lambda_reg * reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_spectral_on_task(
        model: nn.Module,
        train_loader: DataLoader,
        num_epochs: int,
        lr: float,
        device: str,
        lambda_reg: float = 0.0001,
        k: int = 2,
        num_power_iters: int = 1,
        verbose: bool = False,
) -> List[float]:
    """
    Train model on a single task with Spectral Regularization.

    Args:
        model: Neural network
        train_loader: DataLoader for current task
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        lambda_reg: Regularization strength (default 0.0001)
        k: Exponent for spectral norm penalty (default 2)
        num_power_iters: Number of power iterations
        verbose: Print progress

    Returns:
        List of losses per epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    spectral_reg = SpectralRegularizer(model, k=k)

    losses = []
    for epoch in range(num_epochs):
        loss = train_spectral_single_epoch(
            model, train_loader, optimizer, criterion,
            spectral_reg, device, lambda_reg, num_power_iters
        )
        losses.append(loss)

        if verbose:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f} (Spectral Reg)")

    return losses


def train_spectral_model(
        base_model: nn.Module,
        task_perm: Tuple[int, ...],
        train_loaders: List[DataLoader],
        num_epochs: int,
        lr: float,
        device: str,
        lambda_reg: float = 0.0001,
        k: int = 2,
        num_power_iters: int = 1,
        verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh copy of the model on tasks in the given order using Spectral Regularization.

    Args:
        base_model: Model to clone and train
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        lambda_reg: Regularization strength (default 0.0001 as in paper)
        k: Exponent for spectral norm penalty (default 2)
        num_power_iters: Number of power iterations (1 is sufficient)
        verbose: Print progress

    Returns:
        Trained model
    """
    model = copy.deepcopy(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    spectral_reg = SpectralRegularizer(model, k=k)

    model.train()

    for epoch in range(num_epochs):
        for task_id in task_perm:
            loader = train_loaders[task_id]

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(inputs)
                task_loss = criterion(outputs, labels)

                # Spectral regularization: keeps spectral norm close to 1
                # This maintains gradient diversity and trainability
                reg_loss = spectral_reg.compute_regularization(model, num_power_iters)

                # Combined loss
                loss = task_loss + lambda_reg * reg_loss

                loss.backward()
                optimizer.step()

    return model


def run_spectral_experiments(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        num_epochs: int = 30,
        lr: float = 0.01,
        lambda_reg: float = 0.0001,
        k: int = 2,
        num_power_iters: int = 1,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Spectral Regularization experiments across multiple task permutations.

    Args:
        model: Base model to clone for each permutation
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        num_epochs: Training epochs
        lr: Learning rate
        lambda_reg: Regularization strength (default 0.0001 as in paper)
        k: Exponent for spectral norm penalty (default 2)
        num_power_iters: Number of power iterations (1 is sufficient)
        device: Device
        dataset: Dataset name for logging
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary with results and timing information
    """
    import time
    import os

    method_name = "SpectralReg"
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name} experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  epochs={num_epochs}, lr={lr}")
        print(f"  lambda={lambda_reg}, k={k}, power_iters={num_power_iters}")
        print(f"{'=' * 60}\n")

    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()

        if verbose:
            print(f"{method_name} Permutation {seq_id}/{len(perms)}: {perm}")

        # Train model on this permutation
        trained_model = train_spectral_model(
            model, perm, train_loaders,
            num_epochs, lr, device,
            lambda_reg, k, num_power_iters
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
        "method": "SPECTRAL",
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


# ============================================================================
# Monitoring utilities for analyzing spectral properties during training
# ============================================================================

def compute_spectral_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Compute spectral statistics of a model for analysis.

    Returns statistics mentioned in the paper:
    - Average max singular value across layers
    - Average min singular value across layers
    - Average condition number
    - Per-layer spectral norms

    Args:
        model: Neural network

    Returns:
        Dictionary with spectral statistics
    """
    stats = {
        "max_singular_values": [],
        "min_singular_values": [],
        "condition_numbers": [],
        "frobenius_norms": [],
        "layer_names": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            weight = module.weight.detach()
            weight_2d = reshape_weight_for_spectral_norm(weight)

            # Compute full SVD for detailed analysis
            try:
                U, S, V = torch.svd(weight_2d)
                max_sv = S[0].item()
                min_sv = S[-1].item() if S[-1] > 1e-10 else 1e-10
                cond = max_sv / min_sv
            except:
                # Fallback to power iteration
                max_sv, _ = power_iteration(weight_2d)
                max_sv = max_sv.item()
                min_sv = 0.0
                cond = float('inf')

            frob_norm = torch.norm(weight_2d, p='fro').item()

            stats["max_singular_values"].append(max_sv)
            stats["min_singular_values"].append(min_sv)
            stats["condition_numbers"].append(cond)
            stats["frobenius_norms"].append(frob_norm)
            stats["layer_names"].append(name)

    # Compute aggregates
    if stats["max_singular_values"]:
        stats["avg_max_sv"] = sum(stats["max_singular_values"]) / len(stats["max_singular_values"])
        stats["avg_min_sv"] = sum(stats["min_singular_values"]) / len(stats["min_singular_values"])
        stats["avg_condition"] = sum(stats["condition_numbers"]) / len(stats["condition_numbers"])
        stats["total_frob_norm"] = sum(stats["frobenius_norms"])

    return stats


def compute_gradient_diversity(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: str,
        num_samples: int = 100,
) -> Dict[str, float]:
    """
    Compute gradient diversity metrics as described in Section 3.3.

    Measures effective rank of the gradient matrix to assess trainability.

    Args:
        model: Neural network
        data_loader: DataLoader to sample from
        criterion: Loss function
        device: Device
        num_samples: Number of samples to use

    Returns:
        Dictionary with gradient diversity metrics
    """
    model.eval()
    gradients = []

    # Collect per-example gradients
    count = 0
    for inputs, labels in data_loader:
        if count >= num_samples:
            break

        for i in range(min(inputs.size(0), num_samples - count)):
            x = inputs[i:i + 1].to(device)
            y = labels[i:i + 1].to(device).long()

            model.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # Flatten all gradients into a single vector
            grad_vec = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_vec.append(p.grad.view(-1).detach().clone())
            gradients.append(torch.cat(grad_vec))
            count += 1

    if not gradients:
        return {"effective_rank": 0.0, "condition_number": float('inf')}

    # Stack into gradient matrix G = [g_1, ..., g_m]
    G = torch.stack(gradients, dim=1)  # (num_params, num_samples)

    # Compute SVD
    try:
        U, S, V = torch.svd(G)
        S = S[S > 1e-10]  # Filter near-zero singular values

        if len(S) == 0:
            return {"effective_rank": 0.0, "condition_number": float('inf')}

        # Effective rank: exp(-Σ p_i log p_i) where p_i = σ_i / Σσ_j
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        # Condition number
        condition_number = (S[0] / S[-1]).item()

        return {
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "num_singular_values": len(S),
            "max_singular_value": S[0].item(),
            "min_singular_value": S[-1].item(),
        }
    except:
        return {"effective_rank": 0.0, "condition_number": float('inf')}


# ============================================================================
# Convenience wrappers with paper's recommended hyperparameters
# ============================================================================

def run_spectral_experiments_default(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        num_epochs: int = 30,
        lr: float = 0.01,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Spectral Regularization with paper's default hyperparameters.

    Default: lambda=0.0001, k=2, num_power_iters=1
    These defaults work well across CIFAR10, CIFAR100, tiny-ImageNet.
    """
    return run_spectral_experiments(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        perms=perms,
        num_epochs=num_epochs,
        lr=lr,
        lambda_reg=0.0001,  # Paper's default for CIFAR/ImageNet
        k=2,
        num_power_iters=1,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        verbose=verbose,
    )


def run_spectral_experiments_svhn(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        num_epochs: int = 30,
        lr: float = 0.01,
        device: str = "cuda",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Spectral Regularization with hyperparameters tuned for SVHN.

    The paper found lambda=0.001 works best for SVHN.
    """
    return run_spectral_experiments(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        perms=perms,
        num_epochs=num_epochs,
        lr=lr,
        lambda_reg=0.001,  # Paper's default for SVHN
        k=2,
        num_power_iters=1,
        device=device,
        dataset="SVHN",
        output_dir=output_dir,
        verbose=verbose,
    )