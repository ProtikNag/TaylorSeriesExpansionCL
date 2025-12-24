"""
Dark Experience Replay (DER and DER++) implementation for continual learning.

DER stores the network's logits (dark knowledge) sampled throughout the optimization
trajectory and matches current outputs to stored logits using MSE loss.

DER++ additionally includes a classification loss on buffer samples.

Key equations from the paper:
- DER:   L_current + α * E[(z - h_θ(x))²]
- DER++: L_current + α * E[(z - h_θ(x))²] + β * E[CE(y, f_θ(x))]

Reference: Buzzega et al., "Dark Experience for General Continual Learning:
           a Strong, Simple Baseline", NeurIPS 2020
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


def train_der_single_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        buffer: ReplayBuffer,
        device: str,
        alpha: float = 0.5,
        beta: float = 0.0,
        use_derpp: bool = False,
) -> float:
    """
    Train for one epoch with Dark Experience Replay.

    Args:
        model: Neural network
        train_loader: DataLoader for current task
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss)
        buffer: Replay buffer storing (x, y, logits)
        device: Device to train on
        alpha: Weight for logit matching loss (MSE on logits)
        beta: Weight for classification loss on buffer (DER++ only)
        use_derpp: If True, use DER++ (includes classification loss on buffer)

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
        loss = criterion(outputs, labels)

        # Dark Experience Replay from buffer
        replay = buffer.sample(batch_size=inputs.size(0))
        if replay is not None:
            x_buf, y_buf, logits_buf = replay

            # Get current model's logits for buffer samples
            out_buf = model(x_buf)

            # DER: MSE loss between stored logits and current logits
            # This is the core of DER - matching logits directly (Eq. 5 in paper)
            if logits_buf is not None and logits_buf.numel() > 0:
                der_loss = alpha * F.mse_loss(out_buf, logits_buf)
                loss = loss + der_loss

            # DER++: Additional classification loss on buffer samples (Eq. 6 in paper)
            if use_derpp and beta > 0:
                # Sample again for the classification term (as in Algorithm 2)
                # or reuse the same samples for efficiency
                replay_ce_loss = beta * criterion(out_buf, y_buf.long())
                loss = loss + replay_ce_loss

        loss.backward()
        optimizer.step()

        # Add current batch to buffer with current logits
        # Key insight: logits are sampled throughout optimization trajectory
        with torch.no_grad():
            logits = outputs.detach()
            buffer.add_batch(inputs, labels, logits)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_der_on_task(
        model: nn.Module,
        train_loader: DataLoader,
        buffer: ReplayBuffer,
        num_epochs: int,
        lr: float,
        device: str,
        alpha: float = 0.5,
        beta: float = 0.0,
        use_derpp: bool = False,
        verbose: bool = False,
) -> List[float]:
    """
    Train model on a single task with DER/DER++.

    Returns list of losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses = []
    for epoch in range(num_epochs):
        loss = train_der_single_epoch(
            model, train_loader, optimizer, criterion,
            buffer, device, alpha, beta, use_derpp
        )
        losses.append(loss)

        if verbose:
            method = "DER++" if use_derpp else "DER"
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f} ({method})")

    return losses


def train_der_model(
        base_model: nn.Module,
        task_perm: Tuple[int, ...],
        train_loaders: List[DataLoader],
        num_epochs: int,
        lr: float,
        device: str,
        buffer_size: int,
        alpha: float = 0.5,
        beta: float = 0.0,
        use_derpp: bool = False,
        verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh copy of the model on tasks in the given order using DER/DER++.

    Args:
        base_model: Model to clone and train
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Replay buffer capacity
        alpha: Weight for logit matching loss (default 0.5 as in paper)
        beta: Weight for classification loss on buffer (DER++ only, default 0.5)
        use_derpp: If True, use DER++ instead of DER
        verbose: Print progress

    Returns:
        Trained model
    """
    model = copy.deepcopy(base_model).to(device)
    buffer = ReplayBuffer(capacity=buffer_size, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()

    for epoch in range(num_epochs):
        for task_id in task_perm:
            loader = train_loaders[task_id]

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Dark Experience Replay
                replay = buffer.sample(batch_size=inputs.size(0))
                if replay is not None:
                    x_buf, y_buf, logits_buf = replay
                    out_buf = model(x_buf)

                    # DER: Logit matching loss (MSE)
                    if logits_buf is not None and logits_buf.numel() > 0:
                        der_loss = alpha * F.mse_loss(out_buf, logits_buf)
                        loss = loss + der_loss

                    # DER++: Additional classification loss
                    if use_derpp and beta > 0:
                        replay_ce_loss = beta * criterion(out_buf, y_buf.long())
                        loss = loss + replay_ce_loss

                loss.backward()
                optimizer.step()

                # Update buffer with current logits
                with torch.no_grad():
                    buffer.add_batch(inputs, labels, outputs.detach())

    return model


def run_der_experiments(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        buffer_size: int = 500,
        num_epochs: int = 30,
        lr: float = 0.01,
        alpha: float = 0.5,
        beta: float = 0.0,
        use_derpp: bool = False,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run DER/DER++ experiments across multiple task permutations.

    Args:
        model: Base model to clone for each permutation
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        buffer_size: Replay buffer size
        num_epochs: Training epochs
        lr: Learning rate
        alpha: Weight for logit matching loss (default 0.5)
        beta: Weight for classification loss on buffer (DER++ only)
        use_derpp: If True, use DER++ instead of DER
        device: Device
        dataset: Dataset name for logging
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary with results and timing information
    """
    import time
    import os

    method_name = "DER++" if use_derpp else "DER"
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name} experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  buffer_size={buffer_size}, epochs={num_epochs}, lr={lr}")
        print(f"  alpha={alpha}" + (f", beta={beta}" if use_derpp else ""))
        print(f"{'=' * 60}\n")

    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()

        if verbose:
            print(f"{method_name} Permutation {seq_id}/{len(perms)}: {perm}")

        # Train model on this permutation
        trained_model = train_der_model(
            model, perm, train_loaders,
            num_epochs, lr, device, buffer_size,
            alpha, beta, use_derpp
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
        "method": "DER",
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


# Convenience wrapper for DER++
def run_derpp_experiments(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        perms: List[Tuple[int, ...]],
        buffer_size: int = 500,
        num_epochs: int = 30,
        lr: float = 0.01,
        alpha: float = 0.5,
        beta: float = 0.5,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run DER++ experiments (convenience wrapper).

    DER++ combines logit matching with classification loss on buffer samples.
    Default alpha=0.5, beta=0.5 as recommended in the paper.
    """
    return run_der_experiments(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        perms=perms,
        buffer_size=buffer_size,
        num_epochs=num_epochs,
        lr=lr,
        alpha=alpha,
        beta=beta,
        use_derpp=True,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        verbose=verbose,
    )