"""
Elastic Weight Consolidation (EWC) implementation for continual learning.

EWC slows down learning on weights important for previous tasks by adding
a quadratic penalty weighted by the Fisher Information Matrix.

Key equation:
    L_total = L_current + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
           PNAS 2017, https://www.pnas.org/doi/10.1073/pnas.1611835114
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd
import os
import json
from datetime import datetime

from .buffer import ReplayBuffer
from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


def compute_fisher_information(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """
    Compute the diagonal of the Fisher Information Matrix.
    
    Uses the empirical Fisher approximation:
    F_i = E[(d log p(y|x,theta) / d theta_i)^2]
    
    Args:
        model: Neural network
        data_loader: DataLoader for computing Fisher
        device: Device to compute on
        num_samples: Maximum number of samples to use
    
    Returns:
        Dictionary mapping parameter names to Fisher diagonal values
    """
    model.eval()
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    criterion = nn.CrossEntropyLoss()
    sample_count = 0
    
    for inputs, labels in data_loader:
        if sample_count >= num_samples:
            break
            
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        batch_size = inputs.size(0)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2 * batch_size
        
        sample_count += batch_size
    
    # Average Fisher
    for name in fisher:
        fisher[name] /= max(sample_count, 1)
    
    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    optimal_params: Dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Compute the EWC penalty term.
    
    penalty = (1/2) * sum_i F_i * (theta_i - theta*_i)^2
    
    Args:
        model: Current model
        fisher: Fisher Information diagonal
        optimal_params: Parameters from previous task
        device: Device
    
    Returns:
        EWC penalty as a scalar tensor
    """
    penalty = torch.tensor(0.0, device=device)
    
    for name, param in model.named_parameters():
        if name in fisher and name in optimal_params:
            diff = param - optimal_params[name].to(device)
            penalty += (fisher[name] * diff ** 2).sum()
    
    return penalty * 0.5


def train_ewc_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    ewc_lambda: float = 400.0,
    fisher_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    optimal_params_list: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> float:
    """
    Train for one epoch with EWC regularization.
    
    Args:
        model: Neural network
        train_loader: DataLoader for current task
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss)
        device: Device to train on
        ewc_lambda: Regularization strength for EWC penalty
        fisher_list: List of Fisher matrices from previous tasks
        optimal_params_list: List of optimal parameters from previous tasks
    
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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add EWC penalty for all previous tasks
        if fisher_list and optimal_params_list:
            for fisher, optimal_params in zip(fisher_list, optimal_params_list):
                loss = loss + ewc_lambda * ewc_penalty(
                    model, fisher, optimal_params, device
                )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_ewc_on_task(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: str,
    ewc_lambda: float = 400.0,
    fisher_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    optimal_params_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    verbose: bool = False,
) -> Tuple[List[float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Train model on a single task with EWC.
    
    Returns:
        Tuple of (losses per epoch, fisher for this task, optimal params for this task)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    losses = []
    for epoch in range(num_epochs):
        loss = train_ewc_single_epoch(
            model, train_loader, optimizer, criterion,
            device, ewc_lambda, fisher_list, optimal_params_list
        )
        losses.append(loss)
        
        if verbose:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f} (EWC)")
    
    # Compute Fisher for this task after training
    fisher = compute_fisher_information(model, train_loader, device)
    
    # Store optimal parameters
    optimal_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    return losses, fisher, optimal_params


def train_ewc_model(
    base_model: nn.Module,
    task_perm: Tuple[int, ...],
    train_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int = 500,  # Kept for API compatibility, not used in pure EWC
    ewc_lambda: float = 400.0,
    verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh copy of the model on tasks in the given order using EWC.
    
    Args:
        base_model: Model to clone and train
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Not used in pure EWC (kept for API compatibility)
        ewc_lambda: EWC regularization strength
        verbose: Print progress
    
    Returns:
        Trained model
    """
    model = copy.deepcopy(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Store Fisher matrices and optimal parameters for all previous tasks
    fisher_list: List[Dict[str, torch.Tensor]] = []
    optimal_params_list: List[Dict[str, torch.Tensor]] = []
    
    model.train()
    
    for task_idx, task_id in enumerate(task_perm):
        loader = train_loaders[task_id]
        
        if verbose:
            print(f"  Training on task {task_id} (position {task_idx + 1}/{len(task_perm)})")
        
        # Train for num_epochs on this task
        for epoch in range(num_epochs):
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Add EWC penalty
                for fisher, opt_params in zip(fisher_list, optimal_params_list):
                    loss = loss + ewc_lambda * ewc_penalty(
                        model, fisher, opt_params, device
                    )
                
                loss.backward()
                optimizer.step()
        
        # After training on this task, compute Fisher and store parameters
        fisher = compute_fisher_information(model, loader, device)
        fisher_list.append(fisher)
        
        optimal_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        optimal_params_list.append(optimal_params)
    
    return model


def run_ewc_experiments(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    perms: List[Tuple[int, ...]],
    buffer_size: int = 500,  # Kept for API compatibility
    num_epochs: int = 30,
    lr: float = 0.01,
    ewc_lambda: float = 400.0,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run EWC experiments across multiple task permutations.
    
    Args:
        model: Base model to clone for each permutation
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        buffer_size: Not used in pure EWC (kept for API compatibility)
        num_epochs: Training epochs
        lr: Learning rate
        ewc_lambda: EWC regularization strength
        device: Device
        dataset: Dataset name for logging
        output_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Dictionary with results and timing information
    """
    import time
    
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    results = []
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running EWC experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  epochs={num_epochs}, lr={lr}, ewc_lambda={ewc_lambda}")
        print(f"{'=' * 60}\n")
    
    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()
        
        if verbose:
            print(f"EWC Permutation {seq_id}/{len(perms)}: {perm}")
        
        # Train model on this permutation
        trained_model = train_ewc_model(
            model, perm, train_loaders,
            num_epochs, lr, device, buffer_size, ewc_lambda
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
    
    # Save results to proper subdirectories
    paths = ensure_results_dirs(output_dir)
    
    # Save CSV
    csv_path = os.path.join(paths['csv'], f"ewc_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON summary
    json_filename = f"ewc_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": "EWC",
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
        print(f"Saved EWC results to {csv_path}")
        print(f"Total EWC time: {total_elapsed_time:.1f}s")
    
    return {
        "method": "EWC",
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
