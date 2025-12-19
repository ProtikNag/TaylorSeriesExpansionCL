"""
Experience Replay (ER) implementation for continual learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd

from .buffer import ReplayBuffer
from ..utils import evaluate, evaluate_all_tasks


def train_er_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    buffer: ReplayBuffer,
    device: str,
    beta: float = 0.5,
) -> float:
    """
    Train for one epoch with experience replay.
    
    Args:
        model: Neural network
        train_loader: DataLoader for current task
        optimizer: Optimizer
        criterion: Loss function
        buffer: Replay buffer
        device: Device to train on
        beta: Weight for replay loss
    
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
        
        # Replay from buffer
        replay = buffer.sample(batch_size=inputs.size(0))
        if replay is not None:
            x_buf, y_buf, _ = replay
            out_buf = model(x_buf)
            replay_loss = criterion(out_buf, y_buf.long())
            loss = loss + beta * replay_loss
        
        loss.backward()
        optimizer.step()
        
        # Add current batch to buffer
        with torch.no_grad():
            logits = outputs.detach()
            buffer.add_batch(inputs, labels, logits)
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_er_on_task(
    model: nn.Module,
    train_loader: DataLoader,
    buffer: ReplayBuffer,
    num_epochs: int,
    lr: float,
    device: str,
    beta: float = 0.5,
    verbose: bool = False,
) -> List[float]:
    """
    Train model on a single task with ER.
    
    Returns list of losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    losses = []
    for epoch in range(num_epochs):
        loss = train_er_single_epoch(
            model, train_loader, optimizer, criterion,
            buffer, device, beta
        )
        losses.append(loss)
        
        if verbose:
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    return losses


def train_er_model(
    base_model: nn.Module,
    task_perm: Tuple[int, ...],
    train_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int,
    beta: float = 0.5,
    verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh copy of the model on tasks in the given order.
    
    Args:
        base_model: Model to clone and train
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Replay buffer capacity
        beta: Replay loss weight
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
                
                # Replay
                replay = buffer.sample(batch_size=inputs.size(0))
                if replay is not None:
                    x_buf, y_buf, _ = replay
                    out_buf = model(x_buf)
                    replay_loss = criterion(out_buf, y_buf.long())
                    loss = loss + beta * replay_loss
                
                loss.backward()
                optimizer.step()
                
                # Update buffer
                with torch.no_grad():
                    buffer.add_batch(inputs, labels, outputs.detach())
    
    return model


def run_er_experiments(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    perms: List[Tuple[int, ...]],
    buffer_size: int = 500,
    num_epochs: int = 30,
    lr: float = 0.01,
    beta: float = 0.5,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ER experiments across multiple task permutations.
    
    Args:
        model: Base model to clone for each permutation
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        buffer_size: Replay buffer size
        num_epochs: Training epochs
        lr: Learning rate
        beta: Replay loss weight
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
        print(f"\n{'='*60}")
        print(f"Running ER experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  buffer_size={buffer_size}, epochs={num_epochs}, lr={lr}")
        print(f"{'='*60}\n")
    
    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()
        
        if verbose:
            print(f"ER Permutation {seq_id}/{len(perms)}: {perm}")
        
        # Train model on this permutation
        trained_model = train_er_model(
            model, perm, train_loaders,
            num_epochs, lr, device, buffer_size, beta
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
        **{f"Task{t+1}": [r["accuracies"][t] for r in results] 
           for t in range(num_tasks)},
        "Mean": [r["mean_acc"] for r in results],
        "Time_s": [r.get("time_seconds", 0.0) for r in results]
    })
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"er_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"Saved ER results to {csv_path}")
        print(f"Total ER time: {total_elapsed_time:.1f}s")
    
    return {
        "method": "ER",
        "dataset": dataset,
        "results": results,
        "total_time_seconds": total_elapsed_time,
        "summary": {
            "mean_accuracy": df["Mean"].mean(),
            "std_accuracy": df["Mean"].std(),
            "per_task_mean": [df[f"Task{t+1}"].mean() for t in range(num_tasks)],
            "per_task_std": [df[f"Task{t+1}"].std() for t in range(num_tasks)],
            "total_time": total_elapsed_time,
            "avg_time_per_perm": total_elapsed_time / max(len(perms), 1),
        }
    }
