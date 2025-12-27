"""
FedProx-style consolidation for continual learning.

FedProx extends FedAvg with a proximal regularization term during local training:
    L_total = L_task + (mu/2) * ||W - W_g||^2

This encourages the local model to stay close to the global model,
reducing client drift and improving convergence stability.

After local training, consolidation uses weighted averaging similar to FedAvg:
    W_g = (1 - alpha) * W_g + alpha * W_l

Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import json
from datetime import datetime
from itertools import permutations
from typing import List, Tuple, Dict, Optional, Any
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import os
import time

from .buffer import ReplayBuffer
from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


def proximal_loss(
    model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    mu: float = 0.01,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute proximal regularization term: (mu/2) * ||W - W_g||^2
    
    Args:
        model: Current local model
        global_state: Global model state dict
        mu: Proximal regularization strength
        device: Device
    
    Returns:
        Proximal loss term
    """
    prox_loss = torch.tensor(0.0, device=device)
    
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_state:
            global_param = global_state[name].to(device)
            prox_loss = prox_loss + torch.sum((param - global_param) ** 2)
    
    return (mu / 2.0) * prox_loss


def train_ser_with_prox_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    buffer: ReplayBuffer,
    global_state: Dict[str, torch.Tensor],
    device: str,
    beta: float = 1.0,
    temperature: float = 2.0,
    alpha: float = 0.5,
    mu: float = 0.01,
) -> float:
    """
    Train for one epoch with SER + proximal regularization.
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
        
        # Strong replay from buffer
        replay = buffer.sample(batch_size=inputs.size(0))
        if replay is not None:
            x_buf, y_buf, logits_buf = replay
            out_buf = model(x_buf)
            
            replay_ce_loss = criterion(out_buf, y_buf.long())
            
            if logits_buf is not None and logits_buf.numel() > 0:
                soft_targets = F.softmax(logits_buf / temperature, dim=1)
                soft_outputs = F.log_softmax(out_buf / temperature, dim=1)
                distill_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                distill_loss = distill_loss * (temperature ** 2)
                replay_loss = (1 - alpha) * replay_ce_loss + alpha * distill_loss
            else:
                replay_loss = replay_ce_loss
            
            loss = loss + beta * replay_loss
        
        # Add proximal regularization
        prox_loss = proximal_loss(model, global_state, mu, device)
        loss = loss + prox_loss
        
        loss.backward()
        optimizer.step()
        
        # Add current batch to buffer with logits
        with torch.no_grad():
            logits = outputs.detach()
            buffer.add_batch(inputs, labels, logits)
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_der_with_prox_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    buffer: ReplayBuffer,
    global_state: Dict[str, torch.Tensor],
    device: str,
    alpha_der: float = 0.5,
    beta_der: float = 0.5,
    mu: float = 0.01,
) -> float:
    """
    Train for one epoch with DER++ + proximal regularization.
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
        
        # Dark Experience Replay
        replay = buffer.sample(batch_size=inputs.size(0))
        if replay is not None:
            x_buf, y_buf, logits_buf = replay
            out_buf = model(x_buf)
            
            # DER: MSE loss between stored logits and current logits
            if logits_buf is not None and logits_buf.numel() > 0:
                der_loss = alpha_der * F.mse_loss(out_buf, logits_buf)
                loss = loss + der_loss
            
            # DER++: Additional classification loss
            if beta_der > 0:
                replay_ce_loss = beta_der * criterion(out_buf, y_buf.long())
                loss = loss + replay_ce_loss
        
        # Add proximal regularization
        prox_loss = proximal_loss(model, global_state, mu, device)
        loss = loss + prox_loss
        
        loss.backward()
        optimizer.step()
        
        # Add current batch to buffer with logits
        with torch.no_grad():
            logits = outputs.detach()
            buffer.add_batch(inputs, labels, logits)
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_with_prox_model(
    base_model: nn.Module,
    global_model: nn.Module,
    task_perm: Tuple[int, ...],
    train_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int,
    mu: float = 0.01,
    base_method: str = "ser",
    verbose: bool = False,
) -> nn.Module:
    """
    Train a local model with proximal regularization.
    
    Args:
        base_model: Model to clone and train
        global_model: Global model for proximal term
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Replay buffer capacity
        mu: Proximal regularization strength
        base_method: "ser" or "der"
        verbose: Print progress
    
    Returns:
        Trained local model
    """
    model = copy.deepcopy(base_model).to(device)
    global_state = {k: v.clone().detach() for k, v in global_model.state_dict().items()}
    
    buffer = ReplayBuffer(capacity=buffer_size, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
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
                
                # Replay based on method
                replay = buffer.sample(batch_size=inputs.size(0))
                if replay is not None:
                    x_buf, y_buf, logits_buf = replay
                    out_buf = model(x_buf)
                    
                    if base_method == "ser":
                        # SER: CE + distillation
                        replay_ce_loss = criterion(out_buf, y_buf.long())
                        if logits_buf is not None and logits_buf.numel() > 0:
                            temperature = 2.0
                            soft_targets = F.softmax(logits_buf / temperature, dim=1)
                            soft_outputs = F.log_softmax(out_buf / temperature, dim=1)
                            distill_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                            distill_loss = distill_loss * (temperature ** 2)
                            replay_loss = 0.5 * replay_ce_loss + 0.5 * distill_loss
                        else:
                            replay_loss = replay_ce_loss
                        loss = loss + replay_loss
                    else:
                        # DER++: MSE on logits + CE
                        if logits_buf is not None and logits_buf.numel() > 0:
                            der_loss = 0.5 * F.mse_loss(out_buf, logits_buf)
                            loss = loss + der_loss
                        replay_ce_loss = 0.5 * criterion(out_buf, y_buf.long())
                        loss = loss + replay_ce_loss
                
                # Add proximal regularization
                prox_loss = proximal_loss(model, global_state, mu, device)
                loss = loss + prox_loss
                
                loss.backward()
                optimizer.step()
                
                # Update buffer
                with torch.no_grad():
                    buffer.add_batch(inputs, labels, outputs.detach())
    
    return model


def fedprox_update(
    global_model: nn.Module,
    local_model: nn.Module,
    alpha: float = 0.5,
    verbose: bool = False,
) -> nn.Module:
    """
    Update global model using FedProx-style weight averaging.
    
    Same as FedAvg: W_g = (1 - alpha) * W_g + alpha * W_l
    The difference is in local training (proximal regularization).
    """
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()
    
    with torch.no_grad():
        for name in global_state:
            if name in local_state:
                global_state[name] = (1 - alpha) * global_state[name] + alpha * local_state[name]
    
    global_model.load_state_dict(global_state)
    
    if verbose:
        print(f"    FedProx update applied with alpha={alpha}")
    
    return global_model


class FedProxHierarchy:
    """
    Manages model hierarchy for FedProx-based consolidation.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_levels: int,
        device: str = "cuda"
    ):
        self.num_levels = num_levels
        self.device = device
        
        self.models = [
            copy.deepcopy(base_model).to(device)
            for _ in range(num_levels)
        ]
        
        self.level_alphas = [0.5 / (i + 1) for i in range(num_levels)]
    
    @property
    def local_model(self) -> nn.Module:
        return self.models[0]
    
    @property
    def global_model(self) -> nn.Module:
        return self.models[-1]
    
    def get_model(self, level: int) -> nn.Module:
        return self.models[level]
    
    def set_model(self, level: int, model: nn.Module):
        self.models[level] = model.to(self.device)
    
    def propagate_update(
        self,
        alpha: float = 0.5,
        verbose: bool = False
    ):
        """Propagate updates from local to global through all hierarchy levels."""
        for level in range(1, self.num_levels):
            effective_alpha = alpha * self.level_alphas[level]
            fedprox_update(
                self.models[level],
                self.models[level - 1],
                alpha=effective_alpha,
                verbose=verbose
            )


def select_best_permutation_fedprox(
    base_model: nn.Module,
    global_model: nn.Module,
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int,
    mu: float = 0.01,
    base_method: str = "ser",
) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Find the best task ordering within a group by exhaustive search.
    Uses proximal training.
    """
    k = len(train_loaders)
    best_acc = -float('inf')
    best_model = None
    best_perm = None
    
    for perm in permutations(range(k)):
        trained = train_with_prox_model(
            base_model, global_model, perm, train_loaders,
            num_epochs, lr, device, buffer_size, mu, base_method
        )
        
        accs = [evaluate(trained, val_loaders[i], device) for i in range(k)]
        avg_acc = sum(accs) / len(accs)
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model = copy.deepcopy(trained)
            best_perm = perm
    
    if best_model is None:
        best_model = copy.deepcopy(base_model).to(device)
        best_perm = tuple(range(k))
    
    return best_model, best_perm


def _canonicalize_perm(perm: Tuple[int, ...], group_size: int) -> Tuple[int, ...]:
    """Canonicalize permutation by sorting within groups."""
    n = len(perm)
    grouped = []
    for i in range(0, n, group_size):
        group = tuple(sorted(perm[i:i + group_size]))
        grouped.extend(group)
    return tuple(grouped)


def train_fedprox(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_levels: int = 2,
    group_size: int = 2,
    num_epochs: int = 30,
    lr: float = 0.01,
    buffer_size: int = 100,
    alpha: float = 0.5,
    mu: float = 0.01,
    perms: Optional[List[Tuple[int, ...]]] = None,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    base_method: str = "ser",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train using FedProx-style consolidation for continual learning.
    
    Args:
        model: Base model architecture
        train_loaders: Training data loaders per task
        test_loaders: Test data loaders per task
        num_levels: Number of hierarchy levels
        group_size: Tasks per group
        num_epochs: Training epochs
        lr: Learning rate
        buffer_size: Replay buffer size
        alpha: Aggregation interpolation factor
        mu: Proximal regularization strength
        perms: Task permutations to evaluate
        device: Device
        dataset: Dataset name
        output_dir: Output directory
        base_method: Base CL method ("ser" or "der")
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    
    if perms is None:
        from itertools import permutations as gen_perms
        perms = list(gen_perms(range(num_tasks)))[:4]
    
    method_name = f"FedProx-L{num_levels}"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name} + {base_method.upper()} experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  {num_levels} hierarchy levels, group_size={group_size}")
        print(f"  alpha={alpha}, mu={mu}")
        print(f"{'=' * 60}\n")
    
    results = []
    seen_canonical = set()
    canonical_to_results = {}
    canonical_to_time = {}
    
    for seq_id, perm in enumerate(perms, start=1):
        canonical = _canonicalize_perm(perm, group_size)
        
        if canonical in seen_canonical:
            results.append({
                "sequence": perm,
                "accuracies": canonical_to_results[canonical],
                "mean_acc": sum(canonical_to_results[canonical]) / num_tasks,
                "time_seconds": canonical_to_time.get(canonical, 0.0)
            })
            if verbose:
                print(f"{method_name} Perm {seq_id}/{len(perms)}: {perm} (cached)")
            continue
        
        seen_canonical.add(canonical)
        perm_start_time = time.time()
        
        if verbose:
            print(f"{method_name} Permutation {seq_id}/{len(perms)}: {perm}")
        
        # Initialize hierarchy
        hierarchy = FedProxHierarchy(model, num_levels, device)
        
        # Reorder loaders by permutation
        ordered_train = [train_loaders[i] for i in perm]
        ordered_test = [test_loaders[i] for i in perm]
        
        # Build task groups
        task_groups = [
            list(range(i, min(i + group_size, num_tasks)))
            for i in range(0, num_tasks, group_size)
        ]
        
        replay_datasets = []
        
        for g_idx, task_group in enumerate(task_groups):
            group_train = [ordered_train[i] for i in task_group]
            group_test = [ordered_test[i] for i in task_group]
            
            # Add replay to training
            group_train_with_replay = []
            for loader in group_train:
                datasets_list = [loader.dataset] + replay_datasets
                concat = ConcatDataset(datasets_list)
                new_loader = DataLoader(
                    concat,
                    batch_size=loader.batch_size if hasattr(loader, 'batch_size') else 32,
                    shuffle=True,
                    num_workers=0
                )
                group_train_with_replay.append(new_loader)
            
            # Find best permutation within group with proximal training
            local_base = copy.deepcopy(hierarchy.global_model).to(device)
            local_trained, best_local_perm = select_best_permutation_fedprox(
                local_base, hierarchy.global_model, group_train_with_replay, group_test,
                num_epochs, lr, device, buffer_size, mu, base_method
            )
            
            # Set local model
            hierarchy.set_model(0, local_trained)
            
            # FedProx update through hierarchy
            if g_idx == 0:
                # First group: initialize global from local
                for level in range(1, num_levels):
                    hierarchy.set_model(level, copy.deepcopy(local_trained))
            else:
                # Propagate through hierarchy
                hierarchy.propagate_update(alpha=alpha, verbose=verbose)
            
            # Update replay buffer
            replay_datasets.extend([l.dataset for l in group_train])
            random.shuffle(replay_datasets)
            if len(replay_datasets) > buffer_size:
                replay_datasets = replay_datasets[-buffer_size:]
            
            if verbose:
                accs = evaluate_all_tasks(hierarchy.global_model, ordered_test, device)
                print(f"  After group {g_idx}: {[f'{a:.1f}' for a in accs]}")
        
        # Final evaluation
        final_accs = evaluate_all_tasks(hierarchy.global_model, ordered_test, device)
        perm_elapsed_time = time.time() - perm_start_time
        
        canonical_to_results[canonical] = final_accs
        canonical_to_time[canonical] = perm_elapsed_time
        
        results.append({
            "sequence": perm,
            "accuracies": final_accs,
            "mean_acc": sum(final_accs) / len(final_accs),
            "time_seconds": perm_elapsed_time
        })
        
        if verbose:
            print(f"  Final: {[f'{a:.1f}' for a in final_accs]}")
            print(f"  Mean: {results[-1]['mean_acc']:.2f}%, Time: {perm_elapsed_time:.1f}s\n")
    
    total_elapsed_time = time.time() - total_start_time
    
    # Save results
    df = pd.DataFrame({
        "sequence": [str(r["sequence"]) for r in results],
        **{f"Task{t + 1}": [r["accuracies"][t] for r in results]
           for t in range(num_tasks)},
        "Mean": [r["mean_acc"] for r in results],
        "Time_s": [r.get("time_seconds", 0.0) for r in results]
    })
    
    paths = ensure_results_dirs(output_dir)
    
    csv_path = os.path.join(paths['csv'], f"fedprox_L{num_levels}_{base_method}_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    json_filename = f"fedprox_L{num_levels}_{base_method}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": f"FedProx-L{num_levels}+{base_method.upper()}",
        "dataset": dataset,
        "num_levels": num_levels,
        "alpha": alpha,
        "mu": mu,
        "total_time_seconds": total_elapsed_time,
        "mean_accuracy": float(df["Mean"].mean()),
        "std_accuracy": float(df["Mean"].std()),
        "per_task_mean": [float(df[f"Task{t + 1}"].mean()) for t in range(num_tasks)],
        "per_task_std": [float(df[f"Task{t + 1}"].std()) for t in range(num_tasks)],
    }
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    if verbose:
        print(f"Saved FedProx results to {csv_path}")
        print(f"Total FedProx time: {total_elapsed_time:.1f}s")
    
    return {
        "method": f"FedProx-L{num_levels}+{base_method.upper()}",
        "dataset": dataset,
        "num_levels": num_levels,
        "results": results,
        "total_time_seconds": total_elapsed_time,
        "summary": {
            "mean_accuracy": df["Mean"].mean(),
            "std_accuracy": df["Mean"].std(),
            "per_task_mean": [df[f"Task{t + 1}"].mean() for t in range(num_tasks)],
            "per_task_std": [df[f"Task{t + 1}"].std() for t in range(num_tasks)],
            "total_time": total_elapsed_time,
            "avg_time_per_perm": total_elapsed_time / max(len(seen_canonical), 1),
        }
    }


def run_fedprox_ser_experiments(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    perms: List[Tuple[int, ...]],
    buffer_size: int = 500,
    num_epochs: int = 30,
    lr: float = 0.01,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run FedProx + SER experiments."""
    return train_fedprox(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_levels=2,
        group_size=2,
        num_epochs=num_epochs,
        lr=lr,
        buffer_size=buffer_size,
        alpha=0.5,
        mu=0.01,
        perms=perms,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        base_method="ser",
        verbose=verbose,
    )


def run_fedprox_der_experiments(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    perms: List[Tuple[int, ...]],
    buffer_size: int = 500,
    num_epochs: int = 30,
    lr: float = 0.01,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run FedProx + DER experiments."""
    return train_fedprox(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_levels=2,
        group_size=2,
        num_epochs=num_epochs,
        lr=lr,
        buffer_size=buffer_size,
        alpha=0.5,
        mu=0.01,
        perms=perms,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        base_method="der",
        verbose=verbose,
    )
