"""
FedAvg-style consolidation for continual learning.

FedAvg (Federated Averaging) uses simple weight averaging to consolidate
local model updates into a global model:
    W_g = (1 - alpha) * W_g + alpha * W_l

In the CL context:
- Local model: Adapts quickly to current task group
- Global model: Maintains aggregated knowledge via averaging

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
           from Decentralized Data", AISTATS 2017
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
from .ser import train_ser_model
from .der import train_der_model
from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


def fedavg_update(
    global_model: nn.Module,
    local_model: nn.Module,
    alpha: float = 0.5,
    verbose: bool = False,
) -> nn.Module:
    """
    Update global model using FedAvg-style weight averaging.
    
    W_g = (1 - alpha) * W_g + alpha * W_l
    
    Args:
        global_model: Global model to update
        local_model: Local model providing the new weights
        alpha: Interpolation factor (0 = keep global, 1 = replace with local)
        verbose: Print debug info
    
    Returns:
        Updated global model
    """
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()
    
    with torch.no_grad():
        for name in global_state:
            if name in local_state:
                # Simple weighted average
                global_state[name] = (1 - alpha) * global_state[name] + alpha * local_state[name]
    
    global_model.load_state_dict(global_state)
    
    if verbose:
        print(f"    FedAvg update applied with alpha={alpha}")
    
    return global_model


class FedAvgHierarchy:
    """
    Manages model hierarchy for FedAvg-based consolidation.
    
    Similar to HTCL's HierarchicalModel but uses FedAvg for consolidation.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_levels: int,
        device: str = "cuda"
    ):
        self.num_levels = num_levels
        self.device = device
        
        # Create hierarchy: level 0 is local, level L-1 is global
        self.models = [
            copy.deepcopy(base_model).to(device)
            for _ in range(num_levels)
        ]
        
        # Per-level averaging factors (decreasing contribution from lower levels)
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
        """
        Propagate updates from local to global through all hierarchy levels.
        """
        for level in range(1, self.num_levels):
            # Effective alpha decreases for higher levels
            effective_alpha = alpha * self.level_alphas[level]
            fedavg_update(
                self.models[level],
                self.models[level - 1],
                alpha=effective_alpha,
                verbose=verbose
            )


def select_best_permutation_fedavg(
    base_model: nn.Module,
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int,
    base_method: str = "ser",
) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Find the best task ordering within a group by exhaustive search.
    """
    k = len(train_loaders)
    best_acc = -float('inf')
    best_model = None
    best_perm = None
    
    train_fn = train_ser_model if base_method == "ser" else train_der_model
    
    for perm in permutations(range(k)):
        trained = train_fn(
            base_model, perm, train_loaders,
            num_epochs, lr, device, buffer_size
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


def train_fedavg(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_levels: int = 2,
    group_size: int = 2,
    num_epochs: int = 30,
    lr: float = 0.01,
    buffer_size: int = 100,
    alpha: float = 0.5,
    perms: Optional[List[Tuple[int, ...]]] = None,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    base_method: str = "ser",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train using FedAvg-style consolidation for continual learning.
    
    Args:
        model: Base model architecture
        train_loaders: Training data loaders per task
        test_loaders: Test data loaders per task
        num_levels: Number of hierarchy levels
        group_size: Tasks per group
        num_epochs: Training epochs
        lr: Learning rate
        buffer_size: Replay buffer size
        alpha: FedAvg interpolation factor
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
    
    method_name = f"FedAvg-L{num_levels}"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name} + {base_method.upper()} experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  {num_levels} hierarchy levels, group_size={group_size}")
        print(f"  alpha={alpha}")
        print(f"{'=' * 60}\n")
    
    results = []
    seen_canonical = set()
    canonical_to_results = {}
    canonical_to_time = {}
    
    train_fn = train_ser_model if base_method == "ser" else train_der_model
    
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
        hierarchy = FedAvgHierarchy(model, num_levels, device)
        
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
            
            # Find best permutation within group
            local_base = copy.deepcopy(hierarchy.global_model).to(device)
            local_trained, best_local_perm = select_best_permutation_fedavg(
                local_base, group_train_with_replay, group_test,
                num_epochs, lr, device, buffer_size, base_method
            )
            
            # Set local model
            hierarchy.set_model(0, local_trained)
            
            # FedAvg update through hierarchy
            if g_idx == 0:
                # First group: initialize global from local
                for level in range(1, num_levels):
                    hierarchy.set_model(level, copy.deepcopy(local_trained))
            else:
                # Propagate through hierarchy using FedAvg
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
    
    csv_path = os.path.join(paths['csv'], f"fedavg_L{num_levels}_{base_method}_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    json_filename = f"fedavg_L{num_levels}_{base_method}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": f"FedAvg-L{num_levels}+{base_method.upper()}",
        "dataset": dataset,
        "num_levels": num_levels,
        "alpha": alpha,
        "total_time_seconds": total_elapsed_time,
        "mean_accuracy": float(df["Mean"].mean()),
        "std_accuracy": float(df["Mean"].std()),
        "per_task_mean": [float(df[f"Task{t + 1}"].mean()) for t in range(num_tasks)],
        "per_task_std": [float(df[f"Task{t + 1}"].std()) for t in range(num_tasks)],
    }
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    if verbose:
        print(f"Saved FedAvg results to {csv_path}")
        print(f"Total FedAvg time: {total_elapsed_time:.1f}s")
    
    return {
        "method": f"FedAvg-L{num_levels}+{base_method.upper()}",
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


def run_fedavg_ser_experiments(
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
    """Run FedAvg + SER experiments."""
    return train_fedavg(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_levels=2,
        group_size=2,
        num_epochs=num_epochs,
        lr=lr,
        buffer_size=buffer_size,
        alpha=0.5,
        perms=perms,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        base_method="ser",
        verbose=verbose,
    )


def run_fedavg_der_experiments(
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
    """Run FedAvg + DER experiments."""
    return train_fedavg(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_levels=2,
        group_size=2,
        num_epochs=num_epochs,
        lr=lr,
        buffer_size=buffer_size,
        alpha=0.5,
        perms=perms,
        device=device,
        dataset=dataset,
        output_dir=output_dir,
        base_method="der",
        verbose=verbose,
    )
