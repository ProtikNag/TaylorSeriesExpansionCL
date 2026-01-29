"""
Hierarchical Taylor Series Continual Learning (HTCL) implementation.

This module implements the multi-level hierarchy with:
- Fast local adaptation
- Second-order Taylor series global consolidation
- Global model catch-up mechanism for recent tasks
- Configurable L-level hierarchy
- Configurable baseline method (ER, SER, DER, EWC, iCaRL)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import json
from datetime import datetime
from itertools import permutations, product
from typing import List, Tuple, Dict, Optional, Any, Callable
from torch.utils.data import DataLoader, ConcatDataset

from .er import train_er_model
from .ser import train_ser_model
from .der import train_der_model
from .ewc import train_ewc_model
from .icarl import train_icarl_model
from ..utils import (
    evaluate,
    evaluate_all_tasks,
    estimate_diag_hessian,
)
from ..utils.paths import ensure_results_dirs, get_csv_path, get_json_path

import gc


# Registry of baseline training functions
# Each function should have signature:
#   train_fn(base_model, task_perm, train_loaders, num_epochs, lr, device, buffer_size, **kwargs) -> model
BASELINE_TRAIN_REGISTRY: Dict[str, Callable] = {
    "er": train_er_model,
    "ser": train_ser_model,
    "der": train_der_model,
    "ewc": train_ewc_model,
    "icarl": train_icarl_model,
}


def get_baseline_train_fn(baseline: str) -> Callable:
    """
    Get the training function for a given baseline method.
    
    Args:
        baseline: Name of the baseline method (er, ser, der, ewc, icarl)
    
    Returns:
        Training function
    
    Raises:
        ValueError: If baseline is not recognized
    """
    baseline_lower = baseline.lower()
    if baseline_lower not in BASELINE_TRAIN_REGISTRY:
        available = list(BASELINE_TRAIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown baseline '{baseline}'. Available baselines: {available}"
        )
    return BASELINE_TRAIN_REGISTRY[baseline_lower]


def list_available_baselines() -> List[str]:
    """Return list of available baseline methods."""
    return list(BASELINE_TRAIN_REGISTRY.keys())


def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class HierarchicalModel:
    """
    Manages L-level model hierarchy for HTCL.
    
    Level 0 = local (most plastic)
    Level L-1 = global (most stable)
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

        # Per-level learning rates (decreasing plasticity)
        self.level_lrs = [1.0 / (2 ** i) for i in range(num_levels)]

    @property
    def local_model(self) -> nn.Module:
        """Get the local (most plastic) model."""
        return self.models[0]

    @property
    def global_model(self) -> nn.Module:
        """Get the global (most stable) model."""
        return self.models[-1]

    def get_model(self, level: int) -> nn.Module:
        """Get model at specific level."""
        return self.models[level]

    def set_model(self, level: int, model: nn.Module):
        """Set model at specific level."""
        self.models[level] = model.to(self.device)

    def propagate_update(
            self,
            train_loader: DataLoader,
            criterion: nn.Module,
            eta: float = 0.05,
            max_norm: float = 1.0,
            lambda_reg: float = 1000.0,
            verbose: bool = False
    ):
        """
        Propagate updates from local to global through all hierarchy levels.
        
        Each level i receives updates from level i-1 via Taylor expansion.
        """
        for level in range(1, self.num_levels):
            taylor_update(
                self.models[level],  # global at this level
                self.models[level - 1],  # local (from previous level)
                train_loader,
                self.device,
                eta=eta * self.level_lrs[level],
                max_norm=max_norm,
                lambda_reg=lambda_reg,
                verbose=verbose
            )


def taylor_update(
        global_model: nn.Module,
        local_model: nn.Module,
        train_loader: DataLoader,
        device: str = "cuda",
        eta: float = 0.05,
        max_norm: float = 1.0,
        lambda_reg: Optional[float] = None,
        verbose: bool = False,
) -> nn.Module:
    """
    Update global model using second-order Taylor expansion.
    
    Implements Eq. (3) from the paper:
    W_g^(t) = W_g^(t-1) + (H + λI)^(-1) [λ(W_l - W_g) - G]
    
    Args:
        global_model: Global model to update
        local_model: Local model providing the target
        train_loader: Data for computing gradients/Hessian
        device: Device
        eta: Step size
        max_norm: Maximum norm for update clipping
        lambda_reg: Regularization strength (auto-computed if None)
        verbose: Print debug info
    
    Returns:
        Updated global model
    """
    criterion = nn.CrossEntropyLoss()
    global_model.train()
    local_model.eval()

    # Accumulate gradients
    grads = {
        name: torch.zeros_like(param, device=device)
        for name, param in global_model.named_parameters()
        if param.requires_grad
    }

    n_batches = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        global_model.zero_grad()
        outputs = global_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in global_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] += param.grad.detach()

        n_batches += 1

    # Average gradients
    for name in grads:
        grads[name] /= max(n_batches, 1)

    # Estimate diagonal Hessian
    hessians = estimate_diag_hessian(global_model, train_loader, criterion, device)

    # Compute lambda_reg if not provided
    if lambda_reg is None:
        diag_entries = torch.cat([v.flatten() for v in hessians.values()])
        min_eig = float(diag_entries.min().item())
        max_eig = float(diag_entries.max().item())

        if not torch.isfinite(torch.tensor(max_eig)) or max_eig <= 0:
            lambda_reg = max(1e-6, 10.0 * (abs(min_eig) + 1e-6))
        else:
            lambda_reg = 1000.0 * max_eig

    # Apply Taylor update
    local_state = local_model.state_dict()

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if not param.requires_grad:
                continue

            h_diag = hessians[name].to(device)
            local_param = local_state[name].to(device)

            # Compute update: (H + λI)^(-1) [λ(W_l - W_g) - G]
            eps = 1e-8
            denom = h_diag + lambda_reg + eps
            h_inv = 1.0 / denom

            delta_D = local_param - param
            raw_delta = h_inv * (lambda_reg * delta_D - grads[name])

            # Scale and clip
            delta = eta * raw_delta
            delta_norm = delta.norm().item()
            if delta_norm > max_norm:
                delta = delta * (max_norm / (delta_norm + 1e-12))

            # Safety check
            if torch.isnan(delta).any() or torch.isinf(delta).any():
                if verbose:
                    print(f"  Warning: NaN/Inf in delta for {name}, skipping")
                continue

            param.add_(delta)

    return global_model


def global_catchup(
        global_model: nn.Module,
        combined_loader: DataLoader,
        num_iterations: int,
        device: str,
        eta: float = 0.1,
        max_norm: float = 1.0,
        lambda_reg: float = 500.0,
        verbose: bool = False,
) -> nn.Module:
    """
    Allow global model to catch up on recent tasks using Taylor updates.

    This addresses the issue where the global model lags behind
    on recently seen tasks because it uses conservative updates.

    Unlike backpropagation-based fine-tuning, this uses the same
    Taylor series update rule to maintain consistency with HTCL.

    Args:
        global_model: Global model to update
        combined_loader: DataLoader for recent tasks
        num_iterations: Number of Taylor update iterations
        device: Device
        eta: Step size for Taylor update (higher = more aggressive catch-up)
        max_norm: Maximum norm for update clipping
        lambda_reg: Regularization strength (lower = more aggressive toward local)
        verbose: Print progress

    Returns:
        Best performing global model across all iterations
    """
    if not combined_loader or num_iterations <= 0:
        return global_model

    # Track best model
    best_acc = -float('inf')
    best_model_state = copy.deepcopy(global_model.state_dict())

    # Evaluate initial model accuracy
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in combined_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = global_model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    initial_acc = 100.0 * correct / max(total, 1)
    best_acc = initial_acc

    if verbose:
        print(f"    Catchup initial accuracy: {initial_acc:.1f}%")

    for iteration in range(num_iterations):
        # Train a temporary local model on recent tasks (quick adaptation)
        # Start from current global model
        temp_local = copy.deepcopy(global_model).to(device)
        temp_optimizer = optim.SGD(temp_local.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Quick training pass on recent data
        temp_local.train()
        for inputs, labels in combined_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            temp_optimizer.zero_grad()
            outputs = temp_local(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            temp_optimizer.step()

        # Now use Taylor update to move global toward temp_local
        taylor_update(
            global_model,
            temp_local,
            combined_loader,
            device,
            eta=eta,
            max_norm=max_norm,
            lambda_reg=lambda_reg,
            verbose=False
        )

        # Evaluate current model accuracy
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in combined_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = global_model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        current_acc = 100.0 * correct / max(total, 1)

        # Update best model if current is better
        if current_acc > best_acc:
            best_acc = current_acc
            best_model_state = copy.deepcopy(global_model.state_dict())
            if verbose:
                print(f"    Catchup iteration {iteration + 1}/{num_iterations}, Acc: {current_acc:.1f}% (new best)")
        else:
            if verbose:
                print(f"    Catchup iteration {iteration + 1}/{num_iterations}, Acc: {current_acc:.1f}%")

    # Restore best model
    global_model.load_state_dict(best_model_state)

    if verbose:
        print(f"    Catchup complete. Best accuracy: {best_acc:.1f}%")

    return global_model


def select_best_permutation(
        base_model: nn.Module,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        num_epochs: int,
        lr: float,
        device: str,
        buffer_size: int,
        baseline: str = "er",
) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Find the best task ordering within a group by exhaustive search.
    
    Args:
        base_model: Starting model
        train_loaders: Training loaders for tasks in this group
        val_loaders: Validation loaders for evaluation
        num_epochs: Training epochs per permutation
        lr: Learning rate
        device: Device
        buffer_size: Replay buffer size
        baseline: Baseline method to use (er, ser, der, ewc, icarl)
    
    Returns:
        (best_model, best_permutation)
    """
    k = len(train_loaders)
    best_acc = -float('inf')
    best_model = None
    best_perm = None
    
    # Get the appropriate training function for this baseline
    train_fn = get_baseline_train_fn(baseline)

    for perm in permutations(range(k)):
        # Train on this permutation using the selected baseline
        trained = train_fn(
            base_model, perm, train_loaders,
            num_epochs, lr, device, buffer_size
        )

        # Evaluate
        accs = [evaluate(trained, val_loaders[i], device) for i in range(k)]
        avg_acc = sum(accs) / len(accs)

        if avg_acc > best_acc:
            best_acc = avg_acc
            if best_model is not None:
                del best_model
            best_model = copy.deepcopy(trained)
            best_perm = perm

        # Clean up
        del trained
        clear_memory()

    if best_model is None:
        best_model = copy.deepcopy(base_model).to(device)
        best_perm = tuple(range(k))

    return best_model, best_perm


def _canonicalize_perm(perm: Tuple[int, ...], group_size: int) -> Tuple[int, ...]:
    """
    Canonicalize permutation by sorting within groups.
    
    This identifies equivalent permutations under HTCL's grouping.
    """
    n = len(perm)
    grouped = []
    for i in range(0, n, group_size):
        group = tuple(sorted(perm[i:i + group_size]))
        grouped.extend(group)
    return tuple(grouped)


def _intra_group_variants(perm: Tuple[int, ...], group_size: int):
    """
    Given a permutation `perm`, yield all permutations that are canonical-equivalent
    by permuting elements *within each group position*.
    """
    n = len(perm)
    groups = [perm[i:i + group_size] for i in range(0, n, group_size)]
    group_perms = [list(permutations(g)) for g in groups]
    for combo in product(*group_perms):
        # flatten tuple-of-tuples into one tuple
        yield tuple(x for g in combo for x in g)


def generate_canonical_permutations(
        num_tasks: int,
        group_size: int = 2,
        max_perms: int = 20,
        seed: int = 42,
) -> List[Tuple[int, ...]]:
    """
    Simple implementation per your procedure:
      1) take a random permutation of num_tasks
      2) generate all its canonical permutations (permute within groups)
      3) add them one-by-one to the result until max_perms
      4) if still short, pick another random permutation that yields new items
      5) repeat until result has max_perms or we hit max attempts

    Returns a list (length <= max_perms) of permutations (tuples).
    """
    rnd = random.Random(seed)
    task_indices = list(range(num_tasks))
    seen = set()  # canonical forms we've already produced (optional)
    added = set()  # actual permutations added to result
    result: List[Tuple[int, ...]] = []

    max_attempts = max_perms * 200  # safety cap
    attempts = 0

    while len(result) < max_perms and attempts < max_attempts:
        attempts += 1
        # pick a random permutation
        base = tuple(rnd.sample(task_indices, num_tasks))

        # iterate all intra-group variants of this base permutation
        for variant in _intra_group_variants(base, group_size):
            if len(result) >= max_perms:
                break
            # optional: skip if exact variant already added
            if variant in added:
                continue
            # optional: skip if its canonical form already fully covered?
            # (you said to allow canonically similar permutations, so we only skip exact duplicates)
            added.add(variant)
            result.append(variant)

        # small optimization: if this base produced nothing new, continue to next sample
        # loop will exit when result has enough items

    return result


def train_htcl(
        model: nn.Module,
        train_loaders: List[DataLoader],
        test_loaders: List[DataLoader],
        num_levels: int = 2,
        group_size: int = 2,
        num_epochs: int = 30,
        lr: float = 0.01,
        buffer_size: int = 100,
        perms: Optional[List[Tuple[int, ...]]] = None,
        device: str = "cuda",
        dataset: str = "Unknown",
        output_dir: str = "./results",
        baseline: str = "er",
        catchup_enabled: bool = True,
        catchup_epochs: int = 2,
        catchup_lr_factor: float = 0.1,
        eta: float = 1.0,
        max_norm: float = 1.0,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train using Hierarchical Taylor Series Continual Learning.
    
    Args:
        model: Base model architecture
        train_loaders: Training data loaders per task
        test_loaders: Test data loaders per task
        num_levels: Number of hierarchy levels (L)
        group_size: Tasks per group (k)
        num_epochs: Training epochs
        lr: Learning rate
        buffer_size: Replay buffer size
        perms: Task permutations to evaluate
        device: Device
        dataset: Dataset name
        output_dir: Output directory
        baseline: Baseline CL method to use (er, ser, der, ewc, icarl)
        catchup_enabled: Enable global model catch-up
        catchup_epochs: Catch-up training epochs
        catchup_lr_factor: LR multiplier for catch-up
        eta: Taylor update step size
        max_norm: Gradient clipping norm
        verbose: Print progress
    
    Returns:
        Results dictionary with timing information
    """
    import pandas as pd
    import os
    import time

    # Validate baseline
    baseline_lower = baseline.lower()
    if baseline_lower not in BASELINE_TRAIN_REGISTRY:
        available = list(BASELINE_TRAIN_REGISTRY.keys())
        raise ValueError(f"Unknown baseline '{baseline}'. Available: {available}")

    total_start_time = time.time()
    num_tasks = len(train_loaders)

    if perms is None:
        from itertools import permutations as gen_perms
        perms = list(gen_perms(range(num_tasks)))[:4]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running HTCL experiments on {dataset}")
        print(f"  Baseline method: {baseline_lower.upper()}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  {num_levels} hierarchy levels, group_size={group_size}")
        print(f"  catchup_enabled={catchup_enabled}, catchup_epochs={catchup_epochs}")
        print(f"{'=' * 60}\n")

    results = []
    seen_canonical = set()
    canonical_to_results = {}
    canonical_to_time = {}  # Track time per canonical form

    for seq_id, perm in enumerate(perms, start=1):
        canonical = _canonicalize_perm(perm, group_size)

        # Skip if we've seen this canonical form
        if canonical in seen_canonical:
            results.append({
                "sequence": perm,
                "accuracies": canonical_to_results[canonical],
                "mean_acc": sum(canonical_to_results[canonical]) / num_tasks,
                "time_seconds": canonical_to_time.get(canonical, 0.0)
            })
            if verbose:
                print(f"HTCL Perm {seq_id}/{len(perms)}: {perm} (cached)")
            continue

        seen_canonical.add(canonical)
        perm_start_time = time.time()

        if verbose:
            print(f"HTCL Permutation {seq_id}/{len(perms)}: {perm}")

        # Initialize hierarchy
        hierarchy = HierarchicalModel(model, num_levels, device)

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

            # Find best permutation within group using the selected baseline
            local_base = copy.deepcopy(hierarchy.global_model).to(device)
            local_trained, best_local_perm = select_best_permutation(
                local_base, group_train_with_replay, group_test,
                num_epochs, lr, device, buffer_size,
                baseline=baseline_lower
            )

            # Set local model
            hierarchy.set_model(0, local_trained)

            # Build combined loader for Taylor update
            combined_datasets = [l.dataset for l in group_train] + replay_datasets
            combined_loader = DataLoader(
                ConcatDataset(combined_datasets),
                batch_size=64, shuffle=True, num_workers=0
            )

            # Taylor update through hierarchy
            if g_idx == 0:
                # First group: initialize global from local
                for level in range(1, num_levels):
                    hierarchy.set_model(level, copy.deepcopy(local_trained))
            else:
                # Propagate through hierarchy
                hierarchy.propagate_update(
                    combined_loader,
                    nn.CrossEntropyLoss(),
                    eta=eta,
                    max_norm=max_norm,
                    verbose=verbose
                )

            # Global model catch-up on recent tasks using Taylor updates
            if catchup_enabled and g_idx > 0:
                global_catchup(
                    hierarchy.global_model,
                    combined_loader,
                    num_iterations=catchup_epochs,
                    device=device,
                    eta=0.999,  # More aggressive for catch-up
                    max_norm=max_norm,
                    lambda_reg=500.0,  # Higher reg = move more toward local
                    verbose=verbose
                )

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

    # Ensure proper directory structure
    paths = ensure_results_dirs(output_dir)

    # Save CSV to csv/ subdirectory
    csv_path = os.path.join(
        paths['csv'], 
        f"htcl_{baseline_lower}_L{num_levels}_results_{dataset}.csv"
    )
    df.to_csv(csv_path, index=False)

    # Save JSON summary to json/ subdirectory
    json_filename = f"htcl_{baseline_lower}_L{num_levels}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": f"HTCL-L{num_levels}",
        "baseline": baseline_lower,
        "dataset": dataset,
        "num_levels": num_levels,
        "total_time_seconds": total_elapsed_time,
        "mean_accuracy": float(df["Mean"].mean()),
        "std_accuracy": float(df["Mean"].std()),
        "per_task_mean": [float(df[f"Task{t + 1}"].mean()) for t in range(num_tasks)],
        "per_task_std": [float(df[f"Task{t + 1}"].std()) for t in range(num_tasks)],
    }
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)

    if verbose:
        print(f"Saved HTCL results to {csv_path}")
        print(f"Total HTCL time: {total_elapsed_time:.1f}s")

    return {
        "method": f"HTCL-L{num_levels}",
        "baseline": baseline_lower,
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
