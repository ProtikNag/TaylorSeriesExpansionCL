# taylor.py
import torch
import torch.nn as nn
from itertools import permutations
from utils import evaluate, estimate_diag_hessian_exact, clone_model
import random
import itertools
import pandas as pd
import math
from methods.er import train_er_model, run_er_experiments


def train_local_model(base_model, task_perm, train_loaders, num_epochs, lr, device,
                      alpha=0.5, beta=0.5, buffer_size=500):
    """
    Wrapper that trains a (local) ER model on the provided sequence `task_perm`.
    `train_loaders` should be a list of DataLoader objects whose indices are addressed
    by values in `task_perm` (i.e. task_perm contains indices into train_loaders).
    """
    model = train_er_model(base_model, task_perm, train_loaders,
                           num_epochs, lr, device,
                           alpha=alpha, beta=beta, buffer_size=buffer_size)
    return model


def select_best_permutation(base_model, group_train_loaders, group_val_loaders,
                            num_epochs, lr, device):
    """
    Find the best ordering (permutation) of the tasks in a small group by training
    an ER model for each permutation and selecting the one with highest average
    validation accuracy across the group's validation loaders.

    Inputs:
      - group_train_loaders: list of DataLoader objects for this group (length k)
      - group_val_loaders: list of DataLoader objects for this group (length k)
    Returns:
      - best_model (cloned & on device)
    """
    k = len(group_train_loaders)
    if k == 0:
        raise ValueError("group must contain at least one loader")

    best_acc = -float('inf')
    best_model = None

    # permutations over group-local indices: 0..k-1
    for perm in permutations(range(k)):
        local_model = train_local_model(base_model, perm, group_train_loaders,
                                        num_epochs, lr, device)
        local_model.to(device)
        accs = []
        for tid in range(k):
            acc = evaluate(local_model, group_val_loaders[tid], device=device)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model = clone_model(local_model).to(device)

    # In case all permutations failed for some reason, return a clone of base_model
    if best_model is None:
        best_model = clone_model(base_model).to(device)

    return best_model


def taylor_global_update(global_model, local_model, train_loader,
                         lambda_reg=10.0, device='cuda', eta=0.05, max_norm=1.0,
                         verbose=False):
    """
    Compute a damped, diagonal-approx Newton-like update to global_model
    that moves it toward local_model while taking curvature (diag Hessian) and
    gradients into account.

    - lambda_reg: damping/regularization added to diag(H)
    - eta: global multiplicative step size (learning-rate-like)
    - max_norm: clip per-parameter delta norm to this value
    """
    criterion = nn.CrossEntropyLoss()
    global_model.train()
    local_model.eval()

    # Initialize gradient accumulators (on device)
    grads = {name: torch.zeros_like(param, device=device) for name, param in global_model.named_parameters()}

    # Accumulate gradients over train_loader (sum of grads across batches)
    num_batches = 0
    for inputs, labels in train_loader:
        num_batches += 1
        inputs, labels = inputs.to(device), labels.to(device)
        global_model.zero_grad()
        outputs = global_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in global_model.named_parameters():
            if param.grad is not None:
                grads[name] += param.grad.detach()

    if num_batches == 0:
        # Nothing to update on — just return the global model unchanged
        if verbose:
            print("taylor_global_update: train_loader had 0 batches; skip update.")
        return global_model

    # Average gradients per batch (keeps scale consistent)
    for name in grads:
        grads[name] /= float(num_batches)

    # Estimate diagonal Hessian (dictionary mapping param name -> tensor of same shape)
    hessians = estimate_diag_hessian_exact(global_model, train_loader, criterion, device)

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            # get diag-hessian for this parameter, ensure device match
            if name not in hessians:
                if verbose:
                    print(f"Warning: no hessian entry for {name}; skipping update for this param.")
                continue

            h_diag = hessians[name].to(device)

            # Enforce non-negativity (simple PD approximation) and add damping
            # Negative curvature can cause sign flips; clamp to >= 0
            h_diag = h_diag.clamp(min=0.0)

            eps = 1e-8
            denom = h_diag + lambda_reg + eps
            h_inv = 1.0 / denom

            # Ensure local param is on the correct device & dtype
            local_param = local_model.state_dict()[name].to(device)

            # Compute delta (Newton-like direction with gradient + quadratic regularizer)
            # delta = H^{-1} * ( lambda_reg * (local - global) - grad )
            raw_delta = h_inv * (lambda_reg * (local_param - param) - grads[name])

            # scale (eta) and clip per-parameter norm
            delta = eta * raw_delta
            delta_norm = delta.norm().item() if delta.numel() > 0 else 0.0
            if delta_norm > max_norm:
                delta = delta * (max_norm / (delta_norm + 1e-12))

            # Safety: avoid NaNs/Infs
            if torch.isnan(delta).any() or torch.isinf(delta).any():
                if verbose:
                    print(f"NaN/Inf detected in delta for {name}; zeroing this delta.")
                delta = torch.zeros_like(delta)

            # Debug prints (toggle with verbose)
            if verbose and (name.endswith('.weight') or name.endswith('.bias')):
                print(f"[taylor update] {name}: h_diag min/max {h_diag.min().item():.3e}/{h_diag.max().item():.3e}, "
                      f"grad norm {grads[name].norm().item():.3e}, delta norm {delta.norm().item():.3e}")

            # Apply update
            param.add_(delta)

    return global_model


def _canonicalize_perm_by_group(perm, group_size):
    """
    Return a canonical version of perm where each contiguous group of length group_size
    is sorted internally (last group may be shorter and is sorted too).
    This is used to treat intra-group permutations as equivalent.
    """
    n = len(perm)
    grouped = []
    for i in range(0, n, group_size):
        group = tuple(sorted(perm[i:i+group_size]))
        grouped.extend(group)
    return tuple(grouped)


def train_taylor(model, task_train_loaders, task_test_loaders, group_size=2,
                 num_epochs=30, lr=0.01, lambda_reg=10.0, device='cuda',
                 eta=0.05, max_norm=1.0, verbose=False):
    """
    Main driver for Taylor-style global updates on top of ER local training.

    - model: initial model (untrained / base)
    - task_train_loaders, task_test_loaders: lists (length = num_tasks) of DataLoader
      objects in canonical task order.
    - group_size: how many tasks to group together for local permutation search
    """
    num_tasks = len(task_train_loaders)
    if num_tasks == 0:
        raise ValueError("No tasks provided")

    task_indices = list(range(num_tasks))
    results = []

    perms = list(itertools.permutations(task_indices))
    print(f"=== Running Taylor-series experiment across {len(perms)} permutations ===")

    seen = set()
    canonical_to_accuracies = {}
    processed_count = 0

    for seq_id, perm in enumerate(perms, start=1):
        canonical = _canonicalize_perm_by_group(perm, group_size)

        if canonical in seen:
            cached_accs = canonical_to_accuracies[canonical]
            results.append({"sequence": perm, "accuracies": cached_accs})
            if verbose:
                print(f"Skipped permutation #{seq_id} {perm} (canonical {canonical}) — reused results.")
            continue

        # Not seen: compute once for this canonical cluster
        seen.add(canonical)
        processed_count += 1

        # Fresh copy of the global model for this permutation
        global_model = clone_model(model).to(device)

        print(f"\n--- Sequence {seq_id}: {perm} ---")

        # Reorder loaders by permutation (these are the permuted task order)
        ordered_train = [task_train_loaders[i] for i in perm]
        ordered_test = [task_test_loaders[i] for i in perm]

        # Standard Taylor training procedure
        replay_size = 1000
        replay_buffer = []            # stores dataset objects (ConcatDataset will combine them)
        acc_per_task = []

        # Build groups by positions in the permuted order (0..num_tasks-1)
        task_groups = [list(range(i, min(i + group_size, num_tasks)))
                       for i in range(0, num_tasks, group_size)]

        for t, task_group in enumerate(task_groups):
            # Build group-local loader lists (indices are 0..k-1 for the group)
            group_train_loaders = [ordered_train[i] for i in task_group]
            group_val_loaders = [ordered_test[i] for i in task_group]

            # Local search over permutations inside the group (returns model on device)
            local_base_model = clone_model(global_model).to(device)
            local_trained = select_best_permutation(local_base_model, group_train_loaders, group_val_loaders,
                                                    num_epochs, lr, device)
            local_trained.to(device)

            # Build combined dataset for global update: group's datasets + replay buffer datasets
            combined_dataset = [g.dataset for g in group_train_loaders] + list(replay_buffer)
            if len(combined_dataset) == 0:
                # As a fallback, if combined_dataset is empty, use the group's datasets
                combined_dataset = [g.dataset for g in group_train_loaders]

            combined_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(combined_dataset),
                batch_size=64, shuffle=True
            )

            # First group: initialize global_model to local trained (warm start)
            if t == 0:
                global_model.load_state_dict(local_trained.state_dict())
            else:
                global_model = taylor_global_update(global_model, local_trained, combined_loader,
                                                   lambda_reg=lambda_reg, device=device,
                                                   eta=eta, max_norm=max_norm, verbose=verbose)

            # Update replay buffer with datasets from this group
            replay_buffer.extend([g.dataset for g in group_train_loaders])
            random.shuffle(replay_buffer)
            if len(replay_buffer) > replay_size:
                replay_buffer = replay_buffer[-replay_size:]

            # Evaluate global model on *all tasks in the permuted order* so
            # we always store a length-num_tasks accuracies vector.
            accs_all = [evaluate(global_model, ordered_test[i], device=device) for i in range(num_tasks)]
            acc_per_task.append(accs_all)

            if verbose:
                seen_so_far = sum(len(g) for g in task_groups[: t+1])
                print(f"After group {t} (saw {seen_so_far} tasks), accuracies (per permuted task): {accs_all}")

        # final accuracies for this permutation (last recorded all-task evaluation)
        final_accs = acc_per_task[-1] if len(acc_per_task) > 0 else [evaluate(global_model, ordered_test[i], device=device) for i in range(num_tasks)]
        canonical_to_accuracies[canonical] = final_accs
        results.append({"sequence": perm, "accuracies": final_accs})

    # Convert results into DataFrame (each row corresponds to a permutation sequence;
    # columns Task1..TaskN correspond to accuracies on tasks in the permuted order)
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t+1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })

    # Ensure results dir exists (best-effort)
    try:
        df.to_csv("./results/taylor_permutation_results.csv", index=False)
        print("Saved results to ./results/taylor_permutation_results.csv")
    except Exception as e:
        print("Could not save results CSV:", e)

    # Optionally run ER baseline experiments and save their csv as well
    try:
        _, _ = run_er_experiments(model, task_train_loaders, task_test_loaders,
                                  buffer_size=replay_size, num_epochs=num_epochs, lr=lr, device=device)
    except Exception as e:
        print("run_er_experiments failed or returned error:", e)

    return global_model, df
