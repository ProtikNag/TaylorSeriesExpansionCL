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
    model = train_er_model(base_model, task_perm, train_loaders,
                           num_epochs, lr, device,
                           alpha=alpha, beta=beta, buffer_size=buffer_size)
    return model


def select_best_permutation(base_model, task_group_ids, train_loaders, val_loaders, num_epochs, lr, device):
    best_acc = -float('inf')
    best_model = None

    for perm in permutations(task_group_ids):
        local_model = train_local_model(base_model, perm, train_loaders, num_epochs, lr, device)
        accs = []
        for tid in task_group_ids:
            acc = evaluate(local_model, val_loaders[tid], device=device)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model = clone_model(local_model)

    return best_model


def taylor_global_update(global_model, local_model, train_loader, lambda_reg=100.0, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    global_model.train()
    local_model.eval()

    grads = {name: torch.zeros_like(param).to(device) for name, param in global_model.named_parameters()}

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        global_model.zero_grad()
        outputs = global_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in global_model.named_parameters():
            if param.grad is not None:
                grads[name] += param.grad.detach()

    for name in grads:
        grads[name] /= len(train_loader)

    hessians = estimate_diag_hessian_exact(global_model, train_loader, criterion, device)

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            eps = 1e-8
            h_inv = 1.0 / (hessians[name] + lambda_reg + eps)
            delta = h_inv * (lambda_reg * (local_model.state_dict()[name] - param) - grads[name])
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
                 num_epochs=30, lr=0.01, lambda_reg=100.0, device='cuda'):
    num_tasks = len(task_train_loaders)
    task_indices = list(range(num_tasks))
    results = []

    total_perms = math.factorial(num_tasks)
    print(f"=== Running Taylor-series experiment across {len(list(itertools.permutations(task_indices)))} permutations ===")

    seen = set()
    canonical_to_accuracies = {}
    processed_count = 0

    for seq_id, perm in enumerate(itertools.permutations(task_indices), start=1):
        canonical = _canonicalize_perm_by_group(perm, group_size)

        if canonical in seen:
            cached_accs = canonical_to_accuracies[canonical]
            results.append({"sequence": perm, "accuracies": cached_accs})
            print(f"Skipped computation for permutation #{seq_id} {perm} (canonical {canonical}) — reused results.")
            continue

        # Not seen: compute once for this canonical cluster
        seen.add(canonical)
        processed_count += 1

        # Clone fresh model 
        global_model = clone_model(model).to(device)

        print(f"\n--- Sequence {seq_id}: {perm} ---")

        # Reorder loaders by permutation
        ordered_train = [task_train_loaders[i] for i in perm]
        ordered_test = [task_test_loaders[i] for i in perm]

        # Standard Taylor training procedure
        replay_size = 1000
        replay_buffer = []
        acc_per_task = []

        task_groups = [list(range(i, min(i + group_size, num_tasks)))
                       for i in range(0, num_tasks, group_size)]

        for t, task_group in enumerate(task_groups):
            local_base_model = clone_model(global_model).to(device)
            local_trained = select_best_permutation(local_base_model, task_group,
                                                    ordered_train, ordered_test,
                                                    num_epochs, lr, device)

            combined_dataset = [ordered_train[i].dataset for i in task_group] + replay_buffer      # combined data is needed for global update calculation
            combined_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(combined_dataset),
                batch_size=64, shuffle=True
            )

            if t == 0:
                global_model.load_state_dict(local_trained.state_dict())
            else:
                global_model = taylor_global_update(global_model, local_trained,
                                     combined_loader, lambda_reg, device)

            replay_buffer.extend([ordered_train[i].dataset for i in task_group])
            random.shuffle(replay_buffer)
            if len(replay_buffer) > replay_size:
                replay_buffer = replay_buffer[-replay_size:]

            accs = [evaluate(global_model, ordered_test[tid], device=device)
                    for tid in range(max(task_group) + 1)]
            acc_per_task.append(accs)

        final_accs = acc_per_task[-1]
        canonical_to_accuracies[canonical] = final_accs
        results.append({"sequence": perm, "accuracies": acc_per_task[-1]})

    # Convert results into DataFrame
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t+1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })

    df.to_csv("./results/taylor_permutation_results.csv", index=False)
    print("Saved results to taylor_permutation_results.csv")

    _, _ = run_er_experiments(model, task_train_loaders, task_test_loaders, buffer_size=replay_size,
                                    num_epochs=num_epochs, lr=lr, device=device)

    return model, df
