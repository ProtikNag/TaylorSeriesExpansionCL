import torch
import torch.nn as nn
import torch.optim as optim
from itertools import permutations
from tqdm import tqdm
from utils import evaluate, estimate_diag_hessian, clone_model


def train_local_model(base_model, task_perm, train_loaders, num_epochs, lr, device):
    model = clone_model(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        for task_id in task_perm:
            for inputs, labels in train_loaders[task_id]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # labels already remapped to [0–9]
                loss.backward()
                optimizer.step()
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

    hessians = estimate_diag_hessian(global_model, train_loader, criterion, device)

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            eps = 1e-8
            h_inv = 1.0 / (hessians[name] + lambda_reg + eps)
            delta = h_inv * (lambda_reg * (local_model.state_dict()[name] - param) - grads[name])
            param.add_(delta)


def train_taylor(model, task_train_loaders, task_test_loaders, group_size=2,
                 num_epochs=30, lr=0.001, lambda_reg=100.0, device='cuda'):
    model = model.to(device)
    acc_per_task = []
    total_tasks = len(task_train_loaders)

    # Partition tasks into groups
    task_groups = []
    start = 0
    while start < total_tasks:
        end = min(start + group_size, total_tasks)
        task_groups.append(list(range(start, end)))
        start = end

    for t, task_group in enumerate(task_groups):
        print(f"\n=== Training Group {t} with Tasks {task_group} ===")

        local_model = select_best_permutation(model, task_group, task_train_loaders, task_train_loaders,
                                              num_epochs, lr, device)

        if t == 0:
            model.load_state_dict(local_model.state_dict())
        else:
            combined_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([task_train_loaders[i].dataset for i in task_group]),
                batch_size=64, shuffle=True
            )
            taylor_global_update(model, local_model, combined_loader, lambda_reg, device)

        accs = []
        for test_task_id in range(max(task_group) + 1):
            acc = evaluate(model, task_test_loaders[test_task_id], device=device)
            accs.append(acc)
        acc_per_task.append(accs)

    return model, acc_per_task
