import torch
from itertools import permutations
from utils import evaluate, clone_model


def train_local_model(base_model, task_perm, train_loaders, num_epochs, lr, device):
    model = clone_model(base_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for task_id in task_perm:
            for inputs, labels in train_loaders[task_id]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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


def linear_global_update(global_model, local_model, alpha=0.5):
    with torch.no_grad():
        for (name, g_param), (_, l_param) in zip(global_model.named_parameters(), local_model.named_parameters()):
            g_param.data = (1 - alpha) * g_param.data + alpha * l_param.data


def train_linear(model, task_train_loaders, task_test_loaders, group_size=2,
                 num_epochs=30, lr=0.001, alpha=0.5, device='cuda'):
    model = model.to(device)
    acc_per_task = []
    total_tasks = len(task_train_loaders)

    # Partition into task groups
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
            linear_global_update(model, local_model, alpha=alpha)

        current_task_id = max(task_group)
        accs = []
        for test_task_id in range(current_task_id + 1):
            acc = evaluate(model, task_test_loaders[test_task_id], device=device)
            accs.append(acc)
        acc_per_task.append(accs)

    return model, acc_per_task
