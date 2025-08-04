# utils.py

import torch
import numpy as np
import torch.nn as nn
import os
import pickle
import torch


def save_model_and_metrics(name, model, acc_matrix, tag=""):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = f"checkpoints/{name}_{tag}.pt"
    metric_path = f"results/{name}_{tag}.pkl"

    torch.save(model.state_dict(), model_path)
    with open(metric_path, 'wb') as f:
        pickle.dump(acc_matrix, f)


def load_model_and_metrics(name, model_class, num_classes, tag=""):
    model_path = f"checkpoints/{name}_{tag}.pt"
    metric_path = f"results/{name}_{tag}.pkl"

    if not (os.path.exists(model_path) and os.path.exists(metric_path)):
        return None, None

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    with open(metric_path, 'rb') as f:
        acc_matrix = pickle.load(f)
    return model, acc_matrix


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # shape: [batch_size, 10]
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    return acc


def compute_avg_accuracy(acc_matrix):
    """Compute average accuracy after each task"""
    return np.mean([row[-1] for row in acc_matrix])


def compute_avg_forgetting(acc_matrix):
    """
    Computes average forgetting per task:
    F_k = max_{t < T} acc[t][k] - acc[T][k]
    where acc[t][k] is accuracy on task k after training on task t (t >= k).
    """
    num_tasks = len(acc_matrix)
    forgetting = []

    for task_id in range(num_tasks):
        accs_on_task = [acc_matrix[t][task_id] for t in range(task_id, len(acc_matrix)) if task_id < len(acc_matrix[t])]
        if len(accs_on_task) >= 2:
            max_prev = max(accs_on_task[:-1])
            last = accs_on_task[-1]
            forgetting.append(max_prev - last)

    return np.mean(forgetting) if forgetting else 0.0


def estimate_diag_hessian_exact(model, data_loader, criterion, device='cuda'):
    model.eval()
    hessian_diag = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if p.requires_grad}

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # First-order gradients (retain graph for second-order)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        for i, (name, param) in enumerate(model.named_parameters()):
            if not param.requires_grad:
                continue

            grad = grads[i]
            grad2 = torch.autograd.grad(grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
            hessian_diag[name] += grad2.detach()

    for name in hessian_diag:
        hessian_diag[name] /= len(data_loader)

    return hessian_diag


def estimate_diag_hessian(model, data_loader, criterion, device='cuda'):
    """
    Diagonal approximation of Hessian using squared gradients (like in EWC).
    """
    model.eval()
    hessian_diag = {}
    for name, param in model.named_parameters():
        hessian_diag[name] = torch.zeros_like(param.data).to(device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=True)

        for (name, param), grad in zip(model.named_parameters(), grads):
            hessian_diag[name] += grad.pow(2).detach()

    for name in hessian_diag:
        hessian_diag[name] /= len(data_loader)

    return hessian_diag


def clone_model(model):
    """Deep copy model with parameters (detached)"""
    import copy
    return copy.deepcopy(model)
