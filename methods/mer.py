# mer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import copy
from models import get_model
from data import ContinualCIFAR100


class ReplayBuffer:
    def __init__(self, capacity=5000, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []  # list of (tensor, label)

    def add_samples(self, samples):
        """
        Add (x,y) pairs to buffer using reservoir sampling.
        samples: list of (tensor, label)
        """
        for x, y in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append((x.clone().cpu(), y.clone().cpu()))
            else:
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = (x.clone().cpu(), y.clone().cpu())

    def sample(self, current_x, current_y, k=10):
        """
        Return batch with 1 current sample + (k-1) memory samples.
        Shapes: current_x [3,32,32], current_y scalar
        """
        batch_x, batch_y = [current_x.unsqueeze(0)], [current_y.unsqueeze(0)]
        if len(self.buffer) > 0:
            samples = random.sample(self.buffer, min(k - 1, len(self.buffer)))
            for x, y in samples:
                batch_x.append(x.unsqueeze(0))
                batch_y.append(torch.tensor([y]))
        xb = torch.cat(batch_x).to(self.device)  # [B,3,32,32]
        yb = torch.cat(batch_y).to(self.device)  # [B]
        return xb, yb


def mer_update(model, criterion, current_x, current_y, buffer,
               s=5, k=10, alpha=0.01, beta=1.0, device="cuda"):
    """
    Perform MER update (Reptile-style).
    """
    theta_init = copy.deepcopy(model.state_dict())

    for _ in range(s):
        theta_batch_init = copy.deepcopy(model.state_dict())

        for _ in range(k):
            xb, yb = buffer.sample(current_x, current_y, k=k)
            model.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= alpha * param.grad  # inner SGD

        # Inner meta-update
        theta_after = model.state_dict()
        for key in model.state_dict():
            model.state_dict()[key].copy_(
                theta_batch_init[key] + beta * (theta_after[key] - theta_batch_init[key])
            )

    # Outer meta-update
    theta_final = model.state_dict()
    for key in model.state_dict():
        model.state_dict()[key].copy_(
            theta_init[key] + beta * (theta_final[key] - theta_init[key])
        )


def train_task_mer(model, train_loader, criterion, buffer, device):
    model.train()
    for epoch in range(5):  # can increase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            for x, y in zip(data, target):
                # x: [3,32,32], y: scalar
                mer_update(model, criterion, x, y, buffer, device=device)
                buffer.add_samples([(x.cpu(), y.cpu())])  # store sample
    return model


def evaluate(model, test_loaders, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in test_loaders:
            correct, total = 0, 0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            accs.append(100.0 * correct / total if total > 0 else 0)
    return accs


def continual_learning(num_tasks=10, num_classes=10, buffer_size=5000, device="cuda"):
    # Data + model
    data = ContinualCIFAR100(num_tasks=num_tasks)
    train_loaders, test_loaders = data.get_task_loaders()
    model = get_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    # Sequential tasks
    for task_id in range(num_tasks):
        model = train_task_mer(model, train_loaders[task_id], criterion, buffer, device)

        # Evaluate on all seen tasks
        accs = evaluate(model, test_loaders[: task_id + 1], device)
        avg_acc = sum(accs) / len(accs)
        print(f"After learning task {task_id}: {accs} - Avg acc: {avg_acc:.2f}")

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = continual_learning(num_tasks=10, num_classes=10, buffer_size=5000, device=device)


if __name__ == "__main__":
    main()
