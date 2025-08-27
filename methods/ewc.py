# ewc.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
from models import get_model
from data import ContinualCIFAR100


class EWC:
    def __init__(self, model, dataloader, device="cuda", lambda_=5000):
        """
        Compute Fisher Information Matrix (diagonal) after training a task.
        """
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.lambda_ = lambda_

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher(dataloader)

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self, dataloader):
        """
        Estimate Fisher Information Matrix diagonally.
        """
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(dataloader)

        return precision_matrices

    def penalty(self, model):
        """
        Compute EWC penalty term.
        """
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return self.lambda_ * loss


class ReplayBuffer:
    def __init__(self, capacity=5000, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []  # list of (data, target)

    def add_samples(self, dataset, max_per_class=50):
        """
        Store samples from dataset into buffer.
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        all_samples = []

        for data, target in loader:
            for x, y in zip(data, target):
                all_samples.append((x.cpu(), y.cpu()))
            if len(all_samples) >= self.capacity:
                break

        # Shuffle before storing
        random.shuffle(all_samples)

        # Merge with existing buffer
        self.buffer.extend(all_samples)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def get_loader(self, batch_size=64):
        if len(self.buffer) == 0:
            return None
        xs, ys = zip(*self.buffer)
        xs = torch.stack(xs)
        ys = torch.tensor(ys)
        dataset = TensorDataset(xs, ys)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_task(model, train_loader, optimizer, criterion, ewc_list, buffer, device):
    model.train()
    epochs = 7
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Mix with replay buffer
            if buffer is not None:
                replay_loader = buffer.get_loader(batch_size=len(target))
                if replay_loader is not None:
                    replay_data, replay_target = next(iter(replay_loader))
                    replay_data, replay_target = replay_data.to(device), replay_target.to(device)
                    data = torch.cat([data, replay_data])
                    target = torch.cat([target, replay_target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Add EWC penalty if past tasks exist
            if len(ewc_list) > 0:
                for ewc in ewc_list:
                    loss += ewc.penalty(model)

            loss.backward()
            optimizer.step()
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


def continual_learning(num_tasks=10, num_classes=10, lambda_=5000, device="cuda"):
    # Prepare data and model
    data = ContinualCIFAR100(num_tasks=num_tasks)
    train_loaders, test_loaders = data.get_task_loaders()
    model = get_model(num_classes=num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ewc_list = []
    buffer = ReplayBuffer(capacity=5000, device=device)

    # Sequentially train on tasks
    for task_id in range(num_tasks):
        model = train_task(model, train_loaders[task_id], optimizer, criterion, ewc_list, buffer, device)

        # After finishing a task, compute Fisher and store
        ewc_list.append(EWC(model, train_loaders[task_id], device=device, lambda_=lambda_))

        # Store samples in buffer for replay
        buffer.add_samples(train_loaders[task_id].dataset)

        # Evaluate on all tasks seen so far
        accs = evaluate(model, test_loaders[: task_id + 1], device)
        avg_acc = sum(accs) / len(accs)
        print(f"After learning task {task_id}: {accs} - Avg acc: {avg_acc:.2f}")

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = continual_learning(num_tasks=10, num_classes=10, lambda_=5000, device=device)


if __name__ == "__main__":
    main()
