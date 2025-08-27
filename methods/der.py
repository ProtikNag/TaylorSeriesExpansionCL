# der.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from tqdm import tqdm  # NEW
from models import get_model
from data import ContinualCIFAR100


class ReplayBuffer:
    def __init__(self, capacity=5000, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []  # store tuples (x, y, logits)

    def add_sample(self, x, y, logits):
        """
        Add a sample with reservoir sampling.
        x: [3,32,32] tensor
        y: scalar tensor
        logits: [num_classes] tensor
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone().cpu(), y.clone().cpu(), logits.clone().cpu()))
        else:
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = (x.clone().cpu(), y.clone().cpu(), logits.clone().cpu())

    def sample(self, batch_size=64):
        if len(self.buffer) == 0:
            return None
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        xs, ys, zs = zip(*samples)
        xs = torch.stack(xs).to(self.device)
        ys = torch.tensor(ys).to(self.device)
        zs = torch.stack(zs).to(self.device)
        return xs, ys, zs


def train_task_der(model, train_loader, optimizer, criterion, buffer,
                   alpha=0.5, beta=0.5, der_plus=True, device="cuda",
                   task_id=None):
    """
    Train a model on one task using DER/DER++.
    """
    model.train()
    for epoch in range(5):  # can increase if needed
        loop = tqdm(train_loader, desc=f"[Task {task_id}] Epoch {epoch+1}/5", leave=False)
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # DER/DER++ loss from buffer
            replay = buffer.sample(batch_size=len(target))
            if replay is not None:
                x_buf, y_buf, z_buf = replay
                out_buf = model(x_buf)

                # distillation loss on logits
                distill_loss = torch.nn.functional.mse_loss(out_buf, z_buf)
                loss += alpha * distill_loss

                # DER++: also add CE loss on buffer labels
                if der_plus:
                    ce_loss = criterion(out_buf, y_buf)
                    loss += beta * ce_loss

            loss.backward()
            optimizer.step()

            # update buffer with current samples (store logits)
            with torch.no_grad():
                logits = output.detach()
                for x, y, z in zip(data, target, logits):
                    buffer.add_sample(x.cpu(), y.cpu(), z.cpu())

            # Update tqdm bar
            loop.set_postfix(loss=loss.item())

    print(f"[Task {task_id}] Finished training.\n")
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


def continual_learning(num_tasks=10, num_classes=10, buffer_size=5000,
                       alpha=0.5, beta=0.5, der_plus=True, device="cuda"):
    # Data + model
    data = ContinualCIFAR100(num_tasks=num_tasks)
    train_loaders, test_loaders = data.get_task_loaders()
    model = get_model(num_classes=num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    # Sequentially train tasks
    for task_id in range(num_tasks):
        model = train_task_der(model, train_loaders[task_id], optimizer, criterion,
                               buffer, alpha=alpha, beta=beta, der_plus=der_plus,
                               device=device, task_id=task_id)

        # Evaluate after each task
        accs = evaluate(model, test_loaders[: task_id + 1], device)
        avg_acc = sum(accs) / len(accs)
        print(f"After learning task {task_id}: {accs} - Avg acc: {avg_acc:.2f}")

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = continual_learning(num_tasks=10, num_classes=10,
                               buffer_size=5000, alpha=0.5, beta=0.5,
                               der_plus=True, device=device)


if __name__ == "__main__":
    main()
