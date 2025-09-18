# der_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import get_model
from data import ContinualSplitMNIST


# ------------------------
# Replay Buffer
# ------------------------
class ReplayBuffer:
    def __init__(self, capacity=5000, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []  # store tuples (x, y, logits)
        self.n_seen = 0  # for proper reservoir sampling

    def add_sample(self, x, y, logits):
        """Add a sample with true reservoir sampling."""
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone().cpu(), y.clone().cpu(), logits.clone().cpu()))
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
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


# ------------------------
# Training & Evaluation
# ------------------------
def train_task_der(model, train_loader, optimizer, criterion, buffer,
                   alpha=0.5, beta=0.5, der_plus=True, device="cuda",
                   task_id=None, epochs=5):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"[Task {task_id}] Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in loop:
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

            loop.set_postfix(loss=loss.item())

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


# ------------------------
# Continual Learning
# ------------------------
def continual_learning(train_loaders, test_loaders,
                       num_classes=2, buffer_size=500,
                       alpha=0.5, beta=0.5, der_plus=True,
                       device="cuda", epochs=5):
    model = get_model(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    for task_id, train_loader in enumerate(train_loaders):
        model = train_task_der(model, train_loader, optimizer, criterion,
                               buffer, alpha=alpha, beta=beta, der_plus=der_plus,
                               device=device, task_id=task_id, epochs=epochs)

    # Final evaluation on all tasks
    accs = evaluate(model, test_loaders, device)
    return accs


# ------------------------
# Experiment Runner
# ------------------------
def run_permutation_experiment(num_tasks=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ContinualSplitMNIST(num_tasks=num_tasks, batch_size=64, debug=True, samples_per_class=500)
    all_train, all_test = data.get_task_loaders()

    task_indices = list(range(num_tasks))
    results = []

    for seq_id, perm in enumerate(itertools.permutations(task_indices), start=1):
        print(f"=== Running sequence {seq_id}/{len(list(itertools.permutations(task_indices)))}: {perm}")

        # reorder loaders
        train_loaders = [all_train[i] for i in perm]
        test_loaders = [all_test[i] for i in perm]

        accs = continual_learning(train_loaders, test_loaders,
                                  num_classes=2, buffer_size=200,
                                  alpha=0.5, beta=0.5, der_plus=True,
                                  device=device, epochs=5)

        results.append({"sequence": perm, "accuracies": accs})

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t+1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })

    # Save results
    df.to_csv("der_permutation_results.csv", index=False)
    print("Saved results to der_permutation_results.csv")

    # Boxplot
    plt.figure(figsize=(8, 6))
    df[[f"Task{i+1}" for i in range(num_tasks)]].boxplot()
    plt.title("DER Performance Variability Across Task Orders")
    plt.ylabel("Accuracy (%)")
    plt.savefig("der_permutation_boxplot.pdf")
    plt.show()

    return df


if __name__ == "__main__":
    run_permutation_experiment(num_tasks=5)
