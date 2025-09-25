import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate, clone_model
from itertools import permutations
import random
import pandas as pd
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, capacity=500, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []

    def add_sample(self, x, y, z):
        """Store a sample (x=input, y=label, z=logits)"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((x, y, z))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size or len(self.buffer) == 0:
            return None
        samples = random.sample(self.buffer, batch_size)
        x, y, z = zip(*samples)
        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device)
        z = torch.stack(z).to(self.device)
        return x, y, z


def train_der_model(base_model, task_perm, train_loaders, num_epochs, lr, device,
                    alpha=0.5, beta=0.5, buffer_size=500):
    """Train model sequentially across tasks using DER++."""
    model = clone_model(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    model.train()
    for epoch in range(num_epochs):
        for task_id in task_perm:
            for inputs, labels in train_loaders[task_id]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Replay from buffer (DER++)
                replay = buffer.sample(batch_size=len(labels))
                if replay is not None:
                    x_buf, y_buf, z_buf = replay
                    out_buf = model(x_buf)

                    # Distillation loss (logit matching)
                    distill_loss = torch.nn.functional.mse_loss(out_buf, z_buf)
                    loss += alpha * distill_loss

                    # # Cross-entropy on buffer labels
                    # ce_loss = criterion(out_buf, y_buf)
                    # loss += beta * ce_loss

                loss.backward()
                optimizer.step()

                # Add samples to buffer
                with torch.no_grad():
                    logits = outputs.detach()
                    for x, y, z in zip(inputs, labels, logits):
                        buffer.add_sample(x.cpu(), y.cpu(), z.cpu())

    return model


def run_der_experiments(model, task_train_loaders, task_test_loaders,
                        num_epochs=30, lr=0.01, device="cuda"):
    """Run DER experiments across all permutations of tasks."""
    num_tasks = len(task_train_loaders)
    task_indices = list(range(num_tasks))
    results = []
    buffer_size = 5000

    print(f"=== Running DER experiment across {len(list(permutations(task_indices)))} permutations ===")

    for seq_id, perm in enumerate(permutations(task_indices), start=1):
        print(f"\n--- DER Sequence {seq_id}: {perm} ---")

        # Train DER model on this permutation
        local_model = train_der_model(model, perm, task_train_loaders,
                                      num_epochs, lr, device, buffer_size=buffer_size)

        # Evaluate after all tasks
        accs = [evaluate(local_model, task_test_loaders[tid], device=device)
                for tid in task_indices]

        results.append({"sequence": perm, "accuracies": accs})

    # Save results to CSV
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t+1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })
    df.to_csv("der_permutation_results.csv", index=False)
    print("Saved DER results to der_permutation_results.csv")

    # Boxplot
    plt.figure(figsize=(8, 6))
    df[[f"Task{i+1}" for i in range(num_tasks)]].boxplot()
    plt.title("DER++ Performance Variability Across Task Orders")
    plt.ylabel("Accuracy (%)")
    plt.savefig("der_permutation_boxplot.pdf")
    plt.close()

    print("Saved DER boxplot to der_performance_boxplot.pdf")

    return model, df
