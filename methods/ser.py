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

    def __len__(self):
        return len(self.buffer)


def train_ser_model(
        base_model,
        task_perm,
        train_loaders,
        num_epochs,
        lr,
        device,
        alpha=0.5,
        beta=0.5,
        buffer_size=500
):
    model = clone_model(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    model.train()

    for epoch in range(num_epochs):
        for task_id in task_perm:
            old_model = clone_model(model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

            for inputs, labels in train_loaders[task_id]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass through current (trainable) model
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # ---------------------------
                # Backward consistency (DER-style): use buffer stored logits
                # ---------------------------
                replay = buffer.sample(batch_size=len(labels))
                if replay is not None:
                    x_buf, y_buf, z_buf = replay
                    out_buf = model(x_buf)

                    # Distillation loss (logit matching) on buffer
                    distill_loss = torch.nn.functional.mse_loss(out_buf, z_buf)
                    loss = loss + alpha * distill_loss

                    # If you want classification loss on buffer as well (DER++ had it optionally),
                    # uncomment the following lines (kept commented to mirror original DER minimal changes)
                    # ce_loss = criterion(out_buf, y_buf)
                    # loss = loss + beta * ce_loss

                # ---------------------------
                # Forward consistency (SER): compare current model outputs to frozen old_model outputs on current inputs
                # ---------------------------
                # As per the SER paper, skip consistency for the first task (when buffer empty / no prior knowledge).
                if len(buffer) > 0:
                    with torch.no_grad():
                        out_old = old_model(inputs)
                    # MSE between logits of current model and old model on current training data
                    fwd_consistency = torch.nn.functional.mse_loss(outputs, out_old)
                    loss = loss + beta * fwd_consistency

                # Backprop and step
                loss.backward()
                optimizer.step()

                # Add current batch samples to buffer (store inputs, labels, and logits)
                # store on CPU to conserve GPU memory
                with torch.no_grad():
                    logits = outputs.detach().cpu()
                    for x_item, y_item, z_item in zip(inputs.cpu(), labels.cpu(), logits):
                        buffer.add_sample(x_item, y_item, z_item)

    return model


def run_ser_experiments(model, task_train_loaders, task_test_loaders,
                        buffer_size=500, num_epochs=30, lr=0.01, device="cuda",
                        alpha=0.5, beta=0.5):
    """
    Run SER experiments across all permutations of tasks.
    Returns the trained model (on the last permutation) and the results DataFrame.
    """
    num_tasks = len(task_train_loaders)
    task_indices = list(range(num_tasks))
    results = []

    perms = list(permutations(task_indices))
    print(f"=== Running SER experiment across {len(perms)} permutations ===")

    for seq_id, perm in enumerate(perms, start=1):
        print(f"\n--- SER Sequence {seq_id}: {perm} ---")

        # Train SER model on this permutation
        local_model = train_ser_model(model, perm, task_train_loaders,
                                      num_epochs, lr, device,
                                      alpha=alpha, beta=beta, buffer_size=buffer_size)

        # Evaluate after all tasks
        accs = [evaluate(local_model, task_test_loaders[tid], device=device)
                for tid in task_indices]

        results.append({"sequence": perm, "accuracies": accs})

    # Save results to CSV
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t + 1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })
    df.to_csv("./results/ser_permutation_results.csv", index=False)
    print("Saved SER results to ser_permutation_results.csv")

    # Return last-trained model and results dataframe
    return local_model, df

