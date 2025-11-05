# er.py (updated)
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate, clone_model
from itertools import permutations
import random
import pandas as pd
from copy import deepcopy
from collections import Counter


class ReplayBuffer:
    """
    Replay buffer that stores (x, y, z) on CPU to conserve GPU memory.
    Uses reservoir sampling behavior to keep the buffer representative over time.
    Sampling returns up to `batch_size` items (so it works even when the buffer
    is smaller than the requested size).
    """

    def __init__(self, capacity=500, device="cuda"):
        self.capacity = int(capacity)
        self.device = device
        self.buffer = []  # stored as tuples of CPU tensors (x,y,z)
        self.seen = 0  # total items ever seen (for reservoir sampling)

    def add_sample(self, x, y, z):
        """
        Add a sample. x,y,z expected to be tensors (possibly on GPU) but we'll store
        CPU copies to avoid holding GPU RAM in the buffer.
        Uses reservoir sampling replacement when capacity is reached.
        """
        # copy to CPU (detached)
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        z_cpu = z.detach().cpu()

        self.seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((x_cpu, y_cpu, z_cpu))
        else:
            # reservoir sampling: replace a random existing element with prob capacity/seen
            # choose an index in [0, seen-1]; if index < capacity, replace slot
            idx = random.randint(0, self.seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (x_cpu, y_cpu, z_cpu)

    def sample(self, batch_size):
        """
        Return up to `batch_size` samples as tensors moved to `self.device`.
        If the buffer is empty, return None.
        """
        if len(self.buffer) == 0:
            return None
        k = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, k)
        x, y, z = zip(*samples)
        # stack and move to device
        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device)
        z = torch.stack(z).to(self.device)
        return x, y, z

    def __len__(self):
        return len(self.buffer)


def train_er_model(base_model, task_perm, train_loaders, num_epochs,
                   lr, device, buffer_size):
    model = clone_model(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    model.train()

    for epoch in range(num_epochs):
        for task_id in task_perm:
            loader = train_loaders[task_id]
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Replay from buffer (basic CE on replay)
                replay = buffer.sample(batch_size=inputs.size(0))
                if replay is not None:
                    # support both (x_buf, y_buf) and (x_buf, y_buf, z_buf) returns
                    if len(replay) == 3:
                        x_buf, y_buf, _ = replay
                    else:
                        x_buf, y_buf = replay

                    # move replay items to device and correct dtype
                    x_buf = x_buf.to(device)
                    y_buf = y_buf.to(device).long()

                    out_buf = model(x_buf)
                    ce_loss = criterion(out_buf, y_buf)
                    beta = 0.5
                    loss = loss + beta * ce_loss

                # Backpropagate
                loss.backward()
                optimizer.step()

                # Add current batch to buffer (store CPU copies; keep logits for compatibility)
                with torch.no_grad():
                    logits_cpu = outputs.detach().cpu()
                    for x_item, y_item, z_item in zip(inputs.cpu(), labels.cpu(), logits_cpu):
                        # keep same add_sample signature so existing ReplayBuffer works
                        buffer.add_sample(x_item, y_item, z_item)

    return model


def run_er_experiments(model, task_train_loaders, task_test_loaders,
                       buffer_size=500, num_epochs=30, lr=0.01, device="cuda"):
    """
    Run ER experiments across all permutations of tasks.
    Returns the trained model (on the last permutation) and the results DataFrame.
    """
    num_tasks = len(task_train_loaders)
    task_indices = list(range(num_tasks))
    results = []

    perms = list(permutations(task_indices))
    print(f"=== Running ER experiment across {len(perms)} permutations ===")

    for seq_id, perm in enumerate(perms, start=1):
        print(f"\n--- ER Sequence {seq_id}: {perm} ---")

        # Train ER model on this permutation (train_er_model clones base model)
        local_model = train_er_model(
            model, perm, task_train_loaders,
            num_epochs, lr, device, buffer_size
        )

        # Evaluate after all tasks
        accs = [evaluate(local_model, task_test_loaders[tid], device=device)
                for tid in task_indices]

        results.append({"sequence": perm, "accuracies": accs})

    # Save results to CSV
    df = pd.DataFrame({
        "sequence": [r["sequence"] for r in results],
        **{f"Task{t + 1}": [r["accuracies"][t] for r in results] for t in range(num_tasks)}
    })
    try:
        df.to_csv("./results/er_permutation_results.csv", index=False)
        print("Saved ER results to ./results/er_permutation_results.csv")
    except Exception as e:
        print("Could not save ER results CSV:", e)
