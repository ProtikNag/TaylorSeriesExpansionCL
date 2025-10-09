# er.py (updated)
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate, clone_model
from itertools import permutations
import random
import pandas as pd

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
        self.buffer = []            # stored as tuples of CPU tensors (x,y,z)
        self.seen = 0               # total items ever seen (for reservoir sampling)

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


def train_er_model(
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
    """
    Train model sequentially across tasks using Experience Replay (ER).

    - base_model: model to clone
    - task_perm: sequence of indices into train_loaders (e.g., (0,1) or (1,0))
    - train_loaders: list of DataLoader objects
    - alpha: weight for distillation (MSE on logits) stored in the buffer
    - beta: weight for replay CE loss
    - buffer_size: capacity of replay buffer
    """
    model = clone_model(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    buffer = ReplayBuffer(capacity=buffer_size, device=device)

    model.train()

    for epoch in range(num_epochs):
        # iterate tasks in the provided order
        for task_id in task_perm:
            # ensure task_id indexes into given train_loaders
            loader = train_loaders[task_id]
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward on current batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Replay from buffer: obtain up to len(labels) samples (or fewer)
                replay = buffer.sample(batch_size=len(labels))
                if replay is not None:
                    x_buf, y_buf, z_buf = replay
                    out_buf = model(x_buf)

                    # Cross-entropy on buffer labels (classic ER)
                    ce_loss = criterion(out_buf, y_buf)
                    loss = loss + beta * ce_loss

                    # Optional distillation (logit-matching) using stored logits z_buf
                    # Controlled by alpha. If alpha == 0, this term is disabled.
                    if alpha is not None and alpha > 0.0:
                        # ensure shapes match: z_buf and out_buf
                        try:
                            distill_loss = torch.nn.functional.mse_loss(out_buf, z_buf)
                            loss = loss + alpha * distill_loss
                        except Exception:
                            # If shapes mismatch or other errors, skip distillation
                            pass

                # Backprop and step
                loss.backward()
                optimizer.step()

                # Add current batch examples to buffer (store CPU copies)
                with torch.no_grad():
                    # store logits on CPU as well for distillation
                    logits_cpu = outputs.detach().cpu()
                    for x_item, y_item, z_item in zip(inputs.cpu(), labels.cpu(), logits_cpu):
                        buffer.add_sample(x_item, y_item, z_item)

    return model


def run_er_experiments(model, task_train_loaders, task_test_loaders,
                        buffer_size=500, num_epochs=30, lr=0.01, device="cuda",
                        alpha=0.5, beta=0.5):
    """
    Run ER experiments across all permutations of tasks.
    Returns the trained model (on the last permutation) and the results DataFrame.
    """
    num_tasks = len(task_train_loaders)
    if num_tasks == 0:
        raise ValueError("No task loaders supplied")

    task_indices = list(range(num_tasks))
    results = []

    perms = list(permutations(task_indices))
    print(f"=== Running ER experiment across {len(perms)} permutations ===")

    for seq_id, perm in enumerate(perms, start=1):
        print(f"\n--- ER Sequence {seq_id}: {perm} ---")

        # Train ER model on this permutation (train_er_model clones base model)
        local_model = train_er_model(model, perm, task_train_loaders,
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
    try:
        df.to_csv("./results/er_permutation_results.csv", index=False)
        print("Saved ER results to ./results/er_permutation_results.csv")
    except Exception as e:
        print("Could not save ER results CSV:", e)

    # Return last-trained model and results dataframe
    return local_model, df
