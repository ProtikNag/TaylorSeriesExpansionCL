# methods/naive.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import evaluate


def train_naive(model, task_train_loaders, task_test_loaders, num_epochs=5, lr=0.001, device='cuda'):
    """
    Sequentially trains the model on each task without any continual learning mechanism.
    Assumes labels are already remapped to [0–9] for each task.
    """
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    acc_per_task = []
    replay_buffer = []
    replay_size = 5000

    for task_id, train_loader in enumerate(task_train_loaders):
        print(f"\n--- Training on Task {task_id} (Naive) ---")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        combined_datasets = [train_loader.dataset] + replay_buffer
        combined_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(combined_datasets), batch_size=64, shuffle=True
        )

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(combined_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Labels already in range [0–9]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Task {task_id} - Epoch {epoch+1}, Loss: {running_loss:.4f}")

        # Update replay buffer
        replay_buffer.append(train_loader.dataset)
        replay_buffer = replay_buffer[-replay_size:]  # Keep a fixed buffer size

        # Evaluate on all tasks seen so far
        accs = []
        for test_task_id, test_loader in enumerate(task_test_loaders[:task_id+1]):
            acc = evaluate(model, test_loader, device=device)  # No offset needed
            accs.append(acc)
        acc_per_task.append(accs)

    return model, acc_per_task
