# methods/ewc.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import evaluate
from copy import deepcopy


class EWC:
    def __init__(self, model, ewc_lambda=1000.0, device='cuda'):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher(self, data_loader, criterion):
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param).to(self.device)

        self.model.eval()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        for name in fisher:
            fisher[name] /= len(data_loader)

        return fisher

    def consolidate(self, data_loader, criterion):
        self.fisher = self.compute_fisher(data_loader, criterion)
        self.optimal_params = {
            name: param.clone().detach() for name, param in self.model.named_parameters()
        }

    def penalty(self):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss


def train_ewc(model, task_train_loaders, task_test_loaders, num_epochs=5, lr=0.001, ewc_lambda=1000.0, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    ewc_obj = EWC(model, ewc_lambda=ewc_lambda, device=device)

    acc_per_task = []

    for task_id, train_loader in enumerate(task_train_loaders):
        print(f"\n--- Training on Task {task_id} (EWC) ---")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # labels already in range [0–9]

                if task_id > 0:
                    loss += ewc_obj.penalty()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Task {task_id} - Epoch {epoch+1}, Loss: {running_loss:.4f}")

        # Store Fisher and optimal params
        ewc_obj.consolidate(train_loader, criterion)

        # Evaluate on all seen tasks so far
        accs = []
        for test_task_id, test_loader in enumerate(task_test_loaders[:task_id + 1]):
            acc = evaluate(model, test_loader, device=device)
            accs.append(acc)
        acc_per_task.append(accs)

    return model, acc_per_task
