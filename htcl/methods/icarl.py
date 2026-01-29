"""
iCaRL (Incremental Classifier and Representation Learning) implementation.

iCaRL combines three key mechanisms:
1. Nearest-mean-of-exemplars classification
2. Exemplar management using herding
3. Knowledge distillation from previous model state

Key equations:
- Classification: argmin_y ||phi(x) - mu_y||_2, where mu_y = mean of exemplar features
- Distillation loss: sum_y^{old} q_y * log(p_y), where q is softmax of old logits

Reference: Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning",
           CVPR 2017, https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import os
import json
from datetime import datetime

from .buffer import ReplayBuffer
from ..utils import evaluate, evaluate_all_tasks
from ..utils.paths import ensure_results_dirs


class ExemplarManager:
    """
    Manages exemplars for iCaRL using herding selection.
    
    Herding selects exemplars that best approximate the class mean in feature space.
    """
    
    def __init__(self, capacity: int = 500, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        # Class-balanced storage: {class_id: [(features, input, label), ...]}
        self.exemplars: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {}
        self.seen_classes: Set[int] = set()
    
    def get_exemplars_per_class(self) -> int:
        """Calculate exemplars per class given current capacity and seen classes."""
        if not self.seen_classes:
            return self.capacity
        return max(1, self.capacity // len(self.seen_classes))
    
    def add_class(self, class_id: int):
        """Register a new class."""
        self.seen_classes.add(class_id)
        if class_id not in self.exemplars:
            self.exemplars[class_id] = []
    
    def select_exemplars(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        class_id: int,
        num_exemplars: int,
    ):
        """
        Select exemplars using herding algorithm.
        
        Herding greedily selects samples that best approximate the class mean.
        
        Args:
            model: Feature extractor (neural network)
            data_loader: DataLoader containing samples from this class
            class_id: Class to select exemplars for
            num_exemplars: Number of exemplars to select
        """
        model.eval()
        features_list = []
        inputs_list = []
        
        # Extract features for all samples of this class
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Filter for this class
                mask = labels == class_id
                if mask.sum() == 0:
                    continue
                
                class_inputs = inputs[mask].to(self.device)
                
                # Get features (use output before final layer if possible)
                features = model(class_inputs)
                
                for i in range(class_inputs.size(0)):
                    features_list.append(features[i].cpu())
                    inputs_list.append(class_inputs[i].cpu())
        
        if not features_list:
            return
        
        features = torch.stack(features_list)
        inputs = torch.stack(inputs_list)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute class mean
        class_mean = features.mean(dim=0)
        class_mean = F.normalize(class_mean, p=2, dim=0)
        
        # Herding: greedily select exemplars
        selected_indices = []
        current_sum = torch.zeros_like(class_mean)
        
        for _ in range(min(num_exemplars, len(features_list))):
            if len(selected_indices) == len(features_list):
                break
            
            # Compute target (what we want the running mean to be)
            if len(selected_indices) == 0:
                target_mean = class_mean
            else:
                # We want (current_sum + new_feature) / (k+1) to be close to class_mean
                k = len(selected_indices)
                target_mean = class_mean * (k + 1) - current_sum
            
            # Find the feature closest to target
            remaining = [i for i in range(len(features_list)) if i not in selected_indices]
            
            best_idx = None
            best_dist = float('inf')
            
            for idx in remaining:
                dist = torch.norm(features[idx] - target_mean).item()
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                current_sum = current_sum + features[best_idx]
        
        # Store selected exemplars
        self.exemplars[class_id] = [
            (features[i], inputs[i], class_id)
            for i in selected_indices
        ]
    
    def reduce_exemplar_sets(self, num_per_class: int):
        """Reduce exemplar sets when new classes are added."""
        for class_id in self.exemplars:
            self.exemplars[class_id] = self.exemplars[class_id][:num_per_class]
    
    def get_all_exemplars(self) -> List[Tuple[torch.Tensor, int]]:
        """Get all exemplars as (input, label) pairs."""
        all_exemplars = []
        for class_id, exemplar_list in self.exemplars.items():
            for _, inp, label in exemplar_list:
                all_exemplars.append((inp, label))
        return all_exemplars
    
    def get_replay_data(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch from exemplars for replay."""
        all_exemplars = self.get_all_exemplars()
        if not all_exemplars:
            return None
        
        import random
        k = min(batch_size, len(all_exemplars))
        samples = random.sample(all_exemplars, k)
        
        inputs = torch.stack([s[0] for s in samples]).to(self.device)
        labels = torch.tensor([s[1] for s in samples], device=self.device)
        
        return inputs, labels
    
    def compute_class_means(
        self,
        model: nn.Module,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute mean feature vectors for each class from exemplars.
        
        Used for nearest-mean-of-exemplars classification.
        """
        model.eval()
        class_means = {}
        
        for class_id, exemplar_list in self.exemplars.items():
            if not exemplar_list:
                continue
            
            features_list = []
            with torch.no_grad():
                for feat, inp, _ in exemplar_list:
                    # Use stored features or recompute
                    inp_tensor = inp.unsqueeze(0).to(self.device)
                    features = model(inp_tensor)
                    features_list.append(features.squeeze(0))
            
            if features_list:
                features = torch.stack(features_list)
                features = F.normalize(features, p=2, dim=1)
                class_means[class_id] = features.mean(dim=0)
        
        return class_means


def distillation_loss(
    outputs: torch.Tensor,
    old_outputs: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss.
    
    L_distill = -sum_y q_y * log(p_y)
    
    where q = softmax(old_outputs / T), p = log_softmax(outputs / T)
    
    Args:
        outputs: Current model outputs
        old_outputs: Stored outputs from previous model
        temperature: Softmax temperature
    
    Returns:
        Distillation loss
    """
    q = F.softmax(old_outputs / temperature, dim=1)
    p = F.log_softmax(outputs / temperature, dim=1)
    
    # Only distill on classes that were in old outputs
    loss = -torch.sum(q * p, dim=1).mean()
    
    return loss * (temperature ** 2)


def train_icarl_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    exemplar_manager: ExemplarManager,
    old_model: Optional[nn.Module],
    device: str,
    distill_weight: float = 1.0,
    temperature: float = 2.0,
) -> float:
    """
    Train for one epoch with iCaRL.
    
    Combines classification loss on current data with:
    - Replay from exemplars
    - Knowledge distillation from old model
    
    Args:
        model: Neural network
        train_loader: DataLoader for current task
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss)
        exemplar_manager: Manages exemplar storage and retrieval
        old_model: Previous model state for distillation (None for first task)
        device: Device to train on
        distill_weight: Weight for distillation loss
        temperature: Temperature for distillation
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    if old_model is not None:
        old_model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Classification loss on current data
        loss = criterion(outputs, labels)
        
        # Add distillation loss if we have an old model
        if old_model is not None:
            with torch.no_grad():
                old_outputs = old_model(inputs)
            
            # Distillation only on old classes (approximate by using old_outputs dimension)
            dist_loss = distillation_loss(outputs, old_outputs, temperature)
            loss = loss + distill_weight * dist_loss
        
        # Replay from exemplars
        replay = exemplar_manager.get_replay_data(batch_size=inputs.size(0))
        if replay is not None:
            x_replay, y_replay = replay
            out_replay = model(x_replay)
            replay_loss = criterion(out_replay, y_replay.long())
            loss = loss + replay_loss
            
            # Distillation on replay samples too
            if old_model is not None:
                with torch.no_grad():
                    old_out_replay = old_model(x_replay)
                replay_dist = distillation_loss(out_replay, old_out_replay, temperature)
                loss = loss + distill_weight * replay_dist
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_icarl_on_task(
    model: nn.Module,
    train_loader: DataLoader,
    exemplar_manager: ExemplarManager,
    num_epochs: int,
    lr: float,
    device: str,
    old_model: Optional[nn.Module] = None,
    distill_weight: float = 1.0,
    temperature: float = 2.0,
    verbose: bool = False,
) -> List[float]:
    """
    Train model on a single task with iCaRL.
    
    Returns list of losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    
    losses = []
    for epoch in range(num_epochs):
        loss = train_icarl_single_epoch(
            model, train_loader, optimizer, criterion,
            exemplar_manager, old_model, device,
            distill_weight, temperature
        )
        losses.append(loss)
        
        if verbose:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f} (iCaRL)")
    
    return losses


def train_icarl_model(
    base_model: nn.Module,
    task_perm: Tuple[int, ...],
    train_loaders: List[DataLoader],
    num_epochs: int,
    lr: float,
    device: str,
    buffer_size: int = 500,
    distill_weight: float = 1.0,
    temperature: float = 2.0,
    verbose: bool = False,
) -> nn.Module:
    """
    Train a fresh copy of the model on tasks in the given order using iCaRL.
    
    Args:
        base_model: Model to clone and train
        task_perm: Order of task indices
        train_loaders: List of DataLoaders for each task
        num_epochs: Epochs per task
        lr: Learning rate
        device: Device to train on
        buffer_size: Exemplar buffer capacity
        distill_weight: Weight for distillation loss
        temperature: Temperature for distillation
        verbose: Print progress
    
    Returns:
        Trained model
    """
    model = copy.deepcopy(base_model).to(device)
    exemplar_manager = ExemplarManager(capacity=buffer_size, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    
    old_model = None
    
    for task_idx, task_id in enumerate(task_perm):
        loader = train_loaders[task_id]
        
        if verbose:
            print(f"  Training on task {task_id} (position {task_idx + 1}/{len(task_perm)})")
        
        # Discover classes in this task
        task_classes = set()
        for _, labels in loader:
            task_classes.update(labels.numpy().tolist())
        
        for class_id in task_classes:
            exemplar_manager.add_class(class_id)
        
        # Reduce existing exemplar sets to make room for new classes
        exemplars_per_class = exemplar_manager.get_exemplars_per_class()
        exemplar_manager.reduce_exemplar_sets(exemplars_per_class)
        
        # Train on this task
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Distillation from old model
                if old_model is not None:
                    with torch.no_grad():
                        old_outputs = old_model(inputs)
                    dist_loss = distillation_loss(outputs, old_outputs, temperature)
                    loss = loss + distill_weight * dist_loss
                
                # Replay from exemplars
                replay = exemplar_manager.get_replay_data(batch_size=inputs.size(0))
                if replay is not None:
                    x_replay, y_replay = replay
                    out_replay = model(x_replay)
                    replay_loss = criterion(out_replay, y_replay.long())
                    loss = loss + replay_loss
                    
                    if old_model is not None:
                        with torch.no_grad():
                            old_out_replay = old_model(x_replay)
                        replay_dist = distillation_loss(out_replay, old_out_replay, temperature)
                        loss = loss + distill_weight * replay_dist
                
                loss.backward()
                optimizer.step()
        
        # Select exemplars for new classes using herding
        for class_id in task_classes:
            exemplar_manager.select_exemplars(
                model, loader, class_id, exemplars_per_class
            )
        
        # Store current model as old model for next task
        old_model = copy.deepcopy(model)
        old_model.eval()
    
    return model


def run_icarl_experiments(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    perms: List[Tuple[int, ...]],
    buffer_size: int = 500,
    num_epochs: int = 30,
    lr: float = 0.01,
    distill_weight: float = 1.0,
    temperature: float = 2.0,
    device: str = "cuda",
    dataset: str = "Unknown",
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run iCaRL experiments across multiple task permutations.
    
    Args:
        model: Base model to clone for each permutation
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        perms: List of task permutations to evaluate
        buffer_size: Exemplar buffer size
        num_epochs: Training epochs
        lr: Learning rate
        distill_weight: Weight for distillation loss
        temperature: Distillation temperature
        device: Device
        dataset: Dataset name for logging
        output_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Dictionary with results and timing information
    """
    import time
    
    total_start_time = time.time()
    num_tasks = len(train_loaders)
    results = []
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running iCaRL experiments on {dataset}")
        print(f"  {len(perms)} permutations, {num_tasks} tasks")
        print(f"  buffer_size={buffer_size}, epochs={num_epochs}, lr={lr}")
        print(f"  distill_weight={distill_weight}, temperature={temperature}")
        print(f"{'=' * 60}\n")
    
    for seq_id, perm in enumerate(perms, start=1):
        perm_start_time = time.time()
        
        if verbose:
            print(f"iCaRL Permutation {seq_id}/{len(perms)}: {perm}")
        
        # Train model on this permutation
        trained_model = train_icarl_model(
            model, perm, train_loaders,
            num_epochs, lr, device, buffer_size,
            distill_weight, temperature
        )
        
        # Evaluate on all tasks (using standard classifier, not NCM)
        accs = evaluate_all_tasks(trained_model, test_loaders, device)
        perm_elapsed_time = time.time() - perm_start_time
        
        results.append({
            "sequence": perm,
            "accuracies": accs,
            "mean_acc": sum(accs) / len(accs),
            "time_seconds": perm_elapsed_time
        })
        
        if verbose:
            print(f"  Accuracies: {[f'{a:.1f}' for a in accs]}")
            print(f"  Mean: {results[-1]['mean_acc']:.2f}%, Time: {perm_elapsed_time:.1f}s\n")
    
    total_elapsed_time = time.time() - total_start_time
    
    # Create summary DataFrame
    df = pd.DataFrame({
        "sequence": [str(r["sequence"]) for r in results],
        **{f"Task{t + 1}": [r["accuracies"][t] for r in results]
           for t in range(num_tasks)},
        "Mean": [r["mean_acc"] for r in results],
        "Time_s": [r.get("time_seconds", 0.0) for r in results]
    })
    
    # Save results to proper subdirectories
    paths = ensure_results_dirs(output_dir)
    
    # Save CSV
    csv_path = os.path.join(paths['csv'], f"icarl_results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON summary
    json_filename = f"icarl_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(paths['json'], json_filename)
    json_summary = {
        "method": "iCaRL",
        "dataset": dataset,
        "total_time_seconds": total_elapsed_time,
        "mean_accuracy": float(df["Mean"].mean()),
        "std_accuracy": float(df["Mean"].std()),
        "per_task_mean": [float(df[f"Task{t + 1}"].mean()) for t in range(num_tasks)],
        "per_task_std": [float(df[f"Task{t + 1}"].std()) for t in range(num_tasks)],
    }
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    if verbose:
        print(f"Saved iCaRL results to {csv_path}")
        print(f"Total iCaRL time: {total_elapsed_time:.1f}s")
    
    return {
        "method": "iCaRL",
        "dataset": dataset,
        "results": results,
        "total_time_seconds": total_elapsed_time,
        "summary": {
            "mean_accuracy": df["Mean"].mean(),
            "std_accuracy": df["Mean"].std(),
            "per_task_mean": [df[f"Task{t + 1}"].mean() for t in range(num_tasks)],
            "per_task_std": [df[f"Task{t + 1}"].std() for t in range(num_tasks)],
            "total_time": total_elapsed_time,
            "avg_time_per_perm": total_elapsed_time / max(len(perms), 1),
        }
    }
