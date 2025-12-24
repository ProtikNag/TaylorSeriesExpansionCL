"""
Utility functions for HTCL experiments.

IMPORTANT CHANGES:
- ResultsManager now properly organizes files:
  - CSV files go in 'csv/' folder
  - JSON files go in 'json/' folder  
  - Plots go in 'plots/png/' and 'plots/svg/' folders
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda"
) -> float:
    """
    Evaluate model accuracy on a test loader.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    correct, total = 0, 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_all_tasks(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: str = "cuda"
) -> List[float]:
    """Evaluate model on all task test loaders."""
    return [evaluate(model, loader, device) for loader in test_loaders]


def compute_avg_accuracy(acc_matrix: List[List[float]]) -> float:
    """Compute average accuracy from accuracy matrix."""
    if not acc_matrix:
        return 0.0
    return np.mean([np.mean(row) for row in acc_matrix])


def compute_avg_forgetting(acc_matrix: List[List[float]]) -> float:
    """
    Compute average forgetting.
    
    Forgetting for task k is: max(acc[t][k] for t < T) - acc[T][k]
    where T is the final timestep.
    """
    if not acc_matrix:
        return 0.0
    
    num_tasks = len(acc_matrix)
    forgetting = []
    
    for task_id in range(num_tasks):
        accs_on_task = [
            acc_matrix[t][task_id]
            for t in range(task_id, len(acc_matrix))
            if task_id < len(acc_matrix[t])
        ]
        if len(accs_on_task) >= 2:
            max_prev = max(accs_on_task[:-1])
            last = accs_on_task[-1]
            forgetting.append(max_prev - last)
    
    return np.mean(forgetting) if forgetting else 0.0


def compute_backward_transfer(acc_matrix: List[List[float]]) -> float:
    """
    Compute backward transfer (BWT).
    
    BWT measures how much learning new tasks affects old tasks.
    Positive = beneficial, Negative = forgetting.
    """
    if len(acc_matrix) < 2:
        return 0.0
    
    bwt = []
    T = len(acc_matrix) - 1
    
    for i in range(T):
        if i < len(acc_matrix[T]) and i < len(acc_matrix[i]):
            bwt.append(acc_matrix[T][i] - acc_matrix[i][i])
    
    return np.mean(bwt) if bwt else 0.0


def compute_forward_transfer(acc_matrix: List[List[float]], random_baseline: float = 50.0) -> float:
    """
    Compute forward transfer (FWT).
    
    FWT measures how much learning old tasks helps new tasks.
    """
    if len(acc_matrix) < 2:
        return 0.0
    
    fwt = []
    for i in range(1, len(acc_matrix)):
        if i < len(acc_matrix[i-1]):
            # Performance on task i before training on it
            fwt.append(acc_matrix[i-1][i] - random_baseline)
    
    return np.mean(fwt) if fwt else 0.0


def estimate_diag_hessian(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Hessian using squared gradients (Fisher approximation).
    
    This is computationally cheaper than exact Hessian.
    """
    model.eval()
    hessian_diag = {
        name: torch.zeros_like(param.data, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    n_batches = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                hessian_diag[name] += param.grad.pow(2)
        
        n_batches += 1
    
    # Average over batches
    for name in hessian_diag:
        hessian_diag[name] /= max(n_batches, 1)
    
    return hessian_diag


def estimate_diag_hessian_exact(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Hessian using second-order derivatives.
    
    More accurate but more computationally expensive.
    """
    model.eval()
    hessian_diag = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    param_list = [p for p in model.parameters() if p.requires_grad]
    name_list = [n for n, p in model.named_parameters() if p.requires_grad]
    
    n_batches = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # First-order gradients
        grads = torch.autograd.grad(
            loss, param_list, create_graph=True, allow_unused=True
        )
        
        # Second-order gradients (diagonal only)
        for i, (name, grad) in enumerate(zip(name_list, grads)):
            if grad is not None:
                grad2 = torch.autograd.grad(
                    grad, param_list[i],
                    grad_outputs=torch.ones_like(grad),
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if grad2 is not None:
                    hessian_diag[name] += grad2.detach()
        
        n_batches += 1
    
    for name in hessian_diag:
        hessian_diag[name] /= max(n_batches, 1)
    
    return hessian_diag


class ResultsManager:
    """
    Manages saving and loading experiment results.
    
    Directory structure:
        output_dir/
        ├── csv/           # CSV result files
        ├── json/          # JSON metadata/summary files
        ├── plots/
        │   ├── png/       # PNG plots (300 DPI)
        │   └── svg/       # SVG plots (vector)
        └── checkpoints/   # Model checkpoints
    """
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        
        # Properly organized subdirectories
        self.csv_dir = os.path.join(output_dir, "csv")
        self.json_dir = os.path.join(output_dir, "json")
        self.plots_png_dir = os.path.join(output_dir, "plots", "png")
        self.plots_svg_dir = os.path.join(output_dir, "plots", "svg")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        
        # Create directories
        for d in [self.csv_dir, self.json_dir, self.plots_png_dir, 
                  self.plots_svg_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)
    
    def get_csv_path(self, filename: str) -> str:
        """Get full path for a CSV file."""
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        return os.path.join(self.csv_dir, filename)
    
    def get_json_path(self, filename: str) -> str:
        """Get full path for a JSON file."""
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        return os.path.join(self.json_dir, filename)
    
    def save_results(
        self,
        results: Dict[str, Any],
        experiment_name: str,
        timestamp: bool = True
    ) -> str:
        """Save results to JSON file in the json/ directory."""
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_{ts}.json"
        else:
            filename = f"{experiment_name}.json"
        
        path = os.path.join(self.json_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return path
    
    def save_csv(
        self,
        df,  # pandas DataFrame
        filename: str
    ) -> str:
        """Save DataFrame to CSV file in the csv/ directory."""
        path = self.get_csv_path(filename)
        df.to_csv(path, index=False)
        return path
    
    def save_model(
        self,
        model: nn.Module,
        name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": model.state_dict(),
            "metadata": metadata or {}
        }
        path = os.path.join(self.checkpoints_dir, f"{name}.pt")
        torch.save(checkpoint, path)
        return path
    
    def load_model(
        self,
        model: nn.Module,
        name: str
    ) -> Dict:
        """Load model checkpoint."""
        path = os.path.join(self.checkpoints_dir, f"{name}.pt")
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        return checkpoint.get("metadata", {})
    
    def get_plot_paths(self, name: str) -> Tuple[str, str]:
        """Get paths for PNG and SVG versions of a plot."""
        return (
            os.path.join(self.plots_png_dir, f"{name}.png"),
            os.path.join(self.plots_svg_dir, f"{name}.svg")
        )
    
    @staticmethod
    def reorganize_existing_results(base_dir: str) -> None:
        """
        Utility to reorganize existing messy results into proper structure.
        
        Call this once to fix the current file organization:
            ResultsManager.reorganize_existing_results("./results/splitmnist/er")
        """
        import shutil
        
        # Create proper directories
        csv_dir = os.path.join(base_dir, "csv")
        json_dir = os.path.join(base_dir, "json")
        plots_dir = os.path.join(base_dir, "plots")
        plots_png = os.path.join(plots_dir, "png")
        plots_svg = os.path.join(plots_dir, "svg")
        
        for d in [csv_dir, json_dir, plots_png, plots_svg]:
            os.makedirs(d, exist_ok=True)
        
        # Move files based on extension
        for root, dirs, files in os.walk(base_dir):
            # Skip if already in proper location
            if 'csv' in root or 'json' in root or 'plots' in root:
                continue
                
            for file in files:
                src = os.path.join(root, file)
                
                if file.endswith('.json'):
                    dst = os.path.join(json_dir, file)
                elif file.endswith('.csv'):
                    dst = os.path.join(csv_dir, file)
                elif file.endswith('.png'):
                    dst = os.path.join(plots_png, file)
                elif file.endswith('.svg'):
                    dst = os.path.join(plots_svg, file)
                else:
                    continue
                
                if src != dst and os.path.exists(src):
                    print(f"Moving: {src} -> {dst}")
                    shutil.move(src, dst)
        
        print(f"Reorganization complete for: {base_dir}")


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print model summary."""
    n_params = count_parameters(model)
    print(f"\n{name} Summary:")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Architecture: {model.__class__.__name__}")
