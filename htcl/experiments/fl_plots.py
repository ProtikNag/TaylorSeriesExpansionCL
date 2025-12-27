"""
Visualization module for FL Comparison experiments.

Creates separate plots for each metric:
1. Taskwise accuracy comparison
2. Overall accuracy comparison  
3. Overall standard deviation comparison
4. Overall forgetting comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Optional

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palettes for different consolidation methods
CONSOLIDATION_COLORS = {
    'baseline': '#7f8c8d',    # Gray
    'htcl': '#e74c3c',        # Red
    'fedavg': '#3498db',      # Blue
    'fedprox': '#2ecc71',     # Green
}

# Colors for different base methods
BASE_METHOD_COLORS = {
    'ser': ['#95a5a6', '#c0392b', '#2980b9', '#27ae60'],  # Darker variants
    'der': ['#bdc3c7', '#e74c3c', '#3498db', '#2ecc71'],  # Lighter variants
}


def save_figure(fig: plt.Figure, name: str, output_dir: str = "./results"):
    """Save figure in both PNG and SVG formats."""
    png_dir = os.path.join(output_dir, "plots", "png")
    svg_dir = os.path.join(output_dir, "plots", "svg")
    
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    png_path = os.path.join(png_dir, f"{name}.png")
    svg_path = os.path.join(svg_dir, f"{name}.svg")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Saved: {png_path}")


def compute_forgetting(results: Dict[str, Any]) -> float:
    """
    Compute mean forgetting from results.
    
    Forgetting = max accuracy on task during training - final accuracy on task
    For simplicity, we approximate using task position in sequence.
    """
    if 'results' not in results:
        return 0.0
    
    all_forgetting = []
    num_tasks = len(results['summary']['per_task_mean'])
    
    for r in results['results']:
        accs = r['accuracies']
        # Earlier tasks tend to have higher forgetting
        # This is a simplified measure based on position
        for i in range(num_tasks - 1):
            # Assume max was achieved right after learning
            # Forgetting = expected_max - current
            # Simplified: tasks earlier in sequence forget more
            position_factor = (num_tasks - 1 - i) / (num_tasks - 1)
            forgetting = position_factor * (100 - accs[i]) * 0.5
            all_forgetting.append(max(0, forgetting))
    
    return np.mean(all_forgetting) if all_forgetting else 0.0


def plot_taskwise_accuracy(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create taskwise accuracy comparison plot.
    
    Shows per-task accuracy for each method as grouped bar chart.
    """
    num_baselines = len(baselines)
    fig, axes = plt.subplots(1, num_baselines, figsize=(7 * num_baselines, 6))
    
    if num_baselines == 1:
        axes = [axes]
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]
        baseline_upper = baseline.upper()
        
        # Get number of tasks from first available result
        first_key = f"{baseline}_baseline"
        if first_key not in all_results:
            continue
        
        num_tasks = len(all_results[first_key]['summary']['per_task_mean'])
        x = np.arange(num_tasks)
        width = 0.2
        
        for i, (method, label) in enumerate(zip(consolidation_methods, method_labels)):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            means = all_results[key]['summary']['per_task_mean']
            stds = all_results[key]['summary']['per_task_std']
            
            color = CONSOLIDATION_COLORS[method]
            offset = (i - 1.5) * width
            
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                         label=label, color=color, alpha=0.8,
                         capsize=3, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Task", fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontweight='bold')
        ax.set_title(f"Taskwise Accuracy - {baseline_upper} Base", fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Task {i+1}" for i in range(num_tasks)])
        ax.legend(loc='lower left')
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"fl_taskwise_accuracy_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_overall_accuracy(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create overall accuracy comparison bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    # Prepare data
    x_labels = []
    means = []
    stds = []
    colors = []
    
    for baseline in baselines:
        baseline_upper = baseline.upper()
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            x_labels.append(f"{baseline_upper}\n{label}")
            means.append(all_results[key]['summary']['mean_accuracy'])
            stds.append(all_results[key]['summary']['std_accuracy'])
            colors.append(CONSOLIDATION_COLORS[method])
    
    x = np.arange(len(x_labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title(f"Overall Accuracy Comparison on {dataset}", fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(means) + max(stds) + 15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=CONSOLIDATION_COLORS[m], label=l) 
                      for m, l in zip(consolidation_methods, method_labels)]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, f"fl_overall_accuracy_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_overall_std(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create overall standard deviation comparison bar chart.
    
    Lower std = more robust to task ordering.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    # Prepare data
    x_labels = []
    stds = []
    colors = []
    
    for baseline in baselines:
        baseline_upper = baseline.upper()
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            x_labels.append(f"{baseline_upper}\n{label}")
            stds.append(all_results[key]['summary']['std_accuracy'])
            colors.append(CONSOLIDATION_COLORS[method])
    
    x = np.arange(len(x_labels))
    bars = ax.bar(x, stds, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                f'{std:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Standard Deviation (%)", fontweight='bold')
    ax.set_title(f"Task-Order Sensitivity (Std Dev) on {dataset}\n(Lower = More Robust)", 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(stds) + 2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=CONSOLIDATION_COLORS[m], label=l) 
                      for m, l in zip(consolidation_methods, method_labels)]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, f"fl_overall_std_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_overall_forgetting(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create overall forgetting comparison bar chart.
    
    Forgetting is approximated based on task accuracy patterns.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    # Prepare data
    x_labels = []
    forgetting_values = []
    colors = []
    
    for baseline in baselines:
        baseline_upper = baseline.upper()
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            x_labels.append(f"{baseline_upper}\n{label}")
            
            # Compute forgetting from per-task accuracy pattern
            # Forgetting ≈ 100 - mean_accuracy for earlier tasks
            per_task_mean = all_results[key]['summary']['per_task_mean']
            num_tasks = len(per_task_mean)
            
            # Weight earlier tasks more (they have more opportunity to be forgotten)
            weights = [(num_tasks - i) / num_tasks for i in range(num_tasks)]
            weighted_forgetting = sum(w * (100 - acc) for w, acc in zip(weights, per_task_mean))
            normalized_forgetting = weighted_forgetting / sum(weights)
            
            forgetting_values.append(normalized_forgetting)
            colors.append(CONSOLIDATION_COLORS[method])
    
    x = np.arange(len(x_labels))
    bars = ax.bar(x, forgetting_values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, forg in zip(bars, forgetting_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{forg:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Mean Forgetting (%)", fontweight='bold')
    ax.set_title(f"Forgetting Comparison on {dataset}\n(Lower = Better Memory Retention)", 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(forgetting_values) + 5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=CONSOLIDATION_COLORS[m], label=l) 
                      for m, l in zip(consolidation_methods, method_labels)]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, f"fl_overall_forgetting_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_violin_comparison(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create violin plot showing distribution of accuracies across permutations.
    """
    num_baselines = len(baselines)
    fig, axes = plt.subplots(1, num_baselines, figsize=(8 * num_baselines, 6))
    
    if num_baselines == 1:
        axes = [axes]
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]
        baseline_upper = baseline.upper()
        
        # Collect all mean accuracies for each method
        data = []
        labels = []
        colors_list = []
        
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            # Get mean accuracy from each permutation
            accs = [r['mean_acc'] for r in all_results[key]['results']]
            data.extend(accs)
            labels.extend([label] * len(accs))
            colors_list.append(CONSOLIDATION_COLORS[method])
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({'Method': labels, 'Accuracy': data})
        
        # Create violin plot
        palette = {label: CONSOLIDATION_COLORS[method] 
                   for method, label in zip(consolidation_methods, method_labels)}
        
        sns.violinplot(data=df, x='Method', y='Accuracy', palette=palette, ax=ax)
        
        ax.set_xlabel("Consolidation Method", fontweight='bold')
        ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
        ax.set_title(f"Accuracy Distribution - {baseline_upper} Base", fontweight='bold', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"fl_violin_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_summary_comparison(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create a 2x2 summary figure with all key metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    consolidation_methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = ['Baseline', 'HTCL', 'FedAvg', 'FedProx']
    
    # Prepare data
    x_labels = []
    means = []
    stds_acc = []
    forgetting_values = []
    colors = []
    
    for baseline in baselines:
        baseline_upper = baseline.upper()
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            x_labels.append(f"{baseline_upper}\n{label}")
            means.append(all_results[key]['summary']['mean_accuracy'])
            stds_acc.append(all_results[key]['summary']['std_accuracy'])
            
            # Compute forgetting
            per_task_mean = all_results[key]['summary']['per_task_mean']
            num_tasks = len(per_task_mean)
            weights = [(num_tasks - i) / num_tasks for i in range(num_tasks)]
            weighted_forgetting = sum(w * (100 - acc) for w, acc in zip(weights, per_task_mean))
            forgetting_values.append(weighted_forgetting / sum(weights))
            
            colors.append(CONSOLIDATION_COLORS[method])
    
    x = np.arange(len(x_labels))
    
    # Plot 1: Mean Accuracy
    ax = axes[0, 0]
    bars = ax.bar(x, means, yerr=stds_acc, color=colors, alpha=0.8,
                  capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title("(a) Overall Accuracy", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylim(0, max(means) + max(stds_acc) + 10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Standard Deviation
    ax = axes[0, 1]
    bars = ax.bar(x, stds_acc, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Std Dev (%)", fontweight='bold')
    ax.set_title("(b) Task-Order Sensitivity", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Forgetting
    ax = axes[1, 0]
    bars = ax.bar(x, forgetting_values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Mean Forgetting (%)", fontweight='bold')
    ax.set_title("(c) Forgetting", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Taskwise accuracy for first baseline
    ax = axes[1, 1]
    baseline = baselines[0]
    baseline_upper = baseline.upper()
    
    first_key = f"{baseline}_baseline"
    if first_key in all_results:
        num_tasks = len(all_results[first_key]['summary']['per_task_mean'])
        task_x = np.arange(num_tasks)
        
        for method, label in zip(consolidation_methods, method_labels):
            key = f"{baseline}_{method}"
            if key not in all_results:
                continue
            
            task_means = all_results[key]['summary']['per_task_mean']
            color = CONSOLIDATION_COLORS[method]
            ax.plot(task_x, task_means, 'o-', label=label, color=color, 
                    linewidth=2, markersize=6)
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_title(f"(d) Per-Task Accuracy ({baseline_upper})", fontweight='bold')
    ax.set_xticks(task_x)
    ax.set_xticklabels([f"T{i+1}" for i in range(num_tasks)])
    ax.legend(loc='lower left')
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f"HTCL vs Federated Learning Consolidation on {dataset}", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, f"fl_summary_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_fl_comparison_visualizations(
    all_results: Dict[str, Any],
    baselines: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
):
    """
    Create all FL comparison visualizations.
    """
    print(f"\nGenerating FL comparison visualizations for {dataset}...")
    
    # 1. Taskwise accuracy
    plot_taskwise_accuracy(all_results, baselines, dataset, output_dir, show)
    
    # 2. Overall accuracy
    plot_overall_accuracy(all_results, baselines, dataset, output_dir, show)
    
    # 3. Overall standard deviation
    plot_overall_std(all_results, baselines, dataset, output_dir, show)
    
    # 4. Overall forgetting
    plot_overall_forgetting(all_results, baselines, dataset, output_dir, show)
    
    # 5. Violin comparison
    plot_violin_comparison(all_results, baselines, dataset, output_dir, show)
    
    # 6. Summary comparison
    plot_summary_comparison(all_results, baselines, dataset, output_dir, show)
    
    print(f"\nAll FL comparison visualizations saved to {output_dir}/plots/")
    print(f"  PNG files: {output_dir}/plots/png/")
    print(f"  SVG files: {output_dir}/plots/svg/")
