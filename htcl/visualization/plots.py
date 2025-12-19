"""
Visualization module for HTCL experiments.

Creates publication-quality plots and saves them in both PNG and SVG formats.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Optional, Tuple

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

# Color palettes
COLORS = {
    'er': '#3498db',           # Blue
    'htcl_2': '#e74c3c',       # Red
    'htcl_3': '#2ecc71',       # Green
    'htcl_4': '#9b59b6',       # Purple
    'htcl_5': '#f39c12',       # Orange
    'primary': '#2c3e50',      # Dark blue-gray
    'secondary': '#7f8c8d',    # Gray
    'accent': '#1abc9c',       # Teal
}

HIERARCHY_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']


def save_figure(fig: plt.Figure, name: str, output_dir: str = "./results"):
    """Save figure in both PNG and SVG formats."""
    png_dir = os.path.join(output_dir, "plots", "png")
    svg_dir = os.path.join(output_dir, "plots", "svg")
    
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    fig.savefig(os.path.join(png_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(svg_dir, f"{name}.svg"), format='svg', bbox_inches='tight')
    
    print(f"Saved: {name}.png and {name}.svg")


def plot_task_accuracy_comparison(
    er_results: Dict[str, Any],
    htcl_results: Dict[str, Any],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create a comparison plot of task-wise accuracy between ER and HTCL.
    
    Shows box plots for each task comparing the two methods.
    """
    num_tasks = len(er_results["summary"]["per_task_mean"])
    
    # Prepare data
    er_data = []
    htcl_data = []
    
    for r in er_results["results"]:
        er_data.append(r["accuracies"])
    for r in htcl_results["results"]:
        htcl_data.append(r["accuracies"])
    
    er_df = pd.DataFrame(er_data, columns=[f"Task {i+1}" for i in range(num_tasks)])
    htcl_df = pd.DataFrame(htcl_data, columns=[f"Task {i+1}" for i in range(num_tasks)])
    
    er_df["Method"] = "ER"
    htcl_df["Method"] = "HTCL"
    
    combined = pd.concat([er_df, htcl_df])
    melted = combined.melt(id_vars=["Method"], var_name="Task", value_name="Accuracy")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped box plot
    sns.boxplot(
        data=melted, x="Task", y="Accuracy", hue="Method",
        palette={"ER": COLORS["er"], "HTCL": COLORS["htcl_2"]},
        ax=ax, width=0.6
    )
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_title(f"Task-wise Accuracy Comparison on {dataset}", fontweight='bold', fontsize=14)
    ax.legend(title="Method", loc='lower left')
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"task_accuracy_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_hierarchy_comparison(
    comparison_results: Dict[str, Any],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Plot performance comparison across different hierarchy levels.
    
    X-axis: Number of hierarchy levels
    Y-axis: Mean accuracy
    """
    summary = comparison_results["summary"]
    levels = sorted(summary.keys())
    means = [summary[l]["mean_acc"] for l in levels]
    stds = [summary[l]["std_acc"] for l in levels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot with error bars
    bars = ax.bar(
        range(len(levels)), means, yerr=stds,
        color=[HIERARCHY_COLORS[i % len(HIERARCHY_COLORS)] for i in range(len(levels))],
        capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlabel("Number of Hierarchy Levels", fontweight='bold')
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title(f"HTCL Performance vs Hierarchy Depth on {dataset}", fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"L={l}" for l in levels])
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
            f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold'
        )
    
    # Set y-axis limits with padding
    ax.set_ylim(0, max(means) + max(stds) + 10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"hierarchy_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_hierarchy_line_comparison(
    comparison_results: Dict[str, Any],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Line plot showing per-task accuracy for different hierarchy levels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hierarchy_results = comparison_results["hierarchy_comparison"]
    levels = sorted(hierarchy_results.keys())
    
    for i, level in enumerate(levels):
        result = hierarchy_results[level]
        per_task_mean = result["summary"]["per_task_mean"]
        per_task_std = result["summary"]["per_task_std"]
        
        tasks = range(1, len(per_task_mean) + 1)
        
        color = HIERARCHY_COLORS[i % len(HIERARCHY_COLORS)]
        ax.plot(tasks, per_task_mean, 'o-', color=color, label=f'L={level}',
                linewidth=2, markersize=8)
        ax.fill_between(
            tasks,
            np.array(per_task_mean) - np.array(per_task_std),
            np.array(per_task_mean) + np.array(per_task_std),
            alpha=0.2, color=color
        )
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_title(f"Per-Task Accuracy by Hierarchy Level on {dataset}", 
                 fontweight='bold', fontsize=14)
    ax.legend(title="Hierarchy Levels", loc='lower left')
    ax.set_xticks(range(1, len(per_task_mean) + 1))
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"hierarchy_line_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_task_order_robustness(
    er_results: Dict[str, Any],
    htcl_results: Dict[str, Any],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Violin plot showing distribution of accuracies across task orderings.
    """
    num_tasks = len(er_results["summary"]["per_task_mean"])
    
    # Prepare data
    data_records = []
    
    for r in er_results["results"]:
        for i, acc in enumerate(r["accuracies"]):
            data_records.append({
                "Method": "ER",
                "Task": f"Task {i+1}",
                "Accuracy": acc
            })
    
    for r in htcl_results["results"]:
        for i, acc in enumerate(r["accuracies"]):
            data_records.append({
                "Method": "HTCL",
                "Task": f"Task {i+1}",
                "Accuracy": acc
            })
    
    df = pd.DataFrame(data_records)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(
        data=df, x="Task", y="Accuracy", hue="Method",
        palette={"ER": COLORS["er"], "HTCL": COLORS["htcl_2"]},
        ax=ax, split=True, inner="quart", cut=0
    )
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_title(f"Task-Order Robustness: ER vs HTCL on {dataset}", 
                 fontweight='bold', fontsize=14)
    ax.legend(title="Method", loc='lower left')
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"task_order_robustness_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_forgetting_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Bar chart comparing forgetting across different methods.
    """
    from ..utils import compute_avg_forgetting
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    forgetting_values = []
    colors = [COLORS["er"], COLORS["htcl_2"], COLORS["htcl_3"], COLORS["htcl_4"]]
    
    for result in results_list:
        # Compute forgetting for each result
        acc_matrices = [r["accuracies"] for r in result["results"]]
        if acc_matrices:
            # Simple approximation: difference between max and final accuracy per task
            per_task_forgetting = []
            num_tasks = len(acc_matrices[0])
            for task_id in range(num_tasks - 1):  # Exclude last task
                task_accs = [m[task_id] for m in acc_matrices]
                if task_accs:
                    per_task_forgetting.append(max(task_accs) - min(task_accs))
            forgetting_values.append(np.mean(per_task_forgetting) if per_task_forgetting else 0)
        else:
            forgetting_values.append(0)
    
    bars = ax.bar(
        range(len(labels)), forgetting_values,
        color=colors[:len(labels)], alpha=0.8,
        edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Average Forgetting (%)", fontweight='bold')
    ax.set_title(f"Forgetting Comparison on {dataset}", fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    
    # Add value labels
    for bar, val in zip(bars, forgetting_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold'
        )
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"forgetting_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_mean_accuracy_with_std(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Bar chart of mean accuracy with standard deviation error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [r["summary"]["mean_accuracy"] for r in results_list]
    stds = [r["summary"]["std_accuracy"] for r in results_list]
    colors = [COLORS["er"]] + HIERARCHY_COLORS[:len(labels)-1]
    
    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stds,
        color=colors[:len(labels)], alpha=0.8,
        capsize=6, edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title(f"Mean Accuracy Comparison on {dataset}", fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
            f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9
        )
    
    ax.set_ylim(0, max(means) + max(stds) + 15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"mean_accuracy_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_comprehensive_comparison(
    er_results: Dict[str, Any],
    htcl_results_by_level: Dict[int, Dict[str, Any]],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create a comprehensive 2x2 comparison figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get data
    levels = sorted(htcl_results_by_level.keys())
    
    # Plot 1: Mean accuracy comparison
    ax = axes[0, 0]
    labels = ["ER"] + [f"HTCL-L{l}" for l in levels]
    means = [er_results["summary"]["mean_accuracy"]] + \
            [htcl_results_by_level[l]["summary"]["mean_accuracy"] for l in levels]
    stds = [er_results["summary"]["std_accuracy"]] + \
           [htcl_results_by_level[l]["summary"]["std_accuracy"] for l in levels]
    
    colors = [COLORS["er"]] + HIERARCHY_COLORS[:len(levels)]
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors, 
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title("(a) Mean Accuracy Comparison", fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Standard deviation (task-order sensitivity)
    ax = axes[0, 1]
    ax.bar(range(len(labels)), stds, color=colors, alpha=0.8, 
           edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Std. Deviation (%)", fontweight='bold')
    ax.set_title("(b) Task-Order Sensitivity", fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Per-task accuracy for best HTCL vs ER
    ax = axes[1, 0]
    num_tasks = len(er_results["summary"]["per_task_mean"])
    tasks = range(1, num_tasks + 1)
    
    # ER
    ax.plot(tasks, er_results["summary"]["per_task_mean"], 'o-', 
            color=COLORS["er"], label="ER", linewidth=2, markersize=6)
    
    # Best HTCL (highest mean accuracy)
    best_level = max(levels, key=lambda l: htcl_results_by_level[l]["summary"]["mean_accuracy"])
    ax.plot(tasks, htcl_results_by_level[best_level]["summary"]["per_task_mean"], 's-',
            color=COLORS["htcl_2"], label=f"HTCL-L{best_level}", linewidth=2, markersize=6)
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_title("(c) Per-Task Accuracy", fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xticks(tasks)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Hierarchy effect
    ax = axes[1, 1]
    level_means = [htcl_results_by_level[l]["summary"]["mean_accuracy"] for l in levels]
    level_stds = [htcl_results_by_level[l]["summary"]["std_accuracy"] for l in levels]
    
    ax.errorbar(levels, level_means, yerr=level_stds, fmt='o-', 
                color=COLORS["primary"], capsize=5, linewidth=2, markersize=8)
    ax.axhline(y=er_results["summary"]["mean_accuracy"], color=COLORS["er"], 
               linestyle='--', linewidth=2, label="ER baseline")
    
    ax.set_xlabel("Hierarchy Levels (L)", fontweight='bold')
    ax.set_ylabel("Mean Accuracy (%)", fontweight='bold')
    ax.set_title("(d) Effect of Hierarchy Depth", fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xticks(levels)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f"Comprehensive Comparison on {dataset}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, f"comprehensive_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_accuracy_heatmap(
    results: Dict[str, Any],
    method_name: str,
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Heatmap showing accuracy for each permutation and task.
    """
    # Prepare data
    acc_matrix = np.array([r["accuracies"] for r in results["results"]])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        acc_matrix, ax=ax, cmap="RdYlGn", annot=True, fmt=".1f",
        xticklabels=[f"T{i+1}" for i in range(acc_matrix.shape[1])],
        yticklabels=[f"P{i+1}" for i in range(acc_matrix.shape[0])],
        vmin=0, vmax=100,
        cbar_kws={"label": "Accuracy (%)"}
    )
    
    ax.set_xlabel("Task", fontweight='bold')
    ax.set_ylabel("Permutation", fontweight='bold')
    ax.set_title(f"{method_name} Accuracy Heatmap on {dataset}", fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    save_figure(fig, f"accuracy_heatmap_{method_name}_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_time_comparison(
    er_results: Dict[str, Any],
    htcl_results_by_level: Dict[int, Dict[str, Any]],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Create a time comparison plot between ER baseline and HTCL variants.
    
    Shows both total time and average time per permutation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gather timing data
    methods = ["ER (Baseline)"]
    levels = sorted(htcl_results_by_level.keys())
    methods += [f"ER + HTCL-L{l}" for l in levels]
    
    # Total time
    total_times = [er_results.get("total_time_seconds", 0)]
    for l in levels:
        htcl_time = htcl_results_by_level[l].get("total_time_seconds", 0)
        total_times.append(htcl_time)
    
    # Average time per permutation
    avg_times = [er_results.get("summary", {}).get("avg_time_per_perm", 0)]
    for l in levels:
        htcl_avg = htcl_results_by_level[l].get("summary", {}).get("avg_time_per_perm", 0)
        avg_times.append(htcl_avg)
    
    colors = [COLORS["er"]] + HIERARCHY_COLORS[:len(levels)]
    
    # Plot 1: Total time
    ax = axes[0]
    bars = ax.bar(range(len(methods)), total_times, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Total Time (seconds)", fontweight='bold')
    ax.set_title(f"Total Experiment Time on {dataset}", fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Average time per permutation
    ax = axes[1]
    bars = ax.bar(range(len(methods)), avg_times, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_xlabel("Method", fontweight='bold')
    ax.set_ylabel("Avg Time per Permutation (seconds)", fontweight='bold')
    ax.set_title(f"Average Time per Permutation on {dataset}", fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"time_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def plot_hierarchy_time_comparison(
    htcl_results_by_level: Dict[int, Dict[str, Any]],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
) -> plt.Figure:
    """
    Bar chart comparing time across different hierarchy levels.
    """
    levels = sorted(htcl_results_by_level.keys())
    
    total_times = [htcl_results_by_level[l].get("total_time_seconds", 0) for l in levels]
    avg_times = [htcl_results_by_level[l].get("summary", {}).get("avg_time_per_perm", 0) for l in levels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(levels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_times, width, label='Total Time',
                   color=COLORS["primary"], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, avg_times, width, label='Avg per Perm',
                   color=COLORS["accent"], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel("Hierarchy Levels", fontweight='bold')
    ax.set_ylabel("Time (seconds)", fontweight='bold')
    ax.set_title(f"HTCL Computation Time by Hierarchy Depth on {dataset}", 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={l}" for l in levels])
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, f"hierarchy_time_comparison_{dataset}", output_dir)
    
    if show:
        plt.show()
    
    return fig


def create_all_visualizations(
    er_results: Dict[str, Any],
    htcl_results_by_level: Dict[int, Dict[str, Any]],
    dataset: str,
    output_dir: str = "./results",
    show: bool = False,
):
    """
    Create all visualization types for a complete experiment.
    """
    print(f"\nGenerating visualizations for {dataset}...")
    
    # Get a representative HTCL result (e.g., L=2)
    htcl_l2 = htcl_results_by_level.get(2, list(htcl_results_by_level.values())[0])
    
    # 1. Task accuracy comparison
    plot_task_accuracy_comparison(er_results, htcl_l2, dataset, output_dir, show)
    
    # 2. Hierarchy comparison
    comparison_results = {
        "dataset": dataset,
        "hierarchy_comparison": htcl_results_by_level,
        "summary": {
            l: {"mean_acc": r["summary"]["mean_accuracy"], "std_acc": r["summary"]["std_accuracy"]}
            for l, r in htcl_results_by_level.items()
        }
    }
    plot_hierarchy_comparison(comparison_results, dataset, output_dir, show)
    plot_hierarchy_line_comparison(comparison_results, dataset, output_dir, show)
    
    # 3. Task-order robustness
    plot_task_order_robustness(er_results, htcl_l2, dataset, output_dir, show)
    
    # 4. Mean accuracy comparison
    labels = ["ER"] + [f"HTCL-L{l}" for l in sorted(htcl_results_by_level.keys())]
    results_list = [er_results] + [htcl_results_by_level[l] for l in sorted(htcl_results_by_level.keys())]
    plot_mean_accuracy_with_std(results_list, labels, dataset, output_dir, show)
    
    # 5. Comprehensive comparison
    plot_comprehensive_comparison(er_results, htcl_results_by_level, dataset, output_dir, show)
    
    # 6. Heatmaps
    plot_accuracy_heatmap(er_results, "ER", dataset, output_dir, show)
    plot_accuracy_heatmap(htcl_l2, "HTCL-L2", dataset, output_dir, show)
    
    # 7. Time comparisons
    plot_time_comparison(er_results, htcl_results_by_level, dataset, output_dir, show)
    plot_hierarchy_time_comparison(htcl_results_by_level, dataset, output_dir, show)
    
    print(f"All visualizations saved to {output_dir}/plots/")
