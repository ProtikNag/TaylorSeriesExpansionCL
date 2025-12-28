#!/usr/bin/env python3
"""
Alternative Visualizations for FL Comparison Results.

Generates publication-quality alternative figures for:
1. Taskwise accuracy - Radar chart and heatmap
2. Task-order sensitivity (std dev) - Lollipop chart and dot plot

Usage:
    python generate_fl_visualizations.py --results-dir ./results/splitmnist/fl_comparison

Or just run directly (uses default path):
    python generate_fl_visualizations.py
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# Color scheme - colorblind friendly
COLORS = {
    'baseline': '#636363',  # Dark gray
    'htcl': '#d62728',  # Red
    'fedavg': '#1f77b4',  # Blue
    'fedprox': '#2ca02c',  # Green
}

MARKERS = {
    'baseline': 'o',
    'htcl': 's',
    'fedavg': '^',
    'fedprox': 'D',
}

METHOD_LABELS = {
    'baseline': 'Baseline',
    'htcl': 'HTCL (Ours)',
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
}


def load_results(results_dir: str) -> dict:
    """Load all CSV results from the directory."""
    csv_dir = os.path.join(results_dir, 'csv')

    if not os.path.exists(csv_dir):
        print(f"Warning: CSV directory not found: {csv_dir}")
        return {}

    results = {}

    # Auto-discover CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        print(f"  - {f}")
    print()

    for filename in csv_files:
        filepath = os.path.join(csv_dir, filename)
        df = pd.read_csv(filepath)

        # Determine key based on filename
        fname_lower = filename.lower()

        if 'fedavg' in fname_lower and 'ser' in fname_lower:
            results['ser_fedavg'] = df
            print(f"  {filename} -> ser_fedavg")
        elif 'fedavg' in fname_lower and 'der' in fname_lower:
            results['der_fedavg'] = df
            print(f"  {filename} -> der_fedavg")
        elif 'fedprox' in fname_lower and 'ser' in fname_lower:
            results['ser_fedprox'] = df
            print(f"  {filename} -> ser_fedprox")
        elif 'fedprox' in fname_lower and 'der' in fname_lower:
            results['der_fedprox'] = df
            print(f"  {filename} -> der_fedprox")
        elif 'htcl' in fname_lower:
            # HTCL file - use for both SER and DER HTCL
            results['ser_htcl'] = df
            results['der_htcl'] = df
            print(f"  {filename} -> ser_htcl, der_htcl")
        elif 'ser_results' in fname_lower or (fname_lower.startswith('ser_') and 'fedavg' not in fname_lower and 'fedprox' not in fname_lower):
            results['ser_baseline'] = df
            print(f"  {filename} -> ser_baseline")
        elif 'der_results' in fname_lower or (fname_lower.startswith('der_') and 'fedavg' not in fname_lower and 'fedprox' not in fname_lower):
            results['der_baseline'] = df
            print(f"  {filename} -> der_baseline")
        else:
            print(f"  {filename} -> (not mapped)")

    # Print summary
    print(f"\nMapped results: {sorted(results.keys())}")

    # Check for missing baselines and warn
    expected = ['ser_baseline', 'ser_htcl', 'ser_fedavg', 'ser_fedprox',
                'der_baseline', 'der_htcl', 'der_fedavg', 'der_fedprox']
    missing = [k for k in expected if k not in results]
    if missing:
        print(f"\n⚠️  MISSING: {missing}")
        if 'der_baseline' in missing:
            print("   -> DER baseline missing! Need 'der_results_SplitMNIST.csv'")
        if 'ser_baseline' in missing:
            print("   -> SER baseline missing! Need 'ser_results_SplitMNIST.csv'")

    return results


def extract_task_stats(results: dict) -> dict:
    """Extract per-task mean and std from results."""
    stats = {}

    for key, df in results.items():
        # Find task columns (Task1, Task2, etc. but NOT Mean)
        # Handle both 'Task1' and 'Task 1' formats
        task_cols = []
        for col in df.columns:
            if col.startswith('Task'):
                # Extract the number part
                num_part = col.replace('Task', '').strip()
                if num_part.isdigit():
                    task_cols.append(col)

        task_cols = sorted(task_cols, key=lambda x: int(x.replace('Task', '').strip()))

        if not task_cols:
            print(f"Warning: No task columns found in {key}. Columns: {list(df.columns)}")
            continue

        print(f"  {key}: {len(task_cols)} tasks detected ({task_cols})")

        stats[key] = {
            'task_means': [df[col].mean() for col in task_cols],
            'task_stds': [df[col].std() for col in task_cols],
            'overall_mean': df['Mean'].mean() if 'Mean' in df.columns else np.mean([df[col].mean() for col in task_cols]),
            'overall_std': df['Mean'].std() if 'Mean' in df.columns else np.std([df[col].mean() for col in task_cols]),
            'num_tasks': len(task_cols),
        }

    return stats


def plot_radar_chart(stats: dict, output_dir: str):
    """
    Create radar/spider chart for taskwise accuracy comparison.
    Shows performance profile across all tasks for each method.
    """
    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for radar chart")
        return

    num_plots = len(baselines)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6),
                             subplot_kw=dict(projection='polar'))

    if num_plots == 1:
        axes = [axes]

    methods = ['baseline', 'htcl', 'fedavg', 'fedprox']

    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]

        # Find the reference number of tasks
        num_tasks = None
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                num_tasks = stats[full_key]['num_tasks']
                break

        if num_tasks is None:
            continue

        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        # Plot each method
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key not in stats:
                continue

            values = stats[full_key]['task_means'].copy()  # Make a copy!

            # Check for task count mismatch
            if len(values) != num_tasks:
                print(f"Warning: {full_key} has {len(values)} tasks, expected {num_tasks}. Skipping.")
                continue

            values += values[:1]  # Complete the loop

            color = COLORS[method]
            label = METHOD_LABELS[method]

            ax.plot(angles, values, 'o-', color=color, label=label,
                    linewidth=2, markersize=6, alpha=0.9)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Configure radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'Task {i + 1}' for i in range(num_tasks)], fontsize=11)
        ax.set_ylim(50, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=9)
        ax.set_title(f'{baseline.upper()} Base', fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend
    axes[0].legend(loc='upper left', bbox_to_anchor=(-0.1, 1.15), ncol=4, frameon=True)

    plt.suptitle('Task Performance Profile Comparison', fontsize=16, fontweight='bold', y=1.08)
    plt.tight_layout()

    # Save
    save_figure(fig, 'fl_radar_taskwise', output_dir)
    plt.close(fig)


def plot_heatmap(stats: dict, output_dir: str):
    """
    Create heatmap showing accuracy across tasks and methods.
    """
    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for heatmap")
        return

    num_plots = len(baselines)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    methods = ['baseline', 'htcl', 'fedavg', 'fedprox']
    method_labels = [METHOD_LABELS[m] for m in methods]

    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]

        # Find the reference number of tasks
        num_tasks = None
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                num_tasks = stats[full_key]['num_tasks']
                break

        if num_tasks is None:
            continue

        # Build data matrix
        data = []
        valid_methods = []
        valid_labels = []
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                task_means = stats[full_key]['task_means']
                if len(task_means) == num_tasks:
                    data.append(task_means)
                    valid_methods.append(method)
                    valid_labels.append(METHOD_LABELS[method])
                else:
                    print(f"Warning: {full_key} has {len(task_means)} tasks, expected {num_tasks}. Skipping.")
            else:
                # Add NaN row for missing methods
                data.append([np.nan] * num_tasks)
                valid_methods.append(method)
                valid_labels.append(METHOD_LABELS[method])

        if not data:
            continue

        data = np.array(data)

        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

        # Add text annotations
        for i in range(len(valid_methods)):
            for j in range(num_tasks):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 70 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=11, fontweight='bold', color=text_color)

        # Configure axes
        ax.set_xticks(np.arange(num_tasks))
        ax.set_xticklabels([f'Task {i + 1}' for i in range(num_tasks)])
        ax.set_yticks(np.arange(len(valid_methods)))
        ax.set_yticklabels(valid_labels)
        ax.set_xlabel('Task', fontweight='bold')
        ax.set_title(f'{baseline.upper()} Base', fontsize=14, fontweight='bold')

        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Accuracy (%)', fontweight='bold')

    plt.suptitle('Taskwise Accuracy Heatmap', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'fl_heatmap_taskwise', output_dir)
    plt.close(fig)


def plot_lollipop_std(stats: dict, output_dir: str):
    """
    Create lollipop chart for standard deviation comparison.
    More elegant than bar chart for showing rankings.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for lollipop chart")
        return

    # Prepare data
    data = []
    methods = ['baseline', 'htcl', 'fedavg', 'fedprox']

    for baseline in baselines:
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                data.append({
                    'label': f"{baseline.upper()}\n{METHOD_LABELS[method]}",
                    'std': stats[full_key]['overall_std'],
                    'method': method,
                    'baseline': baseline,
                })

    # Sort by std (ascending - lower is better)
    data = sorted(data, key=lambda x: x['std'])

    # Create lollipop chart
    y_pos = np.arange(len(data))

    for i, d in enumerate(data):
        color = COLORS[d['method']]
        marker = MARKERS[d['method']]

        # Draw stem
        ax.hlines(y=i, xmin=0, xmax=d['std'], color=color, alpha=0.7, linewidth=2.5)

        # Draw lollipop head
        ax.scatter(d['std'], i, color=color, s=200, marker=marker,
                   edgecolors='black', linewidths=1.5, zorder=5)

        # Add value label
        ax.text(d['std'] + 0.15, i, f"{d['std']:.2f}%", va='center',
                fontsize=10, fontweight='bold')

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d['label'] for d in data])
    ax.set_xlabel('Standard Deviation (%)', fontweight='bold', fontsize=12)
    ax.set_title('Task-Order Sensitivity Ranking\n(Lower = More Robust to Task Ordering)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(d['std'] for d in data) + 1.5)

    # Add vertical reference line at mean
    mean_std = np.mean([d['std'] for d in data])
    ax.axvline(x=mean_std, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(mean_std + 0.1, len(data) - 0.5, f'Mean: {mean_std:.2f}%',
            fontsize=9, color='gray')

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker=MARKERS[m], color='w', markerfacecolor=COLORS[m],
               markersize=10, label=METHOD_LABELS[m], markeredgecolor='black')
        for m in methods
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)

    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Highlight best performer
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.get_yticklabels()[0].set_color(COLORS[data[0]['method']])

    plt.tight_layout()

    save_figure(fig, 'fl_lollipop_std', output_dir)
    plt.close(fig)


def plot_dot_std_comparison(stats: dict, output_dir: str):
    """
    Create dot plot with confidence-style visualization for std comparison.
    Shows reduction from baseline clearly.
    """
    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for dot plot")
        return

    num_plots = len(baselines)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    methods = ['baseline', 'htcl', 'fedavg', 'fedprox']

    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]
        baseline_upper = baseline.upper()

        # Get baseline std
        baseline_key = f"{baseline}_baseline"
        if baseline_key not in stats:
            continue

        baseline_std = stats[baseline_key]['overall_std']

        # Prepare data for this baseline
        method_data = []
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                method_data.append({
                    'method': method,
                    'std': stats[full_key]['overall_std'],
                    'reduction': baseline_std - stats[full_key]['overall_std'],
                    'reduction_pct': (baseline_std - stats[full_key]['overall_std']) / baseline_std * 100,
                })

        y_pos = np.arange(len(method_data))

        # Draw connecting lines from baseline
        for i, d in enumerate(method_data):
            if d['method'] != 'baseline':
                # Draw line from baseline to this method
                ax.plot([baseline_std, d['std']], [0, i],
                        color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Draw dots
        for i, d in enumerate(method_data):
            color = COLORS[d['method']]
            marker = MARKERS[d['method']]

            ax.scatter(d['std'], i, color=color, s=300, marker=marker,
                       edgecolors='black', linewidths=2, zorder=5,
                       label=METHOD_LABELS[d['method']] if ax_idx == 0 else None)

            # Add reduction annotation for non-baseline
            if d['method'] != 'baseline' and d['reduction'] > 0:
                ax.annotate(f"↓{d['reduction_pct']:.0f}%",
                            xy=(d['std'], i), xytext=(d['std'] - 0.8, i + 0.3),
                            fontsize=9, color=color, fontweight='bold')

        # Configure axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels([METHOD_LABELS[d['method']] for d in method_data])
        ax.set_xlabel('Standard Deviation (%)', fontweight='bold')
        ax.set_title(f'{baseline_upper} Base', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(d['std'] for d in method_data) + 2)

        # Add baseline reference line
        ax.axvline(x=baseline_std, color=COLORS['baseline'], linestyle=':',
                   alpha=0.5, linewidth=2)

        # Grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker=MARKERS[m], color='w', markerfacecolor=COLORS[m],
               markersize=12, label=METHOD_LABELS[m], markeredgecolor='black', linewidth=0)
        for m in methods
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=4, frameon=True)

    plt.suptitle('Task-Order Sensitivity: Reduction from Baseline\n(Lower = More Robust)',
                 fontsize=14, fontweight='bold', y=1.15)
    plt.tight_layout()

    save_figure(fig, 'fl_dot_std_reduction', output_dir)
    plt.close(fig)


def plot_dumbbell_std(stats: dict, output_dir: str):
    """
    Create dumbbell chart comparing baseline std to each consolidation method.
    Clearly shows the improvement (or regression) from baseline.
    """
    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for dumbbell chart")
        return

    # Prepare data: one row per (baseline, method) pair excluding baseline itself
    consolidation_methods = ['htcl', 'fedavg', 'fedprox']

    rows = []
    for baseline in baselines:
        baseline_key = f"{baseline}_baseline"
        if baseline_key not in stats:
            continue
        baseline_std = stats[baseline_key]['overall_std']

        for method in consolidation_methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                method_std = stats[full_key]['overall_std']
                rows.append({
                    'label': f"{baseline.upper()} + {METHOD_LABELS[method]}",
                    'baseline_std': baseline_std,
                    'method_std': method_std,
                    'method': method,
                    'baseline': baseline,
                    'reduction': baseline_std - method_std,
                    'reduction_pct': (baseline_std - method_std) / baseline_std * 100,
                })

    if not rows:
        print("No data available for dumbbell chart")
        return

    # Sort by reduction (best improvement first)
    rows = sorted(rows, key=lambda x: -x['reduction'])

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(rows))

    # Track which methods we've added to legend
    legend_added = {'baseline': False}
    for m in consolidation_methods:
        legend_added[m] = False

    for i, row in enumerate(rows):
        baseline_std = row['baseline_std']
        method_std = row['method_std']
        method = row['method']

        # Draw connecting line (the "dumbbell bar")
        line_color = '#2ecc71' if row['reduction'] > 0 else '#e74c3c'  # Green if improved, red if worse
        ax.plot([baseline_std, method_std], [i, i],
                color=line_color, linewidth=3, alpha=0.7, zorder=1)

        # Draw baseline point (gray circle)
        label_baseline = 'Baseline' if not legend_added['baseline'] else None
        ax.scatter(baseline_std, i, color=COLORS['baseline'], s=150,
                   marker='o', edgecolors='black', linewidths=1.5, zorder=3,
                   label=label_baseline)
        legend_added['baseline'] = True

        # Draw method point (colored by method)
        label_method = METHOD_LABELS[method] if not legend_added[method] else None
        ax.scatter(method_std, i, color=COLORS[method], s=150,
                   marker=MARKERS[method], edgecolors='black', linewidths=1.5, zorder=3,
                   label=label_method)
        legend_added[method] = True

        # Add reduction percentage annotation
        mid_x = (baseline_std + method_std) / 2
        if row['reduction'] > 0:
            ax.annotate(f"↓{row['reduction_pct']:.0f}%",
                        xy=(mid_x, i), xytext=(mid_x, i - 0.35),
                        fontsize=9, fontweight='bold', color=line_color,
                        ha='center', va='top')
        elif row['reduction'] < -0.5:  # Only show if significantly worse
            ax.annotate(f"↑{abs(row['reduction_pct']):.0f}%",
                        xy=(mid_x, i), xytext=(mid_x, i - 0.35),
                        fontsize=9, fontweight='bold', color=line_color,
                        ha='center', va='top')

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row['label'] for row in rows])
    ax.set_xlabel('Standard Deviation (%) — Lower is Better', fontweight='bold', fontsize=12)
    ax.set_title('Task-Order Sensitivity: Baseline → Consolidation Method\n(Green = Improvement, Red = Regression)',
                 fontsize=14, fontweight='bold')

    # Set x limits with padding
    all_stds = [row['baseline_std'] for row in rows] + [row['method_std'] for row in rows]
    ax.set_xlim(0, max(all_stds) + 1.5)

    # Add legend
    ax.legend(loc='lower right', frameon=True)

    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_figure(fig, 'fl_dumbbell_std', output_dir)
    plt.close(fig)


def plot_line_taskwise(stats: dict, output_dir: str):
    """
    Create line plot showing task accuracy progression.
    Better for showing trends across tasks.
    """
    # Determine which baselines have data
    baselines = []
    if any(k.startswith('ser_') for k in stats.keys()):
        baselines.append('ser')
    if any(k.startswith('der_') for k in stats.keys()):
        baselines.append('der')

    if not baselines:
        print("No data available for line plot")
        return

    num_plots = len(baselines)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    methods = ['baseline', 'htcl', 'fedavg', 'fedprox']

    for ax_idx, baseline in enumerate(baselines):
        ax = axes[ax_idx]
        baseline_upper = baseline.upper()

        # Find the reference number of tasks (from any available method for this baseline)
        num_tasks = None
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key in stats:
                num_tasks = stats[full_key]['num_tasks']
                break

        if num_tasks is None:
            print(f"No data found for {baseline_upper}")
            continue

        x = np.arange(1, num_tasks + 1)

        # Plot each method
        for method in methods:
            full_key = f"{baseline}_{method}"
            if full_key not in stats:
                continue

            means = stats[full_key]['task_means']
            stds = stats[full_key]['task_stds']

            # Check for task count mismatch
            if len(means) != num_tasks:
                print(f"Warning: {full_key} has {len(means)} tasks, expected {num_tasks}. Skipping.")
                continue

            color = COLORS[method]
            marker = MARKERS[method]
            label = METHOD_LABELS[method]

            # Plot line with markers
            ax.plot(x, means, marker=marker, color=color, label=label,
                    linewidth=2.5, markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5)

            # Add shaded error region
            ax.fill_between(x,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=color, alpha=0.15)

        # Configure axes
        ax.set_xlabel('Task', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'{baseline_upper} Base', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'T{i}' for i in x])
        ax.set_ylim(50, 100)
        ax.set_xlim(0.5, num_tasks + 0.5)

        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # Legend
        if ax_idx == 0:
            ax.legend(loc='lower left', frameon=True)

    plt.suptitle('Task Accuracy Progression\n(Shaded regions show ±1 std across permutations)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    save_figure(fig, 'fl_line_taskwise', output_dir)
    plt.close(fig)


def save_figure(fig: plt.Figure, name: str, output_dir: str):
    """Save figure in both PNG and SVG formats."""
    # Create output directories
    png_dir = os.path.join(output_dir, 'plots', 'png')
    svg_dir = os.path.join(output_dir, 'plots', 'svg')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    png_path = os.path.join(png_dir, f'{name}.png')
    svg_path = os.path.join(svg_dir, f'{name}.svg')

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate alternative FL comparison visualizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='./results/splitmnist/fl_comparison',
        help='Directory containing CSV results'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Alternative FL Comparison Visualizations")
    print("=" * 60)
    print(f"\nResults directory: {args.results_dir}")

    # Load results
    print("\n--- Loading Results ---")
    results = load_results(args.results_dir)

    if not results:
        print("ERROR: No results found. Check the results directory.")
        return

    # Extract statistics
    print("\n--- Extracting Statistics ---")
    stats = extract_task_stats(results)

    # Generate visualizations
    print("\n--- Generating Visualizations ---")

    print("\n1. Radar chart (taskwise accuracy)...")
    plot_radar_chart(stats, args.results_dir)

    print("\n2. Heatmap (taskwise accuracy)...")
    plot_heatmap(stats, args.results_dir)

    print("\n3. Line plot (taskwise accuracy)...")
    plot_line_taskwise(stats, args.results_dir)

    print("\n4. Lollipop chart (std dev ranking)...")
    plot_lollipop_std(stats, args.results_dir)

    print("\n5. Dot plot (std dev reduction)...")
    plot_dot_std_comparison(stats, args.results_dir)

    print("\n6. Dumbbell chart (std dev comparison)...")
    plot_dumbbell_std(stats, args.results_dir)

    print("\n" + "=" * 60)
    print("Done! New visualizations saved to:")
    print(f"  PNG: {args.results_dir}/plots/png/")
    print(f"  SVG: {args.results_dir}/plots/svg/")
    print("\nNew files generated:")
    print("  - fl_radar_taskwise.png/svg    (Radar chart)")
    print("  - fl_heatmap_taskwise.png/svg  (Heatmap)")
    print("  - fl_line_taskwise.png/svg     (Line plot with shaded std)")
    print("  - fl_lollipop_std.png/svg      (Lollipop ranking chart)")
    print("  - fl_dot_std_reduction.png/svg (Dot plot with reduction %)")
    print("  - fl_dumbbell_std.png/svg      (Dumbbell chart baseline→method)")
    print("=" * 60)


if __name__ == "__main__":
    main()