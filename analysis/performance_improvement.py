from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
results_dir = Path("../results")
figures_dir = Path("../figures")
figures_dir.mkdir(parents=True, exist_ok=True)

der_csv = results_dir / "der_permutation_results.csv"
taylor_csv = results_dir / "taylor_permutation_results.csv"

if not der_csv.exists() or not taylor_csv.exists():
    raise FileNotFoundError(f"Missing CSVs in {results_dir}. Expected: {der_csv}, {taylor_csv}")

# Read
der = pd.read_csv(der_csv)
taylor = pd.read_csv(taylor_csv)

# Task columns detection (columns starting with "Task"), fallback to all except 'sequence'
task_cols = [c for c in der.columns if str(c).lower().startswith("task")]
if not task_cols:
    task_cols = [c for c in der.columns if c.lower() not in ("sequence", "seq", "permutation")]

# Ensure both have same task cols
missing = [c for c in task_cols if c not in taylor.columns]
if missing:
    raise ValueError(f"Task columns {missing} missing in {taylor_csv}")

# Means and stds
der_mean = der[task_cols].mean(axis=0)
der_std  = der[task_cols].std(axis=0)
tay_mean = taylor[task_cols].mean(axis=0)
tay_std  = taylor[task_cols].std(axis=0)

# Save summary CSV
summary = pd.DataFrame({
    "task": task_cols,
    "der_mean": der_mean.values,
    "der_std": der_std.values,
    "taylor_mean": tay_mean.values,
    "taylor_std": tay_std.values
}).set_index("task")
summary_csv = figures_dir / "mean_accuracies_by_task.csv"
summary.to_csv(summary_csv)

# Plotting (professional look)
plt.style.use('classic')  # keep neutral; change to 'ggplot' if you prefer
fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(task_cols))
width = 0.38

# Bars
bars1 = ax.bar(x - width/2, der_mean.values, width,
               yerr=der_std.values, capsize=6, label='DER', linewidth=0.6)
bars2 = ax.bar(x + width/2, tay_mean.values, width,
               yerr=tay_std.values, capsize=6, label='Taylor', linewidth=0.6)

# Axis labels and title
# ax.set_xticks(x)
# ax.set_xticklabels(task_cols, rotation=35, ha='right', fontsize=10)
# ax.set_ylabel("Mean accuracy")
# ax.set_title("Mean accuracy per task: DER vs Taylor", fontsize=13, fontweight='semibold')

# Grid, legend, nice spines
# ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
# ax.set_axisbelow(True)
# ax.legend(frameon=False, fontsize=10)

# Add numeric labels on top of bars (rounded)
def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
svg_path = figures_dir / "mean_accuracies_by_task.svg"
plt.savefig(svg_path, format='svg')
plt.close(fig)

print("Saved:")
print(f" - Summary CSV: {summary_csv}")
print(f" - SVG figure:  {svg_path}")
