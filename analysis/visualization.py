"""
plot_permutation_boxplots.py

Reads a CSV of permutations (one row per permutation; 'sequence' column
is like "(0, 1, 2, 3, 4)" and Task1..Task5 hold the accuracy/metric
for the 1st,2nd,...,5th task in that permutation). Produces publication-quality
boxplots aligned to absolute task identity and computes robust variance
(ignoring extreme values).

Outputs:
 - figures/boxplot_tasks.pdf and .png
 - figures/boxplot_tasks_trimmed.png (alternate)
 - figures/variance_table.csv

Author: adapted for your dataset
"""

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean

# ------------------------- USER CONFIG -------------------------
CSV_PATH = "der_permutation_results.csv"   # set path to your CSV
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Appearance settings (academic)
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 12
FIGSIZE = (7.0, 4.5)     # compact, wide figure
TRIM_PERCENT = 5.0       # percent trim for variance (per tail). Set 0 to disable trimming
SHOW_FLIERS = False      # if False, matplotlib will hide the plotted outliers
COLOR_PALETTE = "tab10"  # standard matplotlib/seaborn palette
RANDOM_SEED = 0
# ----------------------------------------------------------------

# set global matplotlib params
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
})

def parse_sequence_column(s):
    """Parse string like '(0, 1, 2, 3, 4)' into a tuple of ints."""
    if pd.isna(s):
        return ()
    # ast.literal_eval is safe for this format
    try:
        t = ast.literal_eval(s)
        return tuple(int(x) for x in t)
    except Exception:
        # fallback: strip punctuation and split
        s2 = s.strip("()[] ")
        return tuple(int(x) for x in s2.split(",") if x.strip() != "")

def load_and_remap(csv_path):
    """Load CSV and remap Task1..TaskK entries into per-absolute-task lists."""
    df = pd.read_csv(csv_path)
    # find columns named like Task1, Task2, ...
    task_pos_cols = [c for c in df.columns if c.lower().startswith("task")]
    task_pos_cols = sorted(task_pos_cols, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    if "sequence" not in df.columns:
        raise ValueError("'sequence' column not found in CSV.")
    K = len(task_pos_cols)
    # determine absolute task IDs (assume permutations contain ints 0..K-1)
    per_task_values = {}  # map absolute_task_id -> list of values
    for idx, row in df.iterrows():
        seq = parse_sequence_column(row["sequence"])
        if len(seq) != K:
            # tolerate extra whitespace or different formatting by skipping or warning
            raise ValueError(f"Row {idx}: parsed sequence length {len(seq)} != number of Task columns ({K}).")
        for pos_idx, taskcol in enumerate(task_pos_cols):
            abs_task = int(seq[pos_idx])  # absolute task id at this position
            val = row[taskcol]
            per_task_values.setdefault(abs_task, []).append(float(val))
    # ensure tasks are ordered by absolute task id
    max_task = max(per_task_values.keys())
    task_ids = sorted(per_task_values.keys())
    # convert to DataFrame for easier plotting
    per_task_df = pd.DataFrame({
        f"Task {tid}": per_task_values[tid] for tid in task_ids
    })
    return per_task_df

def compute_trimmed_variance(series, trim_pct):
    """Compute variance after trimming `trim_pct` percent each tail (trim_pct in [0,50))."""
    if trim_pct <= 0:
        return float(np.nanvar(series, ddof=1))
    p = trim_pct / 100.0
    # compute lower and upper percentiles
    lo = np.percentile(series, p*100)
    hi = np.percentile(series, 100 - p*100)
    trimmed = series[(series >= lo) & (series <= hi)]
    # if too few values, return NaN
    if len(trimmed) < 2:
        return float(np.nan)
    return float(np.var(trimmed, ddof=1))

def make_boxplot(per_task_df, out_path, showfliers=False, dpi=300):
    """Create a compact boxplot (no excessive whitespace)."""
    sns.set_style("whitegrid")
    palette = sns.color_palette(COLOR_PALETTE, n_colors=len(per_task_df.columns))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # boxplot: use notches to convey CI of median
    box = ax.boxplot(
        [per_task_df[c].dropna().values for c in per_task_df.columns],
        labels=per_task_df.columns,
        notch=False,
        patch_artist=True,
        showfliers=showfliers,
        widths=0.6,
        medianprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        )
    # fill boxes with palette
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    # minimal axes decorations
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Task (absolute identity)")
    ax.set_title("Performance variability across task orderings (per absolute task)")
    # reduce whitespace and make layout tight
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def main():
    np.random.seed(RANDOM_SEED)
    per_task_df = load_and_remap(CSV_PATH)
    # print basic summary
    print("Per-task counts and basic stats:")
    print(per_task_df.describe().T[["count", "mean", "std", "min", "max"]])

    # compute trimmed variances
    variances = []
    for col in per_task_df.columns:
        series = per_task_df[col].dropna().values
        trimmed_var = compute_trimmed_variance(series, TRIM_PERCENT)
        raw_var = float(np.var(series, ddof=1))
        variances.append({
            "task": col,
            "n": len(series),
            f"variance_trim{TRIM_PERCENT}%": trimmed_var,
            "variance_raw": raw_var,
            "mean": float(np.mean(series)),
            "median": float(np.median(series))
        })
    var_df = pd.DataFrame(variances).set_index("task")
    var_csv = OUT_DIR / "variance_table.csv"
    var_df.to_csv(var_csv)
    print(f"\nSaved variance table to {var_csv}")
    print(var_df)

    # figure: main boxplot (hide extreme fliers for visual clarity)
    out_pdf = OUT_DIR / "boxplot_tasks.pdf"
    out_png = OUT_DIR / "boxplot_tasks.png"
    make_boxplot(per_task_df, out_pdf, showfliers=SHOW_FLIERS)
    # also save png
    make_boxplot(per_task_df, out_png, showfliers=SHOW_FLIERS)
    print(f"Saved figures to {out_pdf} and {out_png}")

    # optional: also save a version where we first winsorize or trim extremes before plotting
    # produce trimmed DataFrame for plotting (clip to [p, 100-p] percentiles)
    if TRIM_PERCENT > 0:
        trimmed_df = per_task_df.copy()
        p = TRIM_PERCENT
        for col in per_task_df.columns:
            lo = np.percentile(per_task_df[col].dropna(), p)
            hi = np.percentile(per_task_df[col].dropna(), 100 - p)
            # winsorize by clipping
            trimmed_df[col] = np.clip(per_task_df[col], lo, hi)
        out_png_trim = OUT_DIR / "boxplot_tasks_trimmed.png"
        make_boxplot(trimmed_df, out_png_trim, showfliers=False)
        print(f"Saved trimmed figure to {out_png_trim}")

    # Return a suggested figure caption (printed)
    caption = (
        "Figure X: Distribution of final accuracy (in %) for each absolute task across "
        "all permutations (N = {}). Each box summarizes the central 50% of the distribution "
        "(interquartile range); the horizontal line inside the box is the median. "
        "Whiskers extend to the most extreme points inside 1.5×IQR. Extreme outliers were "
        f"{'hidden in the plotted boxes for clarity' if not SHOW_FLIERS else 'shown'}; "
        f"variance values reported in the accompanying table are computed after trimming the "
        f"top/bottom {TRIM_PERCENT:.0f}% of values to reduce the influence of extreme permutations. "
        "This presentation maps each recorded value back to the task's absolute identity (Task 0..), "
        "so that distributions are comparable across permutations of the same underlying task. "
        "Plot styling uses Times New Roman, font size 12, compact layout optimized for publication."
    ).format(int(var_df["n"].iloc[0]))
    print("\nSuggested caption for the figure (copy into your paper):\n")
    print(caption)

if __name__ == "__main__":
    main()
