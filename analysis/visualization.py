import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ---------- Configuration ----------
er_CSV = "../results/er_permutation_results_20Newsgroups.csv"
TAYLOR_CSV = "../results/taylor_permutation_results_20Newsgroups.csv"
OUT_DIR = "../figures"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 1000
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 20
FIGSIZE_VIOLIN = (10, 6)
FIGSIZE_BOX = (10, 6)

# Colors
VIOLIN_PALETTE = ["#2B547E", "#D87C5D"]    # er (blue), TAYLOR (orange)

# Boxplot styling requested
# Interpreted "134686" as hex color "#134686"
BOX_EDGE_COLOR = "#134686"   # box outline color (user requested)
BOX_FILL_COLOR = "white"
BOX_MEDIAN_COLOR = "#2F5755"  # user-provided green tone for median

# Extreme values (fliers) color: make them black
FLIER_COLOR = "black"

TASK_COLS = ["Task1", "Task2", "Task3", "Task4", "Task5"]

# New labels as requested
TASK_LABELS = ["1st task", "2nd task", "3rd task", "4th task", "5th task"]
task_label_map = {f"Task{i+1}": TASK_LABELS[i] for i in range(len(TASK_COLS))}

# ---------- Helper ----------
def parse_accuracy(x):
    if pd.isna(x):
        return x
    if isinstance(x, str):
        s = x.strip().replace("%", "").replace(",", "")
        try:
            return float(s)
        except:
            return pd.NA
    return float(x)

# ---------- Load data ----------
er_df = pd.read_csv(er_CSV)
taylor_df = pd.read_csv(TAYLOR_CSV)

er_df["Method"] = "er"
taylor_df["Method"] = "TAYLOR"

df_comb = pd.concat([er_df, taylor_df], ignore_index=True)

# Melt to long format for violin
df_long = df_comb.melt(
    id_vars=["sequence", "Method"],
    value_vars=TASK_COLS,
    var_name="Task",
    value_name="Accuracy"
)
df_long["Accuracy"] = df_long["Accuracy"].apply(parse_accuracy).astype(float)
df_long["TaskLabel"] = df_long["Task"].map(task_label_map)

# ---------- Combined Violin Plot (no extrapolation) ----------
sns.set_theme(style="ticks")  # remove grid background
plt.rc("font", family=FONT_FAMILY, size=FONT_SIZE+4)

fig, ax = plt.subplots(figsize=FIGSIZE_VIOLIN, dpi=FIG_DPI)

sns.violinplot(
    data=df_long,
    x="TaskLabel",
    y="Accuracy",
    hue="Method",
    split=True,
    inner="quartile",
    palette=VIOLIN_PALETTE,
    density_norm="width",
    cut=0,              # <--- prevent KDE from extrapolating beyond data min/max
    bw=0.1,           # optional: uncomment to tweak bandwidth (smaller -> less smooth)
    ax=ax
)

# ax.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE+4)
ax.grid(False)

# Ensure xticks/labels set explicitly
tick_positions = list(range(len(TASK_LABELS)))
ax.set_xticks(tick_positions)
ax.set_xticklabels(TASK_LABELS)

# Make quartile lines black and solid (same as you had)
for ln in ax.lines:
    ln.set_color("black")
    ln.set_linestyle("-")
    ln.set_linewidth(1.2)

# Method legend inside bottom-right, transparent
method_leg = ax.legend(title="Method", loc="lower right", frameon=True)
method_leg.get_frame().set_alpha(0.0)

# Quartile legend centered below figure.
quartile_handle = Line2D([0], [0], color="black", linestyle="-", linewidth=1.2)
quartile_leg = ax.legend(
    handles=[quartile_handle],
    # labels=["Quartiles (25th, median, 75th) — solid black"],
    # loc="lower center",
    bbox_to_anchor=(0.5, -0.20),
    frameon=True,
    fontsize=FONT_SIZE - 1
)
quartile_leg.get_frame().set_alpha(0.0)
ax.add_artist(method_leg)

plt.tight_layout(pad=0.6)
out_violin = os.path.join(OUT_DIR, "violin_combined_tasks.svg")
fig.savefig(out_violin, bbox_inches="tight", format="svg", dpi=FIG_DPI)
plt.close(fig)
print(f"Saved violin plot to: {out_violin}")

# ---------- Separate Boxplots (custom colors) ----------
def make_boxplot_for_file(df, method_name, out_name):
    df_m = df.melt(id_vars=["sequence"], value_vars=TASK_COLS,
                   var_name="Task", value_name="Accuracy")
    df_m["Accuracy"] = df_m["Accuracy"].apply(parse_accuracy).astype(float)
    df_m["TaskLabel"] = df_m["Task"].map(task_label_map)

    fig_b, ax_b = plt.subplots(figsize=FIGSIZE_BOX, dpi=FIG_DPI)

    # Boxplot style with requested colors
    boxprops = dict(facecolor=BOX_FILL_COLOR, edgecolor=BOX_EDGE_COLOR, linewidth=1.6)
    whiskerprops = dict(color=BOX_EDGE_COLOR, linewidth=1.2)
    capprops = dict(color=BOX_EDGE_COLOR, linewidth=1.2)
    medianprops = dict(color=BOX_MEDIAN_COLOR, linewidth=1.8)
    # Fliers (extreme values) should be black as requested. Use black markers.
    flierprops = dict(marker='o', markerfacecolor=FLIER_COLOR, markeredgecolor=FLIER_COLOR, markersize=4, alpha=0.9)

    sns.boxplot(
        data=df_m,
        x="TaskLabel",
        y="Accuracy",
        color=BOX_FILL_COLOR,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
        ax=ax_b
    )

    # Ensure ticks are explicitly set before setting labels to avoid UserWarning
    tick_positions = list(range(len(TASK_LABELS)))
    ax_b.set_xticks(tick_positions)
    ax_b.set_xticklabels(TASK_LABELS)

    # Ensure patches have the correct edge color & linewidth (sometimes seaborn redraws)
    for patch in ax_b.artists:
        patch.set_edgecolor(BOX_EDGE_COLOR)
        patch.set_facecolor(BOX_FILL_COLOR)
        patch.set_linewidth(1.6)

    # Also update the lines (whiskers/caps/medians) to ensure colors are applied
    for line in ax_b.lines:
        # Default to BOX_EDGE_COLOR for lines; median lines will be overwritten by medianprops if needed.
        line.set_color(BOX_EDGE_COLOR)
        line.set_linewidth(1.2)

    # Overwrite median lines color with BOX_MEDIAN_COLOR by matching horizontal lines to median values
    medians_by_task = df_m.groupby("TaskLabel")["Accuracy"].median().to_dict()
    for line in ax_b.lines:
        y = line.get_ydata()
        if len(y) == 2 and abs(y[0] - y[1]) < 1e-8:
            yval = float(y[0])
            for med in medians_by_task.values():
                if abs(yval - med) < 1e-6:
                    line.set_color(BOX_MEDIAN_COLOR)
                    line.set_linewidth(1.8)
                    break

    # Ensure fliers (extreme values) remain black by adjusting PathCollections (if any)
    # PathCollections typically represent fliers; set their face/edge colors to black.
    for coll in ax_b.collections:
        try:
            offsets = coll.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue
            # Heuristic: fliers usually produce many small points; we'll set these collections to black.
            coll.set_facecolor(FLIER_COLOR)
            coll.set_edgecolor(FLIER_COLOR)
        except Exception:
            # some collections may not behave the same way across Matplotlib versions; ignore errors
            pass

    ax_b.set_title(f"{method_name} performance across tasks", fontsize=FONT_SIZE + 1)
    ax_b.set_xlabel("Task", fontsize=FONT_SIZE+4)
    ax_b.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE+4)
    ax_b.grid(False)  # remove gridlines

    plt.tight_layout(pad=0.6)
    caption_b = f"{method_name} per-task accuracy distributions across sequences. Box = 25th-75th; line = median (colored). Extreme values are black."
    # plt.figtext(0.5, -0.02, caption_b, wrap=True, ha="center", fontsize=FONT_SIZE)

    out_path = os.path.join(OUT_DIR, out_name)
    fig_b.savefig(out_path, bbox_inches="tight", format="svg", dpi=FIG_DPI)
    plt.close(fig_b)
    print(f"Saved boxplot to: {out_path}")

make_boxplot_for_file(er_df, "er", "boxplot_er_tasks_custom.svg")
make_boxplot_for_file(taylor_df, "TAYLOR", "boxplot_TAYLOR_tasks_custom.svg")
