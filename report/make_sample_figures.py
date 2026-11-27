#!/usr/bin/env python3
"""Sample Track F run on synthetic data.

Exercises the whole figure pipeline end to end: the academic Plotly template,
direct end-labelling, a key-result callout, uncertainty bands, small multiples,
a Plotly-frames GIF animation, an interactive HTML export, the PNG/SVG export
contract, and the manifest writer.

The data here is synthetic and seeded. It is a pipeline smoke test, not a result.

Usage::

    python report/make_sample_figures.py [--seed 0] [--report-dir report]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acviz import (  # noqa: E402
    AC, DIVERGING, EXPORT_SCALE, FONT_MONO, FONT_UI, SEQ_BLUE, W_FULL,
    add_band, add_callout, add_end_labels, append_manifest_entry,
    export_figure, export_interactive, height_for, hex_to_rgba,
    install_template, series_color, write_gif,
)

# ── Configuration (no magic numbers below this block) ───────────────────────
CONFIG = {
    "seed": 0,
    "n_seeds": 5,
    "n_steps": 120,
    "max_env_steps": 2.0e6,
    "n_tasks": 6,
    "methods": [
        {"name": "Min-max (ours)", "ceiling": 0.87, "rate": 3.4, "noise": 0.026},
        {"name": "CLEAR", "ceiling": 0.72, "rate": 5.0, "noise": 0.030},
        {"name": "Fine-tune", "ceiling": 0.56, "rate": 5.6, "noise": 0.038},
    ],
    "anim_frames": 24,
    "anim_fps": 8,
    "gif_scale": 1.4,
    "dataset": "MinAtar (synthetic)",
    "model": "DQN",
}

FIG_WIDTH = W_FULL
FIG_HEIGHT = height_for(W_FULL)


def set_seed(seed: int) -> np.random.Generator:
    """Seed every source of randomness used here and return the numpy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ── Synthetic data ──────────────────────────────────────────────────────────
def make_return_curves(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate per-seed learning curves for each method.

    Returns:
        ``(steps, {method_name: array of shape (n_seeds, n_steps)})``.
    """
    steps = np.linspace(0.0, CONFIG["max_env_steps"], CONFIG["n_steps"])
    unit = steps / CONFIG["max_env_steps"]
    curves: dict[str, np.ndarray] = {}
    for method in CONFIG["methods"]:
        runs = np.empty((CONFIG["n_seeds"], CONFIG["n_steps"]))
        for run_index in range(CONFIG["n_seeds"]):
            offset = rng.normal(0.0, 0.020)
            # Saturating exponential plus correlated observation noise.
            mean = (method["ceiling"] + offset) * (1.0 - np.exp(-method["rate"] * unit))
            walk = np.cumsum(rng.normal(0.0, method["noise"], CONFIG["n_steps"]))
            walk -= np.linspace(0.0, walk[-1], CONFIG["n_steps"])  # pin both ends
            runs[run_index] = np.clip(mean + 0.35 * walk, 0.0, 1.0)
        curves[method["name"]] = runs
    return steps, curves


def make_forgetting_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a task-by-task forgetting matrix, in return delta units."""
    n = CONFIG["n_tasks"]
    matrix = np.full((n, n), np.nan)
    for trained in range(n):
        for evaluated in range(trained + 1):
            gap = trained - evaluated
            # Forgetting deepens with task distance and saturates; occasional
            # backward transfer shows up as a positive entry.
            base = -0.16 * (1.0 - np.exp(-0.55 * gap))
            matrix[trained, evaluated] = base + rng.normal(0.0, 0.035)
    return matrix


def make_retention(rng: np.random.Generator) -> np.ndarray:
    """Generate per-seed final retention for each task, shape (n_seeds, n_tasks)."""
    n = CONFIG["n_tasks"]
    profile = 0.94 - 0.055 * np.arange(n) + 0.02 * np.sin(np.arange(n))
    return np.clip(
        profile[None, :] + rng.normal(0.0, 0.035, (CONFIG["n_seeds"], n)), 0.0, 1.0
    )


def make_trajectory(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2-D loss surface and a consolidation trajectory across it.

    Returns:
        ``(grid_x, grid_y, surface, path)`` where ``path`` has shape (n_frames, 2).
    """
    grid_x = np.linspace(-2.4, 2.4, 90)
    grid_y = np.linspace(-1.8, 1.8, 90)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    def bowl(cx: float, cy: float, depth: float, spread: float) -> np.ndarray:
        """One Gaussian basin of the mixture loss surface."""
        return -depth * np.exp(-((mesh_x - cx) ** 2 + (mesh_y - cy) ** 2) / spread)

    surface = (bowl(-1.4, 0.6, 1.0, 1.1) + bowl(1.3, -0.5, 1.25, 0.9)
               + bowl(0.1, 1.1, 0.55, 0.7) + 0.06 * (mesh_x ** 2 + mesh_y ** 2))

    # Damped drift from the shallow local basin toward the global one, with
    # gradient noise standing in for stochastic updates.
    n_frames = CONFIG["anim_frames"]
    t = np.linspace(0.0, 1.0, n_frames)
    path_x = -1.4 + 2.7 * (1.0 - np.exp(-3.0 * t)) + rng.normal(0.0, 0.035, n_frames)
    path_y = 0.6 - 1.1 * (1.0 - np.exp(-2.4 * t)) + rng.normal(0.0, 0.035, n_frames)
    return grid_x, grid_y, surface, np.column_stack([path_x, path_y])


# ── Figure 1: return curves (two variants) ──────────────────────────────────
def figure_return_curves(
    steps: np.ndarray, curves: dict[str, np.ndarray], report_dir: Path
) -> None:
    """Learning curves with mean ± std bands, as an overlay and as small multiples."""
    figures_dir = report_dir / "figures"
    names = [m["name"] for m in CONFIG["methods"]]
    means = {name: curves[name].mean(axis=0) for name in names}
    stds = {name: curves[name].std(axis=0) for name in names}

    # ── Variant 1 (the paper figure): overlaid lines with direct end labels ──
    overlay = go.Figure()
    for index, name in enumerate(names):
        color = series_color(index)
        add_band(overlay, steps, means[name] - stds[name], means[name] + stds[name], color)
        overlay.add_trace(go.Scatter(
            x=steps, y=means[name], mode="lines", name=name,
            line=dict(color=color, width=1.8 if index == 0 else 1.2,
                      shape="spline", smoothing=0.4),
            hovertemplate=f"{name}<br>%{{x:.2s}} steps · %{{y:.3f}}<extra></extra>",
        ))

    add_end_labels(overlay, [
        (name, steps[-1], means[name][-1], series_color(index))
        for index, name in enumerate(names)
    ], min_gap=0.055)

    # The callout marks where the primary method overtakes the strongest baseline.
    gap = means[names[0]] - means[names[1]]
    crossings = np.where(np.diff(np.sign(gap)) > 0)[0]
    crossover_steps = float(steps[int(crossings[0]) + 1]) if len(crossings) else float("nan")
    if len(crossings):
        cross = int(crossings[0]) + 1
        add_callout(
            overlay, steps[cross], means[names[0]][cross],
            f"crossover at {crossover_steps / 1e6:.2f}M steps", ax=-136, ay=52,
        )

    overlay.update_layout(
        title=dict(text="Mean return across 5 seeds"),
        xaxis=dict(title=dict(text="Environment steps (millions)"),
                   range=[0, CONFIG["max_env_steps"]],
                   tickvals=np.linspace(0, CONFIG["max_env_steps"], 5),
                   ticktext=[f"{v:.1f}" for v in np.linspace(0, 2.0, 5)]),
        yaxis=dict(title=dict(text="Normalized return"),
                   range=[0, 1.0], tickformat=".0%", nticks=6),
    )
    variant_line = export_figure(
        overlay, figures_dir / "return_curves_seed_sweep_line",
        FIG_WIDTH, FIG_HEIGHT, label="Overlay",
    )

    # ── Variant 2: small multiples, one panel per method ─────────────────────
    grid = make_subplots(
        rows=1, cols=len(names), shared_yaxes=True, shared_xaxes=True,
        horizontal_spacing=0.035, subplot_titles=names,
    )
    for index, name in enumerate(names):
        color = series_color(index)
        add_band(grid, steps, means[name] - stds[name], means[name] + stds[name],
                 color, row=1, col=index + 1)
        grid.add_trace(go.Scatter(
            x=steps, y=means[name], mode="lines",
            line=dict(color=color, width=1.8), showlegend=False,
            hovertemplate=f"{name}<br>%{{y:.3f}}<extra></extra>",
        ), row=1, col=index + 1)
        # Per-seed traces, faint, to show the spread the band summarises.
        for run in curves[name]:
            grid.add_trace(go.Scatter(
                x=steps, y=run, mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color=hex_to_rgba(color, 0.22), width=0.6),
            ), row=1, col=index + 1)

    for annotation in grid.layout.annotations:
        annotation.font = dict(family=FONT_UI, size=12, color=AC["text_primary"])
    grid.update_xaxes(title=dict(text="Env steps (M)"), tickvals=[0, 1e6, 2e6],
                      ticktext=["0", "1", "2"], range=[0, CONFIG["max_env_steps"]])
    grid.update_yaxes(range=[0, 1.0], tickformat=".0%", nticks=5)
    grid.update_yaxes(title=dict(text="Normalized return"), row=1, col=1)
    grid.update_layout(title=dict(text="Per-method spread across seeds"),
                       margin=dict(l=64, r=24, t=64, b=56))

    variant_grid = export_figure(
        grid, figures_dir / "return_curves_seed_sweep_grid",
        FIG_WIDTH, height_for(FIG_WIDTH, 2.05), label="Small multiples",
    )

    append_manifest_entry(
        report_dir / "manifest.json",
        id="return_curves_seed_sweep",
        group="training_dynamics",
        group_title="Training dynamics",
        title="Normalized return across seeds",
        caption=(
            f"Mean return with ±1 std across {CONFIG['n_seeds']} seeds; the min-max "
            f"objective overtakes CLEAR at {crossover_steps / 1e6:.2f}M environment steps."
            if np.isfinite(crossover_steps) else
            f"Mean return with ±1 std across {CONFIG['n_seeds']} seeds for three "
            "continual-RL objectives."
        ),
        details=(
            "**Synthetic pipeline check.** Learning curves for three continual-RL "
            "objectives on a stand-in MinAtar task suite.\n\n"
            "- **Method.** Alternating local/global primal-dual policy gradient "
            "(min-max), compared against CLEAR replay and naive fine-tuning.\n"
            "- **Seeds.** 5 per method, shown as mean ± 1 std. Faint per-seed "
            "traces appear in the small-multiples variant.\n"
            "- **What to look for.** The crossover point, and whether the min-max "
            "band separates from the CLEAR band rather than merely shifting.\n\n"
            "Data is generated by `report/make_sample_figures.py` and carries no "
            "experimental meaning."
        ),
        seed=CONFIG["seed"],
        dataset=CONFIG["dataset"],
        model=CONFIG["model"],
        variants=[variant_line, variant_grid],
        metrics=[
            {"name": "final return (ours)", "value": float(means[names[0]][-1]), "unit": ""},
            {"name": "final return (CLEAR)", "value": float(means[names[1]][-1]), "unit": ""},
            {"name": "seeds", "value": CONFIG["n_seeds"], "unit": ""},
        ],
    )


# ── Figure 2: consolidation trajectory (animated) ───────────────────────────
def figure_trajectory(
    grid_x: np.ndarray, grid_y: np.ndarray, surface: np.ndarray,
    path: np.ndarray, report_dir: Path,
) -> None:
    """Parameter trajectory across a loss surface, animated over consolidation steps."""
    figures_dir = report_dir / "figures"
    n_frames = len(path)

    def contour() -> go.Contour:
        """The static loss surface, drawn as line contours."""
        return go.Contour(
            x=grid_x, y=grid_y, z=surface, colorscale=SEQ_BLUE, showscale=False,
            contours=dict(coloring="lines", start=float(surface.min()),
                          end=float(surface.max()), size=0.09),
            line=dict(width=0.8), hoverinfo="skip",
        )

    def trail(upto: int) -> list[go.Scatter]:
        """Trajectory traces up to and including step ``upto``."""
        return [
            go.Scatter(x=path[:upto + 1, 0], y=path[:upto + 1, 1], mode="lines",
                       line=dict(color=AC["blue"], width=2.0, shape="spline"),
                       hoverinfo="skip"),
            go.Scatter(x=[path[upto, 0]], y=[path[upto, 1]], mode="markers",
                       marker=dict(color=AC["red"], size=8,
                                   line=dict(color=AC["bg"], width=1.4)),
                       hovertemplate="step %{text}<extra></extra>",
                       text=[str(upto)]),
        ]

    axis_style = dict(showgrid=False, zeroline=False, nticks=6)
    base_layout = dict(
        title=dict(text="Parameter trajectory under consolidation"),
        xaxis=dict(title=dict(text="Principal direction 1 (a.u.)"),
                   range=[grid_x[0], grid_x[-1]], **axis_style),
        yaxis=dict(title=dict(text="Principal direction 2 (a.u.)"),
                   range=[grid_y[0], grid_y[-1]], scaleanchor="x", scaleratio=1,
                   **axis_style),
        margin=dict(l=64, r=32, t=52, b=56),
    )

    # The GIF carries the motion: Kaleido rasterises each frame independently,
    # so the surface is drawn correctly in every one.
    full_frames = [
        go.Frame(data=[contour(), *trail(i)], name=str(i)) for i in range(n_frames)
    ]
    gif_base = go.Figure(layout=go.Layout(**base_layout))
    gif_path = write_gif(
        gif_base, full_frames, figures_dir / "consolidation_trajectory",
        FIG_WIDTH, height_for(FIG_WIDTH, 1.25),
        fps=CONFIG["anim_fps"], scale=CONFIG["gif_scale"],
    )

    # The interactive export is deliberately NOT a frame animation. plotly.js
    # clears non-animatable trace types (contour, heatmap, ...) for the duration
    # of a transition, so an animated contour renders as a blank panel while it
    # plays. Interactivity here is hover inspection of the whole path instead.
    interactive = go.Figure(data=[
        contour(),
        go.Scatter(
            x=path[:, 0], y=path[:, 1], mode="lines+markers",
            line=dict(color=AC["blue"], width=2.0, shape="spline"),
            marker=dict(color=AC["blue"], size=5,
                        line=dict(color=AC["bg"], width=1.0)),
            text=[str(i) for i in range(n_frames)],
            hovertemplate="step %{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
        ),
        go.Scatter(
            x=[path[-1, 0]], y=[path[-1, 1]], mode="markers",
            marker=dict(color=AC["red"], size=9,
                        line=dict(color=AC["bg"], width=1.4)),
            hovertemplate=f"final, step {n_frames - 1}<extra></extra>",
        ),
    ])
    interactive.update_layout(**base_layout, showlegend=False)
    interactive_path = export_interactive(
        interactive, figures_dir / "consolidation_trajectory"
    )

    # Static variant: a small multiple of four selected frames, so the paper
    # figure shows the progression rather than a blank first frame.
    picks = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
    grid = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.045, vertical_spacing=0.09,
        subplot_titles=[f"step {p}" for p in picks],
    )
    for position, upto in enumerate(picks):
        row, column = divmod(position, 2)
        grid.add_trace(contour(), row=row + 1, col=column + 1)
        for trace in trail(upto):
            trace.showlegend = False
            grid.add_trace(trace, row=row + 1, col=column + 1)
    for annotation in grid.layout.annotations:
        annotation.font = dict(family=FONT_MONO, size=11, color=AC["text_muted"])
    grid.update_xaxes(range=[grid_x[0], grid_x[-1]], showgrid=False,
                      zeroline=False, nticks=4)
    grid.update_yaxes(range=[grid_y[0], grid_y[-1]], showgrid=False,
                      zeroline=False, nticks=4)
    grid.update_xaxes(title=dict(text="Principal direction 1 (a.u.)"), row=2)
    grid.update_yaxes(title=dict(text="Direction 2"), col=1)
    grid.update_layout(title=dict(text="Consolidation trajectory, selected steps"),
                       showlegend=False, margin=dict(l=68, r=24, t=64, b=56))

    variant = export_figure(
        grid, figures_dir / "consolidation_trajectory",
        FIG_WIDTH, height_for(FIG_WIDTH, 1.28), label="Selected frames",
    )
    variant["html"] = interactive_path

    append_manifest_entry(
        report_dir / "manifest.json",
        id="consolidation_trajectory",
        group="training_dynamics",
        group_title="Training dynamics",
        title="Parameter trajectory under consolidation",
        caption="Iterates drift out of the shallow local basin into the global "
                "one over 24 consolidation steps.",
        details=(
            "**Synthetic pipeline check.** A mixture-of-Gaussians loss surface "
            "with the parameter iterate drawn over it.\n\n"
            "- **Why animate.** The *order* in which the iterate leaves the "
            "shallow basin is the point; a single static path hides the pacing "
            "and the early dwell time.\n"
            "- **Static form.** Four selected frames as a small multiple, never "
            "the blank first frame.\n"
            "- **What to look for.** How long the iterate lingers near the local "
            "basin before the drift term dominates.\n\n"
            "Surface and path are generated by `report/make_sample_figures.py`."
        ),
        seed=CONFIG["seed"],
        dataset=CONFIG["dataset"],
        model=CONFIG["model"],
        variants=[variant],
        animation={"gif": gif_path, "html": interactive_path},
        metrics=[
            {"name": "frames", "value": n_frames, "unit": ""},
            {"name": "path length", "value": float(
                np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), "unit": "a.u."},
        ],
    )


# ── Figure 3: forgetting matrix ─────────────────────────────────────────────
def figure_forgetting_matrix(matrix: np.ndarray, report_dir: Path) -> None:
    """Task-by-task forgetting, drawn as a diverging heatmap."""
    figures_dir = report_dir / "figures"
    n = CONFIG["n_tasks"]
    labels = [f"T{i + 1}" for i in range(n)]
    limit = float(np.nanmax(np.abs(matrix)))

    heat = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale=DIVERGING,
        zmid=0.0, zmin=-limit, zmax=limit,
        xgap=2, ygap=2, hoverongaps=False,
        hovertemplate="after %{y}, eval %{x}<br>Δ return %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="Δ return", font=dict(family=FONT_UI, size=12)),
                      tickfont=dict(family=FONT_MONO, size=10,
                                    color=AC["text_muted"]),
                      thickness=10, len=0.78, outlinewidth=0),
    ))

    # Annotate every filled cell; a 6x6 matrix is small enough to read directly.
    for row in range(n):
        for col in range(row + 1):
            value = matrix[row, col]
            heat.add_annotation(
                x=labels[col], y=labels[row], text=f"{value:+.2f}", showarrow=False,
                font=dict(family=FONT_MONO, size=10,
                          color=AC["bg"] if abs(value) > 0.62 * limit
                          else AC["text_primary"]),
            )

    worst_flat = int(np.nanargmin(matrix))
    worst_row, worst_col = divmod(worst_flat, n)
    add_callout(
        heat, labels[worst_col], labels[worst_row],
        f"worst: {matrix[worst_row, worst_col]:+.2f} on {labels[worst_col]}",
        ax=40, ay=-38,
    )

    heat.update_layout(
        title=dict(text="Forgetting matrix"),
        xaxis=dict(title=dict(text="Evaluated task"), showgrid=False, side="bottom"),
        yaxis=dict(title=dict(text="After training task"), showgrid=False,
                   autorange="reversed"),
        margin=dict(l=76, r=32, t=52, b=56),
    )

    side = height_for(W_FULL, 1.32)
    variant = export_figure(
        heat, figures_dir / "forgetting_matrix", W_FULL, side, label="Heatmap",
    )

    append_manifest_entry(
        report_dir / "manifest.json",
        id="forgetting_matrix",
        group="retention",
        group_title="Retention and forgetting",
        title="Forgetting matrix across the task sequence",
        caption="Return delta on each earlier task after training every later "
                "task; blue is forgetting, red is backward transfer.",
        details=(
            "**Synthetic pipeline check.** Entry (i, j) is the change in return "
            "on task j measured after training on task i.\n\n"
            "- **Scale.** Diverging blue → light → red, centred at zero, so the "
            "sign of transfer is readable without a legend lookup. No rainbow.\n"
            "- **What to look for.** Whether forgetting deepens monotonically "
            "with task distance or saturates, and any positive (red) entries "
            "indicating backward transfer.\n\n"
            "Generated by `report/make_sample_figures.py`."
        ),
        seed=CONFIG["seed"],
        dataset=CONFIG["dataset"],
        model=CONFIG["model"],
        variants=[variant],
        metrics=[
            {"name": "mean forgetting", "value": float(np.nanmean(matrix)), "unit": "Δ return"},
            {"name": "worst cell", "value": float(np.nanmin(matrix)), "unit": "Δ return"},
        ],
    )


# ── Figure 4: retention per task (two variants) ─────────────────────────────
def figure_retention(retention: np.ndarray, report_dir: Path) -> None:
    """Final retention per task, as bars with error bars and as a distribution."""
    figures_dir = report_dir / "figures"
    labels = [f"T{i + 1}" for i in range(CONFIG["n_tasks"])]
    means = retention.mean(axis=0)
    stds = retention.std(axis=0)

    # ── Variant 1 (the paper figure): bars with ±1 std error bars ────────────
    bars = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=stds, color=AC["axis"], thickness=1.0, width=4),
        marker=dict(color=hex_to_rgba(AC["blue"], 0.85),
                    line=dict(color=AC["blue"], width=1.0)),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    worst = int(np.argmin(means))
    add_callout(bars, labels[worst], means[worst] - stds[worst],
                f"lowest retention {means[worst]:.2f}", ax=30, ay=40)
    bars.update_layout(
        title=dict(text="Final retention by task"),
        xaxis=dict(title=dict(text="Task in the sequence")),
        yaxis=dict(title=dict(text="Retained return (fraction of peak)"),
                   range=[0, 1.05], tickformat=".0%", nticks=6),
        margin=dict(l=72, r=32, t=52, b=56),
    )
    variant_bars = export_figure(
        bars, figures_dir / "retention_by_task_bars",
        W_FULL, height_for(W_FULL, 1.9), label="Bars",
    )

    # ── Variant 2: the per-seed distribution behind those means ─────────────
    box = go.Figure()
    for index, label in enumerate(labels):
        box.add_trace(go.Box(
            y=retention[:, index], name=label, boxpoints="all", jitter=0.5,
            pointpos=0.0, width=0.55,
            marker=dict(color=AC["blue"], size=5, opacity=0.75),
            line=dict(color=AC["blue"], width=1.2),
            fillcolor=hex_to_rgba(AC["blue"], 0.14),
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))
    box.update_layout(
        title=dict(text="Retention spread across seeds"),
        xaxis=dict(title=dict(text="Task in the sequence")),
        yaxis=dict(title=dict(text="Retained return (fraction of peak)"),
                   range=[0, 1.05], tickformat=".0%", nticks=6),
        showlegend=False, margin=dict(l=72, r=32, t=52, b=56),
    )
    variant_box = export_figure(
        box, figures_dir / "retention_by_task_dist",
        W_FULL, height_for(W_FULL, 1.9), label="Distribution",
    )

    append_manifest_entry(
        report_dir / "manifest.json",
        id="retention_by_task",
        group="retention",
        group_title="Retention and forgetting",
        title="Final retention by task",
        caption="Retained return per task at the end of the sequence, mean ±1 std "
                "over 5 seeds, with the per-seed spread alongside.",
        details=(
            "**Synthetic pipeline check.** Retention is the fraction of a task's "
            "peak return still held once the whole sequence has been trained.\n\n"
            "- **Variants.** Bars carry the trend and the uncertainty; the box "
            "variant shows the distribution the bars summarise. Two views of the "
            "same numbers, not two renderings of one view.\n"
            "- **What to look for.** Whether retention decays smoothly with task "
            "position or drops at a particular boundary, and whether the spread "
            "widens for later tasks.\n\n"
            "Generated by `report/make_sample_figures.py`."
        ),
        seed=CONFIG["seed"],
        dataset=CONFIG["dataset"],
        model=CONFIG["model"],
        variants=[variant_bars, variant_box],
        metrics=[
            {"name": "mean retention", "value": float(means.mean()), "unit": ""},
            {"name": "worst task", "value": labels[worst], "unit": ""},
            {"name": "seeds", "value": CONFIG["n_seeds"], "unit": ""},
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """Generate every sample figure and update the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args(argv)

    CONFIG["seed"] = args.seed
    report_dir = Path(args.report_dir)
    rng = set_seed(args.seed)
    install_template()

    print(f"sample figures | seed={args.seed} dataset={CONFIG['dataset']} "
          f"model={CONFIG['model']} scale={EXPORT_SCALE:g}")

    steps, curves = make_return_curves(rng)
    figure_return_curves(steps, curves, report_dir)
    print("  wrote return_curves_seed_sweep (2 variants)")

    grid_x, grid_y, surface, path = make_trajectory(rng)
    figure_trajectory(grid_x, grid_y, surface, path, report_dir)
    print("  wrote consolidation_trajectory (animated)")

    figure_forgetting_matrix(make_forgetting_matrix(rng), report_dir)
    print("  wrote forgetting_matrix")

    figure_retention(make_retention(rng), report_dir)
    print("  wrote retention_by_task (2 variants)")

    print(f"manifest: {report_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
