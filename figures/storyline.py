"""Intro Fig. 1 (storyline): the same tasks in a different order give a different final accuracy.

One column per method: one dot per task order (final accuracy), a line from the worst to the best order,
and the best-to-worst gap Delta in points. All runs share the paper's single training seed and differ
only in task order. Two outputs:
  storyline        the methods alone (single column)
  storyline_htcl   each method alone (family colour) beside its HTCL variant (blue), full width
For methods without a replay buffer (finetune, EWC, SR) HTCL supplies the same N-sample buffer the
replay methods hold.

    python figures/storyline.py [--runs runs/storyline]

Writes report/figures/storyline/storyline{,_htcl}.{png,svg,pdf}, storyline_stats.json, manifest entries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (AC, FONT_UI, W_FULL, W_SINGLE_COL, append_manifest_entry, export_figure,  # noqa: E402
                   hex_to_rgba, install_template)

OUT = REPO / "report" / "figures" / "storyline"
C_NONE, C_REPLAY, C_REG, C_OURS = AC["text_muted"], AC["amber"], AC["red"], AC["blue"]
METHODS = [("finetune", "Finetune", C_NONE),
           ("er", "ER", C_REPLAY), ("der", "DER", C_REPLAY), ("ser", "SER", C_REPLAY),
           ("ewc", "EWC", C_REG), ("sr", "SR", C_REG)]
FAMILIES = [("Replay", 1, 3, C_REPLAY), ("Regularization", 4, 5, C_REG)]
CHANCE = 50.0  # domain scenario, 2 classes per task
PT = 96 / 72   # CSS px per printed point (figures are drawn at their printed width)
S_LAB, S_TICK, S_SMALL = 9 * PT, 8 * PT, 7 * PT


def load(runs: Path) -> dict[str, dict[int, float]]:
    """'er' / 'er+htcl' -> {order_index: final ACC} over the finished runs planned in runs/<exp>/jobs.jsonl."""
    planned = {json.loads(line)["job_id"] for line in (runs / "jobs.jsonl").read_text().splitlines() if line}
    acc: dict[str, dict[int, float]] = {}
    for job_dir in runs.iterdir():
        if job_dir.name not in planned or not (job_dir / "DONE").exists():
            continue
        job = json.loads((job_dir / "job.json").read_text())
        key = job["method"]["name"] + ("+htcl" if job["method"].get("htcl") else "")
        acc.setdefault(key, {})[job["order_index"]] = json.loads((job_dir / "metrics.json").read_text())["acc"]
    return acc


def summary(by_order: dict[int, float]) -> dict:
    """Per-order accuracy, mean, sample std and best-to-worst gap."""
    values = np.array([by_order[o] for o in sorted(by_order)])
    return {"acc_by_order": {int(o): float(by_order[o]) for o in sorted(by_order)}, "n_orders": int(len(values)),
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "gap": float(values.max() - values.min())}


def column(fig: go.Figure, x: float, r: dict, color: str, bold: bool = False) -> None:
    """Dots for the orders, a worst-to-best line, and the Delta label above."""
    values = list(r["acc_by_order"].values())
    fig.add_trace(go.Scatter(x=[x, x], y=[min(values), max(values)], mode="lines", hoverinfo="skip",
                             line=dict(color=hex_to_rgba(color, 0.55), width=1.4)))
    fig.add_trace(go.Scatter(x=[x] * len(values), y=values, mode="markers", hoverinfo="skip",
                             marker=dict(color=hex_to_rgba(color, 0.85), size=6, line=dict(color=AC["bg"], width=0.8))))
    text = f"Δ{r['gap']:.0f}"
    fig.add_annotation(x=x, y=max(values), yshift=9, showarrow=False, text=f"<b>{text}</b>" if bold else text,
                       font=dict(family=FONT_UI, size=S_TICK, color=color))


def figure(results: dict[str, dict], with_htcl: bool) -> go.Figure:
    """Methods alone, or each method beside its HTCL variant."""
    fig = go.Figure()
    keys = [k for k in results if with_htcl or not k.endswith("+htcl")]
    values = [v for k in keys for v in results[k]["acc_by_order"].values()]
    y0, y1 = min(CHANCE - 2, np.floor(min(values) / 5) * 5), np.ceil((max(values) + 4) / 5) * 5
    fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=CHANCE, layer="below",
                  fillcolor=hex_to_rgba(AC["text_faint"], 0.12), line=dict(width=0))
    fig.add_annotation(x=1.5, y=CHANCE, xanchor="center", yanchor="top", showarrow=False,  # empty gap ER|DER
                       text="chance", font=dict(family=FONT_UI, size=S_SMALL, color=AC["text_muted"]))
    half = 0.2 if with_htcl else 0.0
    for x, (key, _, color) in enumerate(METHODS):
        if key in results:
            column(fig, x - half, results[key], color)
        if with_htcl and f"{key}+htcl" in results:
            column(fig, x + half, results[f"{key}+htcl"], C_OURS, bold=True)
    for name, a, b, color in FAMILIES:
        fig.add_annotation(x=(a + b) / 2, y=1.0, yref="paper", yanchor="bottom", yshift=2, showarrow=False,
                           text=f"<b>{name}</b>", font=dict(family=FONT_UI, size=S_LAB, color=color))
        fig.add_shape(type="line", x0=a - 0.35, x1=b + 0.35, y0=1.0, y1=1.0, yref="paper",
                      line=dict(color=hex_to_rgba(color, 0.6), width=1))
    if with_htcl:  # key over the first column: family colour = method alone, blue = with HTCL
        fig.add_annotation(x=0, y=1.0, yref="paper", yanchor="bottom", yshift=2, showarrow=False,
                           text=f"alone | <span style='color:{C_OURS}'><b>+HTCL</b></span>",
                           font=dict(family=FONT_UI, size=S_SMALL, color=AC["text_muted"]))
    fig.update_xaxes(range=[-0.6, len(METHODS) - 0.4], tickvals=list(range(len(METHODS))),
                     ticktext=[name for _, name, _ in METHODS], ticks="", showline=True,
                     tickfont=dict(family=FONT_UI, size=S_TICK, color=AC["text_primary"]))
    fig.update_yaxes(range=[y0, y1], dtick=10, title=dict(text="Final accuracy (%)", font=dict(family=FONT_UI, size=S_LAB)),
                     tickfont=dict(family=FONT_UI, size=S_TICK))
    fig.update_layout(showlegend=False, margin=dict(l=44, r=6, t=22, b=28), plot_bgcolor=AC["bg"])
    return fig


def main() -> None:
    """Summarize the runs, draw both versions, export, and register them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "storyline")
    args = parser.parse_args()

    results = {k: summary(v) for k, v in load(args.runs).items()}
    install_template()
    OUT.mkdir(parents=True, exist_ok=True)
    details = ("SplitMNIST, 5 tasks x 2 digits, domain-IL (chance 50%), MnistResNet, buffer 50, 500 samples/class, "
               "5 random task orders, one paper-wide training seed (report/figures/storyline/storyline_stats.json).")
    versions = [("storyline", False, W_SINGLE_COL, "The same tasks in a different order",
                 "Final accuracy of each method under 5 orders of the same 5 tasks; Δ is the best-to-worst gap in points.")]
    if any(k.endswith("+htcl") for k in results):
        versions.append(("storyline_htcl", True, W_FULL, "Task-order spread without and with HTCL",
                         "Each method alone and with HTCL under the same 5 task orders; Δ is the best-to-worst gap in points."))
    for stem, with_htcl, width, title, caption in versions:
        fig = figure(results, with_htcl)
        variant = export_figure(fig, OUT / stem, width, 250, label="single" if width == W_SINGLE_COL else "full")
        fig.write_image(OUT / f"{stem}.pdf", width=width, height=250)
        append_manifest_entry(REPO / "report" / "manifest.json", id=stem, group="intro", group_title="Introduction",
                              title=title, caption=caption, details=details, variants=[variant],
                              dataset="SplitMNIST", model="MnistResNet")
        print(OUT / f"{stem}.pdf")
    (OUT / "storyline_stats.json").write_text(json.dumps(results, indent=2) + "\n")
    for key, r in results.items():
        print(f"{key:13s} n {r['n_orders']}  mean {r['mean']:5.1f}  sd {r['sd']:4.1f}  gap {r['gap']:5.1f}")


if __name__ == "__main__":
    main()
