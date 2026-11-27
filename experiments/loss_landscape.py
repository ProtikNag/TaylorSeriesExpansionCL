"""E4: loss landscape of HTCL on class-incremental MNIST (PCA plane of the weight trajectory).

    python -m experiments.loss_landscape [--samples-per-class N] [--grid 75] [--epochs E] [--runs-root runs]

Protocol (paper experiments.tex, F-landscape; docs/DESIGN.md dataset `mnist_classil3`):
- tasks A={0,1,2}, B={3,4,5}, C={6..9} learned in the order A, B, C, Class-IL (global labels,
  10-way head, no task id), SmallMLP 784-100-100-10, SGD lr 0.01, batch 64, 10 epochs, buffer 50;
- HTCL-L2 with k = 1, so every task is its own group. The paper states k = 3, which with three
  tasks gives a single group and no consolidation; k = 1 gives the three steps the figure shows.
  Step 1 initialises the hierarchical model (h_1 = l_1); steps 2 and 3 are Taylor updates
  h_i = h_{i-1} + (H + lambda I)^{-1}(lambda (l_i - h_{i-1}) - g), with l_i trained from h_{i-1};
- after step i we record the local model l_i and the hierarchical model h_i (flattened parameters);
- PCA plane: with mu the mean of the six vectors, u_1, u_2 are the top two right singular vectors
  of [theta_j - mu]_j; point j has coordinates c_j = (u_1, u_2)^T (theta_j - mu) (Li et al. 2018,
  trajectory variant; no filter normalisation since the axes are PCA directions);
- on a G x G grid over the coordinate box (plus a margin) we evaluate theta(a, b) = mu + a u_1 + b u_2
  and store the mean CE per task on the chosen split. Panel i uses the cumulative set A..task_i:
  L_{<=i} = sum_{j<=i} n_j CE_j / sum_{j<=i} n_j, stored as log(1 + L_{<=i}).

Outputs (no plotting): <runs_root>/e4_landscape[__spcN_epE]/landscape.npz and landscape.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from htcl.data import make_loader
from htcl.engine import evaluate_method, load_data, make_method
from htcl.hierarchy import HTCL, resolve
from htcl.run import resolve_dataset, run_dir
from htcl.utils import derive_seed, get_device, seed_everything

EXPERIMENT = "e4_landscape"
# docs/DESIGN.md dataset table, row mnist_classil3 (paper hyperparameter table for MNIST).
DATASET = {"name": "mnist_classil3", "scenario": "class", "lr": 0.01, "batch_size": 64, "epochs": 10,
           "buffer_size": 50}
ORDER = [0, 1, 2]   # A, B, C
MARGIN = 0.25       # grid box = coordinate range padded by this fraction on each side


def task_tensors(dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All (x, y) of a task split stacked on the device (MNIST subsets fit in memory)."""
    xs, ys = zip(*make_loader(dataset, 4096, shuffle=False))
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


@torch.no_grad()
def per_task_ce(model: torch.nn.Module, theta: torch.Tensor, tasks: list[tuple[torch.Tensor, torch.Tensor]]) -> list[float]:
    """Mean cross-entropy of `model` with flat parameters `theta` on each task's samples."""
    vector_to_parameters(theta, model.parameters())
    return [float(F.cross_entropy(model(x), y)) for x, y in tasks]


def pca_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean, top-2 principal directions (2, D), coordinates (n, 2) and explained-variance ratios."""
    mu = points.mean(0)
    _, s, vt = np.linalg.svd(points - mu, full_matrices=False)
    directions = vt[:2]
    return mu, directions, (points - mu) @ directions.T, (s ** 2 / (s ** 2).sum())[:2]


def main(argv: list[str] | None = None) -> None:
    """Train HTCL on A, B, C, record l_i and h_i, evaluate the loss on their PCA plane."""
    parser = argparse.ArgumentParser(prog="experiments.loss_landscape")
    parser.add_argument("--samples-per-class", type=int, help="cap training (and test) samples per class")
    parser.add_argument("--grid", type=int, default=75, help="grid points per axis (paper: 75)")
    parser.add_argument("--epochs", type=int, help="override the 10 epochs per task (smoke runs)")
    parser.add_argument("--method", default="er", help="base method; ER as the reference replay method")
    parser.add_argument("--lam", type=float, help="HTCL lambda (default: 1/lr of the dataset)")
    parser.add_argument("--split", choices=("train", "test"), default="train", help="data the loss is measured on")
    parser.add_argument("--device")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args(argv)

    overrides = {"samples_per_class": args.samples_per_class, "epochs": args.epochs, "runs_root": args.runs_root}
    dataset = resolve_dataset(DATASET, overrides)
    htcl_spec = resolve({"levels": 2, "k": 1, "lam": args.lam}, dataset["lr"])
    job = {"dataset": dataset, "method": {"name": args.method, "htcl": htcl_spec}}
    seed = derive_seed(EXPERIMENT, args.method, htcl_spec)
    device = get_device(args.device)
    out = run_dir({"experiment": EXPERIMENT}, overrides)
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    data = load_data(dataset)
    htcl = HTCL(make_method(job, data, device), data, dataset, htcl_spec, device, seed)
    snapshots, names, acc = [], [], {}
    t0 = time.perf_counter()
    for step, task_id in enumerate(ORDER, start=1):
        htcl.learn([task_id])
        local, hier = htcl.method.model, htcl.levels[-1]
        for name, model in ((f"l{step}", local), (f"h{step}", hier)):
            snapshots.append(parameters_to_vector(model.parameters()).detach().cpu().double().numpy())
            names.append(name)
            acc[name] = evaluate_method(htcl.method, data, "test", device, weights=model.state_dict())
    train_seconds = time.perf_counter() - t0

    mu, directions, coords, explained = pca_plane(np.stack(snapshots))
    # Distance of each snapshot from the plane: the plane is exact only if this is ~0.
    residual = np.linalg.norm(np.stack(snapshots) - mu - coords @ directions, axis=1)
    lo, hi = coords.min(0), coords.max(0)
    pad = MARGIN * (hi - lo)
    alphas = np.linspace(lo[0] - pad[0], hi[0] + pad[0], args.grid)
    betas = np.linspace(lo[1] - pad[1], hi[1] + pad[1], args.grid)

    probe = htcl.method.model  # its weights are overwritten point by point; training is finished
    probe.eval()
    tasks = [task_tensors(getattr(data.tasks[t], args.split), device) for t in ORDER]
    counts = np.array([len(y) for _, y in tasks], dtype=np.float64)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    dir_t = torch.tensor(directions, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    loss_per_task = np.zeros((len(ORDER), args.grid, args.grid))  # [task, beta index, alpha index]
    for j, b in enumerate(betas):
        for i, a in enumerate(alphas):
            loss_per_task[:, j, i] = per_task_ce(probe, mu_t + float(a) * dir_t[0] + float(b) * dir_t[1], tasks)
    point_loss = np.array([per_task_ce(probe, torch.tensor(s, dtype=torch.float32, device=device), tasks)
                           for s in snapshots])  # true (off-plane) loss at each snapshot, [snapshot, task]
    grid_seconds = time.perf_counter() - t0

    # Panel i: sample-weighted mean CE over tasks A..i.
    weights = np.tril(np.ones((len(ORDER), len(ORDER)))) * counts
    weights /= weights.sum(1, keepdims=True)
    cumulative = np.einsum("pt,tji->pji", weights, loss_per_task)
    np.savez_compressed(out / "landscape.npz", alphas=alphas, betas=betas, loss_per_task=loss_per_task,
                        log1p_cumulative=np.log1p(cumulative), coords=coords, names=np.array(names),
                        point_loss=point_loss, log1p_point_cumulative=np.log1p(point_loss @ weights.T),
                        residual=residual, explained_variance=explained, params=np.stack(snapshots).astype(np.float32))
    meta = {"experiment": EXPERIMENT, "dataset": dataset, "method": job["method"], "order": ORDER,
            "task_classes": [data.tasks[t].classes for t in ORDER], "seed": seed, "device": str(device),
            "split": args.split, "split_sizes": counts.astype(int).tolist(), "grid": args.grid, "margin": MARGIN,
            "names": names, "coords": dict(zip(names, coords.tolist())), "residual": dict(zip(names, residual.tolist())),
            "explained_variance": explained.tolist(), "test_acc": acc,
            "train_seconds": train_seconds, "grid_seconds": grid_seconds,
            "arrays": {"loss_per_task": "[task, beta, alpha] mean CE", "log1p_cumulative": "[panel, beta, alpha]",
                       "point_loss": "[snapshot, task] mean CE at the true snapshot weights"}}
    (out / "landscape.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: grid {args.grid}x{args.grid}, explained {explained.round(3).tolist()}, "
          f"train {train_seconds:.1f}s, grid {grid_seconds:.1f}s")
    for name in names:
        print(f"  {name}: coords {np.round(coords[names.index(name)], 3).tolist()} test acc "
              f"{np.round(acc[name], 1).tolist()}")


if __name__ == "__main__":
    main()
