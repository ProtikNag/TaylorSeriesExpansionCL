# HTCL: Hierarchical Taylor Series-based Continual Learning

Code for the paper *Hierarchical Second-Order Consolidation for Continual Learning* (anonymous submission).

HTCL wraps a base continual learner (finetune, EWC, SR, ER, SER, DER) and processes the task stream in
consecutive **groups of k tasks** (k=2). For each group it trains a local learner from the persistent
hierarchical model $\mathbf w_1$, trying both within-group orders and keeping the better on validation (so the
result is invariant to within-group order), then merges $\mathbf w_1$ toward that local with the closed-form
update

$$\mathbf w_i^{(t)} = \mathbf w_i^{(t-1)} + (\mathbf H_i + \lambda_i \mathbf I)^{-1}\big(\lambda_i\,\Delta\mathbf d_i^{(t)} - \mathbf g_i\big),$$

with a diagonal curvature $\mathbf H_i$ (empirical Fisher by default) and $\lambda_i = 1/\text{lr}$. The merge
is a step-size search: the update is a direction and HTCL keeps the $\alpha \in \{0,.25,.5,.75,1\}$ step (or
the local) with the best accuracy on all seen tasks' validation. **`docs/ALGORITHM.md` is the full spec and
the reason for each part; `docs/DESIGN.md` is the code contract.** The code default is this algorithm — a bare
`htcl: {k: 2}` runs it; `schedule: online` is the one alternative schedule, kept for the R1 ablation.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # training, experiments, tests
.venv/bin/pip install -r requirements-viz.txt      # figures and dashboard only
.venv/bin/python -m pytest -q tests                 # about 5 s on a laptop
```

The device is picked as `--device` > `HTCL_DEVICE` > mps > cuda > cpu.

## Datasets

All data lives under `data/` (git-ignored). Everything except ImageNet downloads on first use.

| key | source | location | notes |
|---|---|---|---|
| `splitmnist` | torchvision MNIST | `data/MNIST` | 5 tasks of 2 digits |
| `cifar100` | torchvision CIFAR-100 | `data/cifar-100-python` | 10 tasks (5/20 in the storyline) |
| `news20` | scikit-learn `fetch_20newsgroups` | scikit-learn cache (`~/scikit_learn_data`) | TF-IDF fit on train only |
| `cora` | PyG `Planetoid` | `data/Planetoid` | full graph, tasks are node masks |
| `imagenet100` | ILSVRC2012, CMC 100-class split | `data/imagenet100` | build with `scripts/prepare_imagenet100.py` |
| `mnist_classil3` | MNIST | `data/MNIST` | E4 only, Class-IL, 3 tasks |

ImageNet-100 needs a local ILSVRC2012 copy:

```bash
python scripts/prepare_imagenet100.py --imagenet-dir /path/to/ILSVRC2012 [--val-labels LOC_val_solution.csv]
```

It writes symlinks to `data/imagenet100/{train,val}/<wnid>/`. The pretrained ViT-S/16 weights
(`vit_small_patch16_224`) come from the timm / Hugging Face cache on first use.

**Local proxy.** Without ImageNet, `python scripts/make_imagenet100_proxy.py` downloads Tiny-ImageNet and
builds `data/imagenet100_proxy` (100 classes, 50 train and 50 val images per class, upsampled to 224). Point
`imagenet_root` at it in a copy of `configs/e9_imagenet.yaml`, with a different `experiment` name. Proxy runs
carry `proxy=True` in their dataset label, so they never pass for ImageNet-100 results. Tiny-ImageNet images
come from ImageNet and may overlap the ViT's pretraining data.

## Running experiments

Each experiment is one YAML file in `configs/`. A job is one (dataset, method, task order). All jobs are
independent, checkpointed and resumable.

```bash
python -m htcl.run plan      configs/e1_main.yaml [overrides]          # write jobs.jsonl, print counts
python -m htcl.run run       configs/e1_main.yaml --all [overrides]    # run every job (failures are reported, the rest continue)
python -m htcl.run run       configs/e1_main.yaml --job 17 [overrides] # one job, e.g. one Slurm array task
python -m htcl.run status    configs/e1_main.yaml [overrides]          # done / partial / pending / alias per group
python -m htcl.aggregate     configs/e1_main.yaml [overrides]          # summary.csv and summary.json
```

**Overrides.** `--samples-per-class N`, `--orders N` and `--epochs N` shrink a run. Each one adds a suffix to the
run directory (`runs/e1_main__spc50_o3/`), so reduced and full runs never mix. Aggregate a run with the same
overrides you ran it with.

**Other options.**
- `--filter key=value` (repeatable, ANDed) selects jobs by a dotted dataset or method field, matching the full
  name or a suffix. Examples: `--filter method=ser`, `--filter dataset=cifar100`, `--filter htcl.levels=3`,
  `--filter htcl=False`.
- `--device cpu|mps|cuda` picks the device.
- `--runs-root DIR` moves the run directories under DIR.
- `--max-steps N` stops each job after N steps (tasks, or groups for HTCL's group schedule) and leaves its checkpoint.

**Resume.** Re-run the same command. Jobs with `DONE` are skipped. A job with `checkpoint.pt` restores its
models, method state (buffers, EWC anchors, every HTCL level, the open revision window and its snapshot), the
partial accuracy matrix and all RNG states, then continues from the next task or group. A killed job therefore loses at most one step. On CPU a resumed
job reproduces the uninterrupted result exactly. On MPS it matches within nondeterminism, and in our checks
it matched exactly.

**Order deduplication.** HTCL's group schedule with `search: best` does not depend on the order inside a
group, so jobs whose canonical order was already planned become aliases and are not run. The online schedule
(default) depends on the arrival order and is never aliased. They count in the statistics through the
canonical run, and `n_unique_runs` is reported next to `n_orders`.

### Paper items

The commands below are for the full protocol. For a reduced local run, add the overrides from the next
section to every command, including `aggregate`.

| Paper item | Config | Command | Output |
|---|---|---|---|
| Fig. 1, order sensitivity (storyline) | `configs/storyline.yaml` | `run configs/storyline.yaml --all` | `runs/storyline/` |
| Main table (E1) | `configs/e1_main.yaml` | `run configs/e1_main.yaml --all` | `runs/e1_main/` |
| SER vs. number of levels (E2) | `configs/e2_ser_levels.yaml` | `run configs/e2_ser_levels.yaml --all` | `runs/e2_ser_levels/` |
| Long sequence, CIFAR-100 DER (E3) | `configs/e1_main.yaml` | reused from E1: `aggregate configs/e1_main.yaml`, rows `cifar100`, `der` | `runs/e1_main/` |
| Loss landscape (E4) | `experiments/loss_landscape.py` | `python -m experiments.loss_landscape [--grid 75] [--lam ...]` | `runs/e4_landscape/landscape.{npz,json}` |
| First-order and DiLoCo table (E5) | `configs/e5_firstorder_diloco.yaml` | `run configs/e5_firstorder_diloco.yaml --all` | `runs/e5_firstorder_diloco/` |
| Window-size break-even (E6) | `configs/e6_group_size.yaml` | `run configs/e6_group_size.yaml --all` | `runs/e6_group_size/` |
| Buffer-size sweep (E7) | `configs/e7_buffer.yaml` | `run configs/e7_buffer.yaml --all` | `runs/e7_buffer/` |
| Federated comparison (E8) | `configs/e8_federated.yaml` | `run configs/e8_federated.yaml --all` | `runs/e8_federated/` |
| ImageNet-100, ViT-S/16 (E9) | `configs/e9_imagenet.yaml` | `run configs/e9_imagenet.yaml --all` | `runs/e9_imagenet/` |
| Ablation: HTCL vs V1 (no merge), V2 (single jump), random order, first-order, online schedule (R1) | `configs/r1_ablation.yaml` | `run configs/r1_ablation.yaml --all` | `runs/r1_ablation/` |
| Compute-matched repeats (R2) | `configs/r2_compute_matched.yaml` | `run configs/r2_compute_matched.yaml --all` | `runs/r2_compute_matched/` |
| Sensitivity to λ and curvature (R3) | `configs/r3_sensitivity.yaml` | `run configs/r3_sensitivity.yaml --all` | `runs/r3_sensitivity/` |

`run` stands for `python -m htcl.run run`. Run `python -m htcl.aggregate <config>` after each experiment.
The storyline config fixes 50 samples per class itself.

### Reduced local runs vs. full runs

The configs encode the full protocol: full data (`samples_per_class: null`), 20 task orders (all t! orders
when t! < 20), and the paper's per-dataset lr, batch size, epochs and buffer size. Smaller runs only add
overrides:

| Run | Overrides | Use |
|---|---|---|
| Smoke | `--samples-per-class 5 --orders 2 --epochs 1 --runs-root runs/_smoke` | Checks that every code path runs. The numbers are meaningless (often chance). |
| Local validation | `--samples-per-class 50` (small datasets, 20 orders); `--samples-per-class 50 --orders 3` (CIFAR-100, ImageNet-100) | All metrics and resume on a laptop GPU (MPS). |
| Full | none | Paper numbers. Run on a cluster with `--job N` per array task. |

Orders are a fixed seeded sequence, so `--orders 3` gives the first 3 orders of the full run. `--samples-per-class`
also caps the test set per class, unless the config sets `test_samples_per_class`.

## Outputs

```
runs/<experiment>[__spcN_oN_epN]/
  jobs.jsonl            # the plan, one resolved job per line (rewritten by plan/run/status)
  summary.csv, .json    # written by htcl.aggregate
  <job_id>/
    job.json            # resolved config: dataset, method with every default filled in, order, seed
    acc_matrix.json     # A[l][task_id] in percent, per variant
    metrics.json        # ACC, forgetting, BWT, final per-task accuracy, learned_pos (R2: per-variant)
    timing.json         # wall-clock per step and in total, device
    final_model.pt      # deployed weights (HTCL: top level; R2: weight average)
    DONE                # completion marker
    checkpoint.pt       # only while the job is unfinished
```

`job_id` is a hash of the resolved job, so a changed hyperparameter, including a method default, gives a new
directory. The summary has one row per (dataset, method, variant) with the mean and sample std (ddof=1) over
orders of ACC, forgetting and BWT, the per-task final accuracy by task id, mean wall-clock, `n_orders`,
`n_unique_runs` and `n_missing`. Metric definitions are in `docs/DESIGN.md`.

## Figures and report

- `figures/FIGURE_STYLE.md` is the style spec for paper figures (palette, type, export contract). Each paper
  figure gets one script under `figures/` that reads `runs/<experiment>/summary.json` or the per-job
  `acc_matrix.json` files, and writes PNG (300 dpi), SVG and PDF.
- `report/acviz.py` holds the Plotly template and export helpers. Figure scripts import it. Every exported
  figure is appended to `report/manifest.json`.
- `report/paperfig.py` holds the matplotlib style for appendix figures, drawn at printed size.
- `python report/build_dashboard.py` turns `report/manifest.json` into `report/index.html`, a single-page
  results dashboard. `python report/verify_dashboard.py` checks the dashboard with a headless browser.
- `python report/make_sample_figures.py` runs the whole figure pipeline on synthetic data, as a smoke test.
- `figures/preview_in_paper.py` places a figure in a local copy of the manuscript to check legibility at
  printed size. The manuscript is not part of this repository.

## Repository layout

```
htcl/          package: data, models, buffer, engine, metrics, permutations, consolidation, hierarchy,
               methods/ (finetune, er, der, ser, ewc, sr, icarl, dualnet, federated), run and aggregate CLIs
configs/       one YAML per experiment
experiments/   standalone experiment scripts (E4 loss landscape)
figures/       figure scripts and style spec
report/        visualization kit and results dashboard
scripts/       dataset preparation, anonymity check
tests/         one runnable check per non-trivial piece of logic
docs/DESIGN.md the design contract
```

Before releasing, run `scripts/check_anonymity.sh`. It scans every tracked or unignored file for e-mail
addresses, absolute home and cluster paths, the local git identity, and private terms listed in the
git-ignored file `.anonymity_terms`.
