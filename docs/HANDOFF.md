# Handoff (tnnls-release)

State of the TNNLS revision on 2026-11-26, for any session that picks the work up (laptop or
cluster). Read `CLAUDE.md` (project rules), this file, `README.md`, `docs/DESIGN.md` (the binding contract), and
`reviews/PLAN.md` (storyline and decisions), `reviews/CONCERNS.md` and the manuscript in `paper/` before acting.

## Where things stand

- **The code was rebuilt from scratch on this branch.** An audit of the old code found that ER, SER and
  DER were not continual learning (the epoch loop ran outside the task loop), HTCL always ran ER locally,
  forgetting was never computed, and model selection used test data. None of the old numbers are reused.
- **HTCL is the group schedule, k=2** (`schedule: group`) — the algorithm the paper describes. All paper
  configs set it explicitly (the code default `schedule: online` is kept only for the ablation):
  - The stream is split into consecutive groups of k tasks. For each group a local learner trains from
    the persistent model w_1, trying both orders of the pair and keeping the better one on *validation*
    data (bit-identical for within-group-reshuffled orders — the within-window invariance, C15).
    finetune/EWC/SR get HTCL's N-sample reservoir replayed during local training.
  - The global is merged toward that local by the closed-form Taylor step, kept under either a safety
    check (`guard`) or a step-size search (`merge_search`: pick α ∈ {0,.25,.5,.75,1} or the local on
    seen-task validation; `merge_last` merges a final short group too). The online schedule
    (`schedule: online`, consolidate every task, revise every k) is a documented ablation.
  - Catch-up is not part of the algorithm.
- **Tests and smoke:** `pytest -q tests/` passes, and every config passes a CPU/MPS smoke run.
- **Storyline (intro Fig. 1):**
  - Configuration: SplitMNIST, 5 tasks of 2 digits, 500 samples per class, 5 task orders, six methods,
    each alone and with HTCL (60 jobs, `configs/storyline.yaml`).
  - Laptop result for the methods alone: final accuracy moves by 5 to 23 points between orders.
  - The laptop freezes under these runs, so **all runs move to the cluster**, the storyline's 30 baseline
    jobs included. Baseline and HTCL columns must come from the same device.
  - Figure: `figures/storyline.py --runs <dir>` writes `report/figures/storyline/storyline.*` (methods
    alone) and `storyline_htcl.*` (each method beside its HTCL variant). It is drawn on the laptop from the
    numbers the cluster pushes.

## Decisions that bind every run

| Topic | Decision |
|---|---|
| Seed | One training seed for the whole paper (`PAPER_SEED = 0`, `htcl/run.py`), the same for every method and task order. Never add seed sweeps; the spread across task orders is the statistic. |
| Orders | 20 task orders per experiment in the full protocol (all t! when t! < 20). HTCL group-schedule canonical duplicates are aliased, never rerun. Online HTCL jobs are never aliased. |
| Memory | The deployed models (HTCL's top model and every baseline) hold the same replay buffer N. The k-task window HTCL's local learner trains on is training-time memory and is reported (`htcl_window_tasks`). For methods without a buffer (finetune, EWC, SR) HTCL supplies the N-sample buffer. |
| λ | Default 1/lr of the dataset, resolved at planning time (it enters the job id). λ = 1 collapses to chance. |
| Selection | A 10% validation split of each task's training data. Test data is only for reporting. |
| Metrics | Accuracy matrix per run; ACC, standard forgetting, BWT; sample std (ddof=1) across orders; per-task std by task id. |
| Protocol | Domain-IL for every dataset (Class-IL is a flag, used by E4 only). |

## Open issues (decide before the full-data runs)

1. **Cora never trains at the paper's 10 full-batch epochs.** Seven of eight methods give the same
   constant-predictor ACC. At 100 epochs, finetune reaches 73.7, ER 75.1 and SER 88.3
   (`runs/_smoke/cora_epochs/`). The Cora epochs need raising.
2. **CIFAR-100 at reduced data is near chance.** At 50 or 150 samples per class, ER reaches 18–19%
   against 10% chance. Local CIFAR numbers only validate code paths; real CIFAR numbers need full data
   on a GPU.
3. **EWC λ is per dataset.** CIFAR-100 uses 10, because 400 and 50 diverge at lr 0.1; the others use 400.
   `htcl/methods/ewc.py` documents this.
4. **DER sits below ER on SplitMNIST** (73 vs 80 at 500 samples per class, buffer 50). Check DER's α
   before the main table.
5. **SER on CIFAR-100 at lr 0.1 diverges deterministically at 5 samples per class.** The engine's
   non-finite guard fails such jobs loudly. Check whether this also happens at 50 samples per class
   and at full data.
6. **Real ImageNet-100 is not prepared.** Only the Tiny-ImageNet proxy has been used
   (`scripts/make_imagenet100_proxy.py`).
7. **Missing pieces:**
   - Slurm and cloud launchers (`scripts/`)
   - per-figure scripts other than the storyline
   - the dashboard (`report/build_dashboard.py` on `report/manifest.json`)

## Division of work

| Where | What |
|---|---|
| Laptop | Figures and the dashboard, drawn from the numbers the cluster pushes; paper text; review. No training runs (the laptop freezes). |
| Cluster (Slurm, CUDA) | **All training runs.** First the storyline (`configs/storyline.yaml`, all 60 jobs), then the P4 validation run (every main experiment at 50 samples per class). **No figures or other visualizations on the cluster**: the terminal there cannot show them. Push numbers only (below). |
| Rented GPU (later) | ImageNet-100 (E9) at full data. |

### 1. Storyline on the cluster (first)

`python -m htcl.run run configs/storyline.yaml --all` (no overrides; split it into Slurm array slices by
`--filter method=X --filter htcl=True|False` if useful), then `python -m htcl.aggregate configs/storyline.yaml`.
Push the numbers as soon as these 60 jobs finish, before starting P4: the laptop is waiting for them.

### 2. P4 validation on the cluster

Overrides:
- Small datasets (SplitMNIST, 20NG, Cora): `--samples-per-class 50` (20 orders).
- CIFAR-100 and the ImageNet proxy: `--samples-per-class 50 --orders 3`.

| Config | Datasets in scope | Notes |
|---|---|---|
| `e1_main` | all four | Run CIFAR rows with `--filter dataset=cifar100 --orders 3` into their own run dir |
| `e2_ser_levels` | SplitMNIST | |
| `e5_firstorder_diloco` | SplitMNIST | |
| `e6_group_size` | SplitMNIST | |
| `e7_buffer` | SplitMNIST, CIFAR-100 | CIFAR at 3 orders |
| `e8_federated` | SplitMNIST | |
| `r1_ablation`, `r2_compute_matched`, `r3_sensitivity` | SplitMNIST, CIFAR-100 | CIFAR at 3 orders |
| `e9_imagenet` (proxy copy) | Tiny-ImageNet proxy | Optional; build the proxy on a login node first |

Every job is independent and resumable. A Slurm array task can run one job (`--job N`) or one slice
(`--all --filter method=X --filter dataset=Y`). `python -m htcl.run status CONFIG [overrides]` shows
done / partial / pending, and rerunning a command resumes partial jobs.

**Compute nodes may lack internet.** Download MNIST, CIFAR-100, 20NG (scikit-learn cache) and Cora on a
login node first. One `plan` plus a 1-job run per dataset does it.

### Reporting back (numbers only, no figures)

- Run `python -m htcl.aggregate CONFIG [same overrides]` for each experiment.
- Export every finished run directory's small files into `results/cluster/<run dir name>/`, keeping the
  `<job_id>/` layout, so the laptop can point `figures/*.py --runs` at it:

  ```bash
  rsync -a --prune-empty-dirs --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
        --include='DONE' --exclude='*' runs/<run dir name>/ results/cluster/<run dir name>/
  ```

  This keeps `jobs.jsonl`, `summary.{csv,json}` and, per job, `job.json`, `metrics.json`,
  `acc_matrix.json`, `timing.json` and `DONE`. It never copies checkpoints or model files.
- `results/` is git-ignored, so add these files with `git add -f results/cluster`.
- Commit to the branch `cluster/results` (create it from `tnnls-release` once; later pushes add commits)
  and push. Commit message: the experiments and overrides included.
- Write `results/cluster/VALIDATION_REPORT.md` covering:
  - per config: jobs done / failed and failure messages;
  - ACC, forgetting and BWT means and stds across orders;
  - anything that looks wrong: NaN, chance-level rows, negative forgetting, HTCL worse than its base;
  - wall time per job on the GPU type used.
- Never commit checkpoints, `final_model.pt`, data or anything under `runs/`.
- Code fixes found on the cluster go to `tnnls-release` only after the author agrees, never mixed into
  `cluster/results` commits.
- On the laptop, `git fetch origin cluster/results && git checkout origin/cluster/results -- results/cluster`
  brings the numbers in, without switching branches.
