# HTCL code design (tnnls-release)

This document is the contract the code follows. When code and this document disagree, fix one of them; never leave them inconsistent.

## Package layout

```
htcl/
  __init__.py
  data.py           # datasets, splits, scenarios, subsampling
  models.py         # architectures per dataset
  buffer.py         # reservoir replay buffer (sample level)
  engine.py         # sequential CL trainer: task loop, eval, accuracy matrix, checkpoint/resume
  metrics.py        # ACC, forgetting, BWT, per-task stats, aggregation across orders
  permutations.py   # order sampling, canonicalization, dedup
  consolidation.py  # Taylor update (eq. hier_update), Fisher / Hutchinson curvature, first-order, DiLoCo
  hierarchy.py      # HTCL wrapper: group schedule (the algorithm; merge guard / step-size search), online ablation
  methods/
    __init__.py     # registry: name -> "module:Class", resolved lazily with importlib
    base.py         # Method interface
    finetune.py er.py ser.py der.py ewc.py sr.py icarl.py dualnet.py
    federated.py    # FedAvg, FedProx consolidation variants (task = client)
  run.py            # CLI: plan / run / status
  aggregate.py      # CLI: summary tables per experiment
  utils.py          # seeding, device, atomic save, timing
configs/            # one YAML per experiment
experiments/        # standalone scripts (loss landscape)
figures/            # one script per paper figure (TMLR acviz kit), FIGURE_STYLE.md
report/             # acviz.py, paperfig.py, dashboard builder + verifier, manifest.json
scripts/            # data preparation (ImageNet-100 and its proxy), Slurm/cloud launchers, anonymity check
tests/test_core.py  # one runnable check per non-trivial piece of logic
```

## Protocol

- **Scenario.**
  - `domain` (default): shared head with `classes_per_task` outputs; labels are remapped per task to `0..C-1`. Task identity is not given at inference.
  - `class`: global labels and a head over all classes. Task identity is not given at inference.
- **Splits.**
  - Each task's training data is split into train (90%) and validation (10%), stratified and seeded by the dataset seed.
  - Validation is used only for HTCL's order selection (window revision; group search) and for any tuning.
  - Test data is used only for evaluation.
- **Subsampling.**
  - `samples_per_class: int | null` caps training samples per class, drawn randomly with the dataset seed (not the first N). It is applied before the validation split.
  - The test set is capped at `test_samples_per_class` (default: same value, or `null` for the full set).
- **Task split.** Classes are assigned to tasks in a fixed seeded shuffle (dataset seed 0), independent of the order. Task id `j` always refers to the same classes. An *order* is a permutation of task ids.
- **Optimizer.** SGD, momentum 0.9, weight decay 1e-4. lr, batch size and epochs come from the dataset config (paper hyperparameter table). A fresh optimizer is created per task.

## Engine contract

A job is (experiment name, method spec, dataset spec, order, order index). `job_id` is a stable hash of the resolved job config.

- **Training.** Training is sequential: `for task in order: method.begin_task; for epoch: for batch: method.observe; method.end_task`. The epoch loop is never outside the task loop.
- **Evaluation.**
  - After each task `l` in the order, the deployed model is evaluated on the test set of every task. This fills row `l` of the accuracy matrix `A[l][task_id]`, with columns indexed by task id, not by position.
  - For HTCL, the deployed model after task `l` is w_{L-1} after that task's step (online schedule: after the revision when the task closed a window). The group schedule deploys the model of the last completed group (HTCL section).
- **Outputs.** `runs/<experiment>/<job_id>/` contains:
  - `job.json`: the resolved config
  - `acc_matrix.json`
  - `metrics.json`
  - `timing.json`
  - `checkpoint.pt`: latest state, atomic write; deleted once `DONE` is written (checkpoints hold the buffer and all HTCL levels, and the final state is in `final_model.pt`)
  - `final_model.pt`: deployed weights (HTCL: w_{L-1}; repeats: the weight average)
  - `DONE`: marker file
- **Accuracies** are percentages (0–100) everywhere.
- **Steps.** The engine iterates over *steps*: one task for baselines and HTCL's online schedule, one group of k tasks for HTCL's group schedule. A task counts as learned at the last position of its step (`learned_pos`, written to `metrics.json`).
- **Memory without HTCL (baseline control).** A method spec with `memory: true` gives a buffer-free method (finetune/EWC/SR) an N-sample reservoir (N = dataset `buffer_size`) that is replayed into each task's training stream — the same memory HTCL hands those methods, but with no hierarchy. It is a method option on the plain `Sequential` learner (not a new method, not HTCL); the reservoir is checkpointed, and the row is labelled `+mem=N` with its own job id. This isolates the replay memory from the schedule.
- **Compute-matched repeats (R2).** A method spec with `repeats: n` trains n copies of the baseline from the same initialisation in lockstep; repeat r trains task t after reseeding with `hash(job seed, r, t)`. After every task the engine records rows for each repeat, the weight average (`wavg`: mean weights, BN statistics recalibrated as for HTCL over the current task's training data plus repeat 0's buffer) and the logit ensemble (`ens`). `metrics.json` then holds `variants` {`best_val` (repeat with the best final mean validation accuracy), `mean` (mean of the repeats' metrics), `wavg`, `ens`} plus the per-repeat metrics, and `aggregate` emits one row per variant.
- **Checkpoint.** Written after every step. It holds model(s), method state (buffer, EWC anchors, …), for online HTCL the open window, its index and its snapshot S (so a job killed mid-window resumes bit-identically on CPU), the next step, the partial accuracy matrix, per-step timings, and RNG states for python, numpy, torch and the device. It is loaded with `map_location="cpu"`: `Method.load_state_dict` receives CPU tensors and moves what it needs to the device.
- **Resume.** A job with `DONE` is skipped. A job with `checkpoint.pt` restores everything and continues. A resumed job must give the same final metrics as an uninterrupted one on CPU (checked bitwise on CPU and MPS for every method). On CUDA, `seed_everything` sets cuDNN deterministic (benchmark off), `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=True)`; ops without a deterministic kernel (torch_geometric's scatter in the Cora GCN, `scatter_reduce` in iCaRL's NME scores) only warn, so CUDA reruns and resumes agree within floating-point nondeterminism (not bitwise) on those paths.
- **Seeds.** One training seed for the whole paper (`PAPER_SEED = 0` in `htcl/run.py`, overridable per config with `seed`), identical across methods and task orders, so the spread across orders is due to the order alone. The seed is part of the job id. Seeding happens at job start, so a job's result does not depend on which other jobs ran before it in the same process. R2 repeats derive their per-repeat seeds from it.

## Metrics (`metrics.py`)

For an accuracy matrix `A` (T×T, rows = after training position l, cols = task id), with `j` ranging over tasks in learning order:

- **ACC** = mean of the final row.
- **Forgetting** = $\frac{1}{|F|}\sum_{j \in F} \max_{l \in [\text{lp}(j), T-2]} (A_{l,j}) - A_{T-1,j}$. This is the standard definition: max over rows at or after task j was learned, excluding the final row.
- **BWT** = $\frac{1}{|F|}\sum_{j \in F} (A_{T-1,j} - A_{\text{lp}(j),j})$.
- Here lp(j) is the row at which j counts as learned and F = {j : lp(j) < T-1} are the tasks that can be forgotten. For baselines and online HTCL lp(j) = pos(j) and |F| = T-1, which is the textbook formula. For group-schedule HTCL, lp(j) is the last position of j's group, because the deployed model only contains j once the group is consolidated; using pos(j) would compare against the previous group's model, which never saw j, and inflate BWT. Tasks of the last group are not in F. Both are NaN when F is empty (T = 1 or a single group).
- **Common (group) basis.** Only needed for the group schedule: it never counts forgetting inside a group and drops its last group from F. `metrics.json` therefore also holds `group_basis` {k: {forgetting, bwt}}: forgetting and BWT with lp(j) = end of j's group of k applied to the same matrix. Group HTCL-k reports its own k (equal to its main values); every other job (baselines, every variant, repeats included, and online HTCL) reports every k in 2..T-1. `aggregate` emits `forgetting_g{k}`, `bwt_g{k}` for each group-schedule k in the experiment (NaN for group rows with another k). Online HTCL and baselines share the standard definitions and are directly comparable. Tables that put group-schedule HTCL and baselines together use these columns, or footnote the asymmetry. In particular, R1 puts a `schedule: group` row next to online rows: its main `forgetting_mean`/`bwt_mean` are on the group basis, so that row is compared on `forgetting_g2`/`bwt_g2` (emitted for every R1 row).
- **Across orders.** Compute the mean and **sample std (ddof=1)** of ACC, forgetting and BWT; the per-task final accuracy mean and std (by task id); mean wall-clock; `n_orders` and `n_unique_runs`.
- **Canonical duplicates.** For group-schedule HTCL, rows that are canonical duplicates reuse the canonical run's metrics. They are included in the std, per the protocol decision, and `n_unique_runs` is reported next to it. The duplicate's per-task accuracies are mapped by **task id**, so they are correct for the duplicate order too.

## Permutations

- `sample_orders(T, n, seed)` draws `min(n, T!)` distinct orders uniformly at random. When `T! <= n` it returns all of them.
- `canonical(order, k)` sorts task ids within each consecutive group of size k; the last group may be smaller.
- When jobs are generated, group-schedule HTCL jobs whose canonical form was already scheduled become aliases (`alias_of=<job_id>`) and are not run. Baseline jobs are never aliased. Online HTCL jobs are never aliased: their rows depend on the arrival order. Group HTCL with `search: random` is not aliased either. Its σ is drawn from the global RNG, whose state at a group start depends only on the canonical history, so aliasing would be valid; it is left out because no experiment uses it and the invariance is tested only for `search: best`.
- Orders are drawn with `orders_seed` (default 0) and depend only on T, so every method and experiment with the same T sees the same orders. Draws are sequential with rejection, so a reduced `--orders n` run uses a prefix of the full run's orders (when n < T!).

## HTCL (`hierarchy.py`, `consolidation.py`)

Notation from the paper:
- The levels are local model w_ℓ and hierarchical models w_1 … w_{L-1}. `levels: L` counts the local model too, so HTCL-L2 = local + w_1, and HTCL-L3 = local + w_1 + w_2.
- The deployed model is w_{L-1}.
- **TaylorStep**(w_i toward target): Δd = target − w_i; g_i, H_i at w_i, computed on the consolidation data (the task(s) just learned plus the replay buffer) in **eval mode** (no BN stat updates); w_i ← w_i + (H_i + λ_i I)^{-1}(λ_i Δd − g_i), elementwise with diagonal H. After each level's update its BN statistics are recalibrated: reset the running stats and run forward passes in train mode with `no_grad` over the same consolidation data, **shuffled** (the data is a concatenation of task blocks; task-homogeneous batches would miss the between-task variance) with the incomplete trailing batch dropped, so the cumulative average weights batches equally.
- The very first consolidation initializes every level to the local model with BN recalibrated (unless `consolidate: false`). Otherwise w_1 would keep the local model's training EMA, which after a short first task is far from converged (CIFAR-100, spc 10, 1 epoch: |g| at w_1 = 8.3e4 with the EMA against 7.9 recalibrated) and made the next Taylor step diverge.

### Online schedule (`schedule: online`, documented ablation)

The learner never waits for a group: every arriving task is learned and consolidated at once, so every row of the accuracy matrix is a real, up-to-date deployed model. Every k tasks (a *window*) the learner revises the window by replaying it in the best order. One engine step per task t:

1. **Window opens** (window empty, k > 1): snapshot S of everything a revision may roll back: all w_i, the method state (buffer, EWC anchors, …), HTCL's own buffer, DiLoCo momentum, FedAvg counter. The window list starts empty.
2. **Seed** the task's local training with `derive_seed(job seed, window index, position in window, task id, window prefix up to and including t)`. Everything in steps 3-5 draws from this stream, so it is a function of (S, window prefix) only.
3. **Local.** local ← w_1 (the method state carries on), `method.rebase()` (EWC's anchors move to w_1), train task t with the base method (`train_task`). FedProx adds (μ/2)‖w − w_1‖².
4. **Cascade.** w_1 ← TaylorStep(w_1 toward local); w_i ← TaylorStep(w_i toward w_{i-1}) for i ≥ 2, on task t's training data plus the replay buffer; λ_i and η_i per level.
5. Methods without a replay buffer add task t's training data to HTCL's own N-sample reservoir.
6. Append t to the window. When it holds k tasks, **revise**:
   - The arrival order's score is the mean **validation** accuracy of the deployed model w_{L-1} on the window's tasks, computed on the online state.
   - For every other permutation σ of the window (`itertools.permutations` order): restore S, replay steps 2-5 for the tasks in σ order (same seeding rule, so the prefix is σ's), score the same way.
   - Adopt the highest-scoring state (all w_i, method state, buffers). Ties keep the arrival order (the revision is then a no-op); among other ties the first in permutations order wins. Clear the window.
   - Because of the seeding rule, replaying the arrival order from S reproduces the online state bit for bit (tested), so the arrival candidate is never retrained: a revision trains (k! − 1)·k extra tasks.
   - Every revision (arrival order, adopted order, all scores) is logged in `metrics.json` (`htcl_revisions`).
7. Record row t of the accuracy matrix with the deployed model after this step (after the revision when the step closed a window).

Tasks left in an incomplete window at the end of the sequence are not revised. `k: 1` never revises (plain online consolidation). Rows are one per task, so forgetting and BWT use the standard per-task definitions (lp(j) = pos(j)). Results depend on the arrival order inside a window, so online jobs are never aliased.

**Memory.** During a window the local learner holds the training data of up to k past tasks (`htcl_window_tasks` = k in `metrics.json`) so that a revision can replay them. This is extra **training-time** memory; the deployed model's replay buffer is the baselines' N, and HTCL gets no extra deployed buffer.

**λ default.** `lam: null` (default) resolves at planning time to 1/lr of the dataset, so the number is in the job spec and the job id. With H ≈ 0 the update reduces to Δd − g/λ, so λ = 1/lr makes the curvature-free limit one SGD step from the local model. λ = 1 collapsed to chance in probes (SplitMNIST, lr 0.01: λ 1 → 50%, λ 100 → 74.6%). Configs that set `lam` keep their value.

### Group schedule (`schedule: group`, the algorithm the paper describes)

The algorithm of record (k=2). The learner waits for a group G_t of k consecutive tasks (the last group may be smaller), and the deployed model only changes at group ends. All paper configs set `schedule: group` explicitly; the code default stays `online` for back-compatible ablations. One engine step per group:

1. Snapshot the method state at the group start (buffer, EWC anchors, etc.).
2. **Local search.** For each permutation σ of the group's tasks (sorted, lexicographic): deep-copy w_1^{(t-1)} and the method state snapshot, `method.rebase()`, train sequentially on σ with the base method, score A^σ = mean validation accuracy of the local model over the group's tasks. Ties are broken by first σ.
3. Adopt the winning local model w_ℓ^{(t)} and the winning run's method state.
4. First group: initialize every level as above. Otherwise cascade TaylorStep (w_1 toward w_ℓ^{(t)}, w_i toward w_{i-1}^{(t)}) on the group's training data plus the replay buffer, with BN recalibration.
5. Methods without a replay buffer: add the winning run's training data (in σ order) to HTCL's own reservoir.
6. Within a group the deployed model is the previous group's, so those rows repeat the test row recorded when that group finished (kept in the HTCL state; re-evaluating the old weights with the new method state would be wrong for iCaRL's exemplar classifier). The first group deploys its own result. A task counts as learned at the end of its group (Metrics).

The consolidation data is built from the **sorted** group, because its block order drives BN recalibration's shuffle and the Fisher subset. With that, the group schedule with `search: best` is invariant to the order inside a group (tested: two canonically equal orders give identical matrices), so its canonical duplicates are aliased (Permutations).

**Replay-protected local.** Buffer-free base methods (finetune, EWC, SR) replay HTCL's own N-sample reservoir during local training (step 2), so their local is protected from forgetting earlier groups like the buffered methods (ER/SER/DER) are. Without this HTCL did not help those methods.

**Merge of the cascade into the hierarchy** — one of two mutually exclusive policies on the Taylor step:
- `guard: true` (the shipped default behaviour): keep the whole cascade only if the deployed model's seen-task validation is not below the local's; otherwise adopt the local at every level (the "never worse than base" property, C7). All-or-nothing.
- `merge_search: true`: treat the cascade as a *direction*. Candidate levels are w_old + α·(w_stepped − w_old) for α in `merge_grid` (default {0, .25, .5, .75, 1}), plus the local model itself. Each candidate's BN is recalibrated, then its deployed model is scored on seen-task validation; the best wins, ties go to the larger α (the local candidate ranks below α=0). `merge_grid: [1.0]` with the tie rule reproduces the single Taylor jump. The chosen α (or `"local"`) per merge is logged in `metrics.json` under `htcl_merge` (`alpha_hist`, `log`).

**Incomplete final group.** A final group shorter than k is, by default, *not* merged (`merge_last: false`): consolidating a lone task overshoots and wipes earlier tasks, so the replay-protected base-trained local is deployed directly. `merge_last: true` merges it like any other group (safe only with `merge_search`, whose α=0/local candidates absorb the overshoot).

**Curvature.** J is the **mean** cross-entropy over the consolidation data (the step's training data + buffer); g is computed over all of it. `fisher` (default) is the mean over per-sample gradients squared, computed on a seeded random subset of up to `curvature_samples` samples (default 512) with `torch.func.vmap(grad)` in memory-bounded chunks (fallback: a per-sample loop, with a warning). `hutchinson` is the diagonal Hessian estimate E[z ⊙ Hz] with Rademacher z (`hutchinson_probes`, default 4), where Hz is a forward-over-reverse product (`torch.func.jvp` of `grad`; double backward through max-pooling is unsupported on MPS); negative entries are clipped to 0 so H + λI stays positive. `identity` sets H = 0, the first-order variant (eq. firstorder). Because J is a per-sample mean, Fisher entries are small (smoke measurement on SplitMNIST: median ≈ 0, max ≈ 0.25), so H only changes (H + λI) appreciably for λ ≲ 0.1; at the default λ = 1/lr (10 to 1000) the step is close to first order.

**Hyperparameter defaults** (`hierarchy.DEFAULTS`, filled into the resolved job spec so they enter the job id): **schedule group, merge_search true, merge_last true** (the algorithm of record — a bare `htcl: {k: 2}` runs it), levels 2, k 2, λ null = 1/lr (scalar or one value per level), curvature fisher, curvature_samples 512, aggregator taylor, search best, consolidate true, merge_grid {0,.25,.5,.75,1}, guard false (the single-jump+guard V2 ablation sets merge_search false, guard true), step_scale 1.0, clip_norm none, fedprox_mu 0.01, diloco_lr 1.0, diloco_momentum 0.0.

**Model consolidated.** Only `method.model` is consolidated; a method that keeps extra networks outside `model` is not covered.

**Variants**:
- `step_scale` (η, scalar or per level): w ← w + η · update
- `clip_norm`: caps the global L2 norm of each level's Taylor update
- `aggregator`:
  - `taylor` (default)
  - `diloco`: outer Nesterov step on the pseudo-gradient w_1 − mean_σ(w_ℓ^σ) over all k! local candidates (group schedule; online each step has one candidate, so the mean is the local model and w_1 ← local). The defaults (lr 1, momentum 0) make it w_1 ← mean_σ(w_ℓ^σ) exactly, the paper's DiLoCo variant (appendix) and E5's "DiLoCo mean" row; `diloco_lr`/`diloco_momentum` (e.g. 0.7/0.9, the DiLoCo paper's outer optimizer) give the momentum form, which is not what the paper reports.
  - `fedavg`: cumulative FedAvg, w_1 ← running average of the local models across steps (group schedule: the selected ones), weighted by the number of training samples. This is not the literal FedAvg rule: with one client per round, FedAvg would set w_1 to the local model, which is `consolidate: false` (E8 reference row). Level i > 1 runs the same running average of the freshly updated level i-1, with the same sample counts. Implemented through `methods/federated.py` (`fedavg_update`, `fedprox_penalty`).
  - `fedprox`: local training adds (μ/2)‖w − w_1‖²; aggregation as fedavg
- `search: random`: pick a random σ instead of the best one (ablation R1). Online: each revision adopts a uniformly drawn permutation of the window, from `random.Random(derive_seed(job seed, window index, "random", arrival order of the window))` (no replay when it is the arrival order). The arrival order is in the seed so that, under the single paper seed, different task orders draw independently.
- `consolidate: false`: deploy the local model, with no consolidation (ablation R1); every level becomes a copy of it, and the next step starts from it. Online, this is the local model of the last task (revisions still score and adopt window orders, using that model as the deployed one).

## Methods (`methods/`)

`Method(model, cfg, device)` (`methods/base.py`) implements:
- `begin_task(task_id, train_loader)`: the base class creates a fresh SGD optimizer; the engine calls `model.train()` just before it
- `observe(x, y, task_id) -> loss`: inputs already on the device; the loss must go through `self.optimize(loss)`, which adds the FedProx term when HTCL sets `self.prox`
- `end_task(task_id, train_loader)`
- `prepare_eval()`: called before every evaluation (e.g. recompute iCaRL class means with the current weights)
- `predict(x) -> logits`: must depend only on the current `self.model` weights and the method state, because HTCL and R2 load other weights into `self.model` before evaluating
- `state_dict()` and `load_state_dict()`, covering everything needed for resume (tensors arrive on the CPU)

`cfg` is the resolved dataset spec (lr, momentum, weight_decay, batch_size, epochs, buffer_size, scenario, …) plus `num_outputs`, `num_tasks` and the method spec's own keys (α, β, λ, …). Each method lists its hyperparameter defaults in `DEFAULTS` (or overrides the classmethod `defaults(dataset)`, as SER does for its per-dataset α/γ); `run.resolve_method` fills them into the method spec at planning time, so they enter the job id and `job.json`. The registry (`methods/__init__.py`) maps every name in the table to `module:Class` and imports lazily. Every model exposes `.features(x)` (penultimate features) and a linear `.head`.

The replay buffer is `buffer.ReservoirBuffer(capacity_in_samples)`. It stores (x, y, task_id, logits?) on the CPU; `sample(n, device)` returns (x, y, task, logits-or-None) and raises on an empty buffer (callers check `len(buffer)` first). Its randomness uses torch's global generator, which the checkpoint restores. HTCL uses `method.buffer` as the replay part of the consolidation data. A method without one (finetune, EWC, SR) gets HTCL's own `ReservoirBuffer(buffer_size)`, filled after each step's consolidation with that step's training data (group schedule: the adopted run's) and saved in the HTCL state, so the consolidation data always covers earlier tasks (paper: replay buffer plus the current task(s); PLAN D4). The standalone finetune/EWC/SR baselines store 0 samples, so their HTCL rows use N raw past samples (in g, H, BN recalibration and catch-up) that the baseline does not: this is a capacity difference, not a matched comparison. Every such job's label carries `+replay=N` (`hierarchy.NO_BUFFER_METHODS`, checked against the method classes in tests), and tables must keep that tag or a footnote, so replay gains are not attributed to the hierarchy. No matched control (baseline + the same reservoir) is run yet.

| name | Algorithm (reference) |
|---|---|
| finetune | plain SGD, lower reference |
| er | reservoir replay: each step concatenates a buffer batch of equal size (Chaudhry et al. 2019) |
| der | DER / DER++: MSE on stored logits (α), optional CE on buffer labels (β; 0 means DER) (Buzzega et al. 2020) |
| ser | Strong Experience Replay: backward consistency (logit MSE on buffer, as DER) plus forward consistency (current-batch logits vs. the old model's logits; the frozen model sees the same joint current+buffer batch, in train mode with BatchNorm and in eval mode without it, so the term is 0 at the task start) (Zhuo et al. 2023) |
| ewc | EWC with per-sample diagonal Fisher after each task, sum of quadratic penalties (Kirkpatrick et al. 2017); model kept in train mode during training. The Fisher is computed in eval mode on a copy whose BN statistics are recalibrated on the task's training data (the training EMA is stale after a short task: CIFAR-100 spc 20, 1 epoch, max F 9.5e8 against 65). Stability cap: each coordinate's penalty is scaled by min(1, 1/(lr·λ·Σ_t F_t)), so heavy-ball SGD (unstable above lr·curvature = 3.8) stays stable; it is inactive wherever lr·λ·ΣF ≤ 1. λ per dataset: 10 on CIFAR-100 (λ ≥ 50 diverges at lr 0.1 without the cap; sweep {1, 10, 50} in `ewc.py`), 400 elsewhere |
| sr | spectral regularization: penalize (σ_max(W)^p − 1)² per layer (Lewandowski et al. 2024) |
| icarl | iCaRL: herding exemplars on penultimate features keyed by (task_id, label), over every training sample of the task under the plain transform (`TaskDataset.plain`, unshuffled, no dropped batch), stored in that view; distillation, NME classifier at inference (Rebuffi et al. 2017) |
| dualnet | DualNet: slow learner trained with Barlow Twins SSL on the buffer (own SGD, no weight decay: the supervised optimizer already applies the protocol's 1e-4 to those parameters), fast learner adaptation on the slow features (Pham et al. 2021) |

## Datasets (paper Tables)

| key | classes | tasks | per task | model | lr | batch | epochs | buffer |
|---|---|---|---|---|---|---|---|---|
| splitmnist | 10 | 5 | 2 | conv→1×1 proj→ResNet-18 | 0.01 | 64 | 10 | 50 |
| cifar100 | 100 | 10 (5/20 for storyline) | 10 | ResNet-18 (3×3 stem, no maxpool) | 0.1 | 32 | 10 | 500 |
| news20 | 20 | 5 | 4 | MLP 20000→1024→512→C, TF-IDF fit on train only, headers/footers/quotes removed | 1e-3 | 32 | 10 | 100 |
| cora | 7 | 3 | 3/2/2 | 2-layer GCN on the full graph, task = node mask, stratified 80/20 node split | 0.01 | full | 10 | 50 |
| imagenet100 | 100 | 10 | 10 | timm ViT-S/16 pretrained | 1e-3 | 32 | 5 | 500 |
| mnist_classil3 | 10 | 3 | {0,1,2},{3,4,5},{6..9} | MLP | 0.01 | 64 | 10 | 50 |

For local validation, `imagenet100` can point at a proxy ImageFolder built from Tiny-ImageNet (100 classes, upsampled to 224). Proxy runs are labelled as such everywhere.

Data notes:
- Class-to-task assignment is a seeded shuffle for every dataset except `mnist_classil3` (fixed groups). SplitMNIST pairs are therefore not the consecutive digits the paper mentions; only the pairing differs, since orders are permuted anyway.
- MNIST and CIFAR-100 are normalised with their channel statistics; CIFAR-100 train items get crop (pad 4) and horizontal flip. ImageNet-100 uses timm's transforms from the `vit_small_patch16_224` pretrained config (mean = std = 0.5, bicubic, crop 0.9). Validation and test use the plain transforms.
- Cora items are node indices; the GCN holds the full graph (non-persistent buffers) and returns the requested rows, so every step propagates over the whole graph. `batch_size: full` means one batch per task.
- A shuffled loader holding more than one batch drops its incomplete trailing batch (a batch of 2–4 samples blows up BatchNorm at the SplitMNIST ResNet's 1×1 spatial size; reshuffling each epoch still visits every sample). A dataset of at most one batch is kept whole.

## CLI

```
python -m htcl.run plan   configs/e1_main.yaml [--samples-per-class 50] [--orders 3] [--epochs 1]
python -m htcl.run run    configs/e1_main.yaml (--job N | --all) [--filter method=ser ...] [overrides] [--device mps]
python -m htcl.run status configs/e1_main.yaml [overrides]
python -m htcl.aggregate  configs/e1_main.yaml [overrides]   # runs/<exp>/summary.{csv,json}
```

Overrides change the experiment's run directory suffix (e.g. `runs/e1_main__spc50/`, `__spc10_o2_ep1`), so reduced runs never mix with full runs. `--samples-per-class` also caps the test set unless the config sets `test_samples_per_class`. `--runs-root DIR` moves the experiment directories under DIR (smoke runs use `runs/_smoke`). `run` and `status` re-plan deterministically and rewrite `jobs.jsonl`; `--job N` indexes it (0-based, for array jobs); `--filter key=value` (repeatable, ANDed) matches a dotted dataset/method field by full name or suffix (`num_tasks=5`, `htcl.levels=2`), with `method`/`dataset` meaning their names and `htcl=True|False`. Under `--all` a failing job is reported and the rest continue. `--max-steps N` stops each job after N steps, leaving its checkpoint (resume testing). An `imagenet_root` containing `PROXY.txt` (written by `scripts/make_imagenet100_proxy.py`) adds `proxy=True` to the dataset label.

**Config schema.** `experiment`, `orders`, optional `orders_seed`, and either `datasets` + `methods` (cartesian product) or `blocks: [{datasets, methods}, …]`. A dataset entry needs `name, lr, batch_size, epochs, buffer_size` (defaults: scenario domain, seed 0, momentum 0.9, weight_decay 1e-4, data_root data, num_tasks per dataset). A method entry is `{name, <method keys>, htcl: {…}}` or `{name, repeats: n}`. Any entry may carry `sweep: {dotted.key: [values]}`, expanded as a cartesian product. Keys starting with `_` (YAML anchors for shared hyperparameters) are ignored. Labels in summaries are the dataset name plus swept keys (and `scenario=class`), and the method name plus its as-written keys.
