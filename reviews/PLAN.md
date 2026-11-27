# TNNLS Revision: Execution Plan

Written 2026-11-26. Order of work, decisions taken by default, and the full list of experiments to run.

## What the audit found (why we rebuild instead of patching)

Three parallel auditors checked the code and an independent adversarial verifier re-checked each finding (`scratchpad/understand/audit_*.json`). 41 of 43 findings were confirmed. The ones that invalidate current numbers:

1. **ER, SER and DER are not continual learning.** The epoch loop is outside the task loop (`er.py:146`, `ser.py:179`, `der.py:185`), so every epoch cycles through all tasks. This affects the standalone baselines and HTCL's local training.
2. **HTCL always runs ER locally.** The runner never passes the baseline to `train_htcl` (`runner.py:143`). This explains C16: identical HTCL results across baselines.
3. **The second-order term is numerically inert.** λ is hard-coded to 1000 and Fisher entries are about 1e-3, so the update reduces to clipped interpolation, which is the paper's own first-order ablation (`htcl.py:142`).
4. **Forgetting is never computed.** No accuracy matrix is recorded, and the plotted "forgetting" is the max−min across permutations.
5. **iCaRL memory is overwritten every task** (remapped labels). iCaRL also uses no nearest-mean-of-exemplars classifier and herds on logits.
6. **EWC leaves the model in eval()** after computing the Fisher, so BatchNorm is frozen for tasks 2 onward. Its Fisher is the batch-mean gradient squared.
7. **SER lacks its defining consistency term. DualNet cannot run. SR is unregistered.**
8. **Cora runs an MLP, not a GCN.** CLI defaults silently override per-dataset config (epochs, buffer).
9. Already known: test-set selection (C1), full-data replay in HTCL (C14), clustered permutations (C15), per-task columns misaligned between HTCL and baselines.

The training core, metrics and baselines need to be re-implemented, not patched.

## Defaults I am taking (override any of them)

| # | Decision | Default |
|---|---|---|
| D1 | Consolidation rule | Exactly eq. (hier_update): $(\mathbf H+\lambda\mathbf I)^{-1}(\lambda\Delta\mathbf d-\mathbf g)$, with λ configurable per level. The code's extra η·2^{-level} scaling and per-tensor clipping become optional flags, off by default. |
| D2 | Global catch-up (not in Algorithm 1) | Off by default. Kept as an explicit flag and reported as an ablation. |
| D3 | Curvature | Diagonal empirical Fisher from per-sample gradients (default). Option: Hutchinson diagonal Hessian. |
| D4 | HTCL replay | The same N-sample reservoir buffer the baselines use |
| D5 | `Eval` for order selection | A 10% validation split carved from each task's training data. Test data is used only for final reporting. |
| D6 | Forgetting | Standard: $\frac{1}{T-1}\sum_{j<T}\max_{l\in[j,T-1]}(A_{l,j}-A_{T,j})$. The full accuracy matrix is logged for every run. |
| D7 | Std across orders | Sample std (ddof=1) over the 20 orders, with canonical duplicates included per your rule. The number of unique HTCL runs is always reported. |
| D8 | Protocol | Domain-IL (shared head, per-task label remap) for all datasets. Class-IL is implemented as a flag (global labels, full head, no task ID) and run only if you keep the ImageNet Class-IL column. |
| D9 | Optimizer | SGD, momentum 0.9, weight decay 1e-4, per-dataset lr, batch size and epochs from the paper's hyperparameter table |
| D10 | Seeds | Seed derived from (experiment, order index), so any single job reproduces exactly on resume |

## Order of work

| Phase | What | Output |
|---|---|---|
| **P1. Repository** | Protect the paper (the original folder on Desktop no longer exists): put `paper/` and `reviews/` under their own private nested git repos. Create orphan branch **`tnnls-release`** (no history, so no author names in the log). Port the generic TMLR visualization kit (`acviz.py`, `paperfig.py`, dashboard builder and verifier, FIGURE_STYLE.md). Ignore `paper/`, `reviews/`, `example/`, `data/`, `results/` and fonts in the release. | Clean branch |
| **P2. Code rebuild** | New engine: sequential trainer, accuracy matrix, metrics, checkpoint/resume at every task boundary, permutation sampler with canonical dedup at job-generation time, YAML experiment configs, per-job CLI. Data with `samples_per_class`, validation split, Domain-IL and Class-IL, Cora GCN on the full graph. Baselines re-implemented: ER, SER, DER, EWC, SR, iCaRL, DualNet. HTCL per D1–D5, plus first-order, DiLoCo, FedAvg and FedProx (ported from `federated-comp`). Unit checks for the update equation, reservoir, canonicalization, forgetting and resume determinism. Then adversarial code review against the paper. | Audited code + tests |
| **P3. Storyline figure** | Cheap experiment (below), the figure in TMLR style, independent journal-bar review, revise until it passes | `fig_storyline.{pdf,png,svg}` + dashboard |
| **P4. Local validation run** | Every experiment below on MPS at 50 samples per class, resumable | Results + validation report + dashboard |
| **P5. Figures** | All paper figures rebuilt with the TMLR kit from P4 results, each reviewed by independent agents | Figure set |
| **P6. Scale-up kit** | Hyperion sbatch arrays (small datasets and CIFAR) and the RunPod launcher (ImageNet), same configs at full data | Launch scripts |
| P7. Paper restructure | Reorganize per the TMLR paper. Rewrite the theory claims (C6–C8). | Later, after full results |

## Experiment list

Common protocol: 20 distinct random task orders (all $t!$ when $t!<20$). Canonical dedup applies to HTCL only; baselines run every order. The validation split drives selection. Logged metrics: accuracy matrix, ACC, sample std across orders, forgetting, BWT, per-task std across orders, wall-clock, number of unique HTCL runs.

**S. Storyline (P3, new intro Fig. 1).** Question: does the final accuracy of standard methods depend on task order, and does that dependence grow with sequence length?
- Split CIFAR-100 cut into **5, 10 and 20 tasks** (20, 10 and 5 classes per task), the sequence-length stressor
- **ER** (replay family) and **EWC** (regularization family), 20 orders each
- 50 samples per class
- 120 runs, about 3 h on MPS
- HTCL is added as a third line once P2 is validated, as in the TMLR teaser where ours is one family

**Main paper experiments (P4), all at 50 samples per class:**

| ID | Paper item | Configuration |
|---|---|---|
| E1 | Main table | {SplitMNIST, CIFAR-100, 20NG, Cora} × {ER, SER, DER, EWC, SR, iCaRL, DualNet} × {base, HTCL-L2, HTCL-L3} |
| E2 | Fig. SER (a–e) | SplitMNIST, SER, HTCL-L2…L5 (L2 and L3 reused from E1) |
| E3 | Fig. long sequence | CIFAR-100, DER, HTCL-L3 (reused from E1) |
| E4 | Loss landscape | MNIST Class-IL, 3 tasks, MLP, PCA plane |
| E5 | First-order / DiLoCo table | SplitMNIST × {ER, SER, DER, EWC, SR} × {HTCL-1st (H=0), DiLoCo mean} |
| E6 | Group-size break-even | SplitMNIST, SER, k = 1…5 |
| E7 | Buffer sweep | {ER, SER, DER} ± HTCL-L2; MNIST {30, 50, 100, 200, 500}, CIFAR {50, 200, 1000} |
| E8 | Federated comparison | SplitMNIST, {SER, DER} local × {HTCL, FedAvg, FedProx} |
| E9 | ImageNet-100 (ViT-S/16) | {ER, SER, DER, EWC, SR} ± HTCL-L2, Domain-IL (Class-IL pending your decision) |
| R1 | C4 ablation | {SER, DER} × {MNIST, CIFAR} × {random order + Taylor, selected order + no consolidation, selected + first-order} |
| R2 | C5 compute-matched | {SER, DER} × {MNIST, CIFAR}: 5 repeats (best on val / mean), weight averaging, logit ensemble |
| R3 | C10/C17 sensitivity | SER × {MNIST, CIFAR}: λ ∈ {10, 100, 1000, 1e4, 1e5} × curvature {Fisher, Hutchinson diag, identity} |
| D | Old motivation box plots (BCL, MER, sine) | **Proposed: drop.** The code is not in any branch, and the storyline figure takes over their role. |

## MPS time estimate (rough; re-measured after P2)

Per-run guesses at 50 samples per class: MNIST 10 s base / 45 s HTCL; 20NG and Cora a few seconds; CIFAR 90 s / 6 min; ImageNet ViT-S 6 min / 30 min.

| Scope | 20 orders everywhere | Recommended local validation |
|---|---|---|
| Small datasets (MNIST, 20NG, Cora: E1, E2, E4–E8, R1–R3) | ~20 h | 20 orders: ~20 h |
| CIFAR-100 (E1, E3, E7, R1–R3) | ~83 h | 3 orders: ~13 h |
| ImageNet-100 Domain-IL (E9) | ~60 h | 3 orders: ~9 h |
| **Total** | **~165 h (~7 days)** | **~42 h (~1.75 days)** |

Recommendation: run the full 20-order protocol locally only on the small datasets, and 3 orders on CIFAR and ImageNet. That exercises every code path, the resume logic and every metric. The 20-order CIFAR and ImageNet runs then go to Hyperion and rentals in P6.

## Status and decisions, 2026-11-26 evening (supersedes the defaults above where they differ)

**Code.**
- The rebuild is done on branch `tnnls-release` and pushed. 39 tests pass, and every config passed
  smoke runs.
- An independent review found real bugs, which are fixed: BN recalibration ordering, the missing
  replay buffer for non-replay methods under HTCL, SER/EWC divergence handling, and invalid
  group-schedule aliasing.

**Author decisions made today:**
- **HTCL is the online schedule.** The learner does not wait for a group. Each arriving task is learned by
  the local model (from w_1) and consolidated into the hierarchy at once, so every accuracy-matrix row is a
  real, up-to-date model. Every k tasks the last window is replayed from a snapshot in every other order,
  and the order with the best *validation* accuracy is kept; ties keep the arrival order. Catch-up is
  dropped. The old group schedule is only an ablation (R1). This answers C2, the doubt that waiting for
  groups is classical CL.
- **λ defaults to 1/lr.** λ = 1 collapses to chance. With H ≈ 0, the update becomes one SGD step from
  the local model.
- **One training seed for the whole paper** (`PAPER_SEED = 0`). It is the same for every method and
  order. No seed sweeps; task orders carry the statistics (20 per experiment).
- **Memory.** The deployed models (HTCL's global model and every baseline) hold the same buffer N. The
  k-task window is training-time memory, stated explicitly. For finetune, EWC and SR, HTCL supplies the
  N-sample buffer, and the paper says that without replay these methods cannot retain memory.
- **Paper organization follows the TMLR paper**, opening with a storyline figure.

**Storyline (intro Fig. 1).**
- Setup:
  - SplitMNIST, 5 tasks of 2 digits, domain-IL (chance 50%)
  - 500 samples per class, MnistResNet, buffer 50
  - 5 random task orders, the single seed
- Methods alone (best-to-worst final accuracy across the 5 orders):

  | Method | Gap |
  |---|---|
  | Finetune | 23 points |
  | SR | 19 |
  | EWC | 16 |
  | DER | 13 |
  | SER | 6 |
  | ER | 5 |

- Argument:
  1. Benchmarks fix one order.
  2. The same tasks in another order give a different learner.
  3. Neither replay nor regularization removes this.
  4. A learner cannot choose its order but can revise it once a few tasks are seen.
  5. HTCL does that with second-order consolidation.
- Figure:
  - one column per method, a dot per order, the worst-to-best line, and Δ
  - families: no protection, replay, regularization
  - the HTCL version puts each method beside its HTCL variant (blue)
  - the author rejected marker shapes per order, schematic side panels and density plots
- The HTCL storyline runs (k = 2, L2) were running on the laptop on 2026-11-26.
- Not guaranteed. Revision guarantees only that the window's validation score is not worse than the
  arrival order's. Within-window order no longer matters, but order across windows still does.
  Tightening is an empirical claim to verify.
- Ruled out on the way:
  - 50 samples per class: seed noise was as large as the order spread.
  - CIFAR-100 at reduced data: near chance.
  - The sequence-length stressor: no growth with length.

**Compute split.**
- Laptop: storyline and figures.
- Cluster: P4 validation at 50 samples per class on CUDA (`docs/HANDOFF.md`).
- Rentals: ImageNet-100 later.

**Open:**
- Cora epochs (10 barely trains).
- CIFAR at reduced data is near chance.
- DER below ER on SplitMNIST (check α).
- E5 mixes schedules: DiLoCo uses the group schedule, so the table needs a footnote.
- The theory rewrite (C6–C8), stated for the online schedule.
- Delete the old illustrative "Standard CL vs With HTCL" histogram figure in the intro once Fig. 1 has
  HTCL; its numbers are not measured.
- **Before submission:**
  - make the repository private
  - create an anonymous code repository, as for TMLR
  - remove paper/, reviews/ and example/ from the release
  - do a thorough review
