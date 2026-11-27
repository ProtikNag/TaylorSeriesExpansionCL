# Review Concerns Tracker (NeurIPS 2026 → TNNLS)

Source reviews: [neurips2026/reviews.md](neurips2026/reviews.md). Paper: [../paper/tnnls_main.tex](../paper/tnnls_main.tex).

Status legend: `open` · `in progress` · `done` · `wontfix (justified)`

Concerns are deduplicated across reviewers and ordered by the sequence in which we should address them. C1 comes first because every number in the paper depends on it.

| ID | Concern | Raised by | Type | Status |
|---|---|---|---|---|
| C1 | `Eval` data for permutation selection; test leakage | AC (1), SA31 (Q1) | code + paper | open |
| C2 | Setting legitimacy: intra-group lookahead vs. online CL | AC (1), cfM1, SA31 | paper | open |
| C3 | ImageNet-100 "class-incremental" protocol and output space | AC (4) | code + paper | open |
| C4 | Disentangle search vs. compute vs. second-order consolidation | AC (2), cfM1, SA31 (Q2) | experiments | open |
| C5 | Compute-matched baselines | AC (2), cfM1, SA31 (Q2) | experiments | open |
| C6 | Eqs. (1)–(2) are permutation-invariant | AC, tHSB | paper (theory) | open |
| C7 | Theorem A.3 overstated; no guarantee on final model | AC (3), cfM1, tHSB, SA31 (Q3) | paper (theory) | open |
| C8 | Proposition A.4 coverage assumption unjustified | AC, tHSB, SA31 | paper (theory) | open |
| C9 | Novelty of update vs. proximal / curvature / order-robust CL | AC (3), tHSB, cfM1 | paper (related work) | open |
| C10 | Sensitivity to diagonal Hessian and $\lambda$ | cfM1 | experiments | open |
| C11 | Dependence on replay; no-buffer / small-buffer results on all datasets | cfM1 | experiments | open |
| C12 | Limitations section too weak | cfM1 | paper | open |
| C13 | (self-found) ImageNet-100 table says 200 permutations; `gpu_imagenet100.sh` runs `--num-perms 3` | internal | code + paper | open |
| C14 | (self-found, **critical**) HTCL replays all past tasks' full training data; baselines get 50–500 samples | internal | code + rerun | open |
| C15 | (self-found) Permutation sampling: 40 "permutations" are within-group swaps of 2–10 base orders; HTCL is invariant to these by construction, so its std is computed over 2 (CIFAR) or 10 (MNIST) unique runs vs. 40 for baselines | internal | code + rerun | open |
| C16 | (self-found) Saved HTCL results are identical across ER / DER / SER baselines | internal | code check | open |
| C17 | (self-found) "Hessian" in the code is the diagonal empirical Fisher (squared batch gradients) | internal | paper | open |

---

## C1. What data does `Eval` use? Is test data excluded from selection?

**Reviewers.** AC point 1; SA31 weakness 1 and Q1.

**Current state (verified in code).** Selection uses the **test set**.
- [htcl/methods/htcl.py:656-660](../htcl/methods/htcl.py#L656-L660) passes `group_test` (built from `test_loaders`) into `select_best_permutation` as `val_loaders`.
- [htcl/methods/htcl.py:426](../htcl/methods/htcl.py#L426) then scores each intra-group permutation on those loaders.
- A second selection step exists that the paper does not describe. `global_catchup` keeps the best iterate by accuracy on `combined_loader` (current-group train data plus replay), [htcl/methods/htcl.py:366-369](../htcl/methods/htcl.py#L366-L369). This is train-set selection, not test leakage, but it is undisclosed model selection.

**Paper locations.** Algorithm 1 line `Evaluate Acc ← Eval(w)` in [paper/sections/method.tex:107](../paper/sections/method.tex#L107); $\mathrm{Eval}_\tau$ in [paper/sections/appendix_v2.tex:92-93](../paper/sections/appendix_v2.tex#L92-L93). Neither states the data source.

**Plan.**
1. Add a per-task held-out validation split carved from the training set (e.g. 10%), disjoint from test, in `htcl/data/datasets.py`.
2. Route `select_best_permutation` to validation loaders; keep `test_loaders` for final reporting only.
3. Rerun all HTCL results. Baseline numbers do not change (they do no selection), but every HTCL row does.
4. Document both selection steps (permutation selection on validation, catch-up best-iterate selection on train+replay) in the method section and in Algorithm 1.

**Notes.** This is the highest-risk item. Until it is fixed, no other experimental concern should be run, since all new numbers would inherit the leak.

---

## C2. What access does HTCL require within a group? Does it fit continual learning?

**Reviewers.** AC point 1 (second half); cfM1 (strict online CL, "tasks cannot be revisited or reordered"); SA31 weakness 1 (intra-group lookahead, latency).

**Plan.** State the setting explicitly as a *buffered / mini-batch-of-tasks* CL protocol: tasks arrive in groups of $k$, the learner may hold all $k$ task datasets until the group closes, then discards them except for the replay buffer. Contrast with strict online CL. Quantify latency ($k$ tasks of delay). Consider a $k=1$ row (no search, consolidation only) as the "online-compatible" variant; this doubles as an ablation arm for C4.

---

## C3. ImageNet-100 class-incremental protocol

**Reviewer.** AC point 4.

**Current state (verified in code).** Every model has one shared head with `num_classes_per_task` outputs ([htcl/config/config.py:94](../htcl/config/config.py#L94), [htcl/models/architectures.py:26](../htcl/models/architectures.py#L26)), and labels are remapped per task to $\{0,\dots,C_t-1\}$ via `RemappedSubset` ([htcl/data/datasets.py:120-133](../htcl/data/datasets.py#L120-L133)). That is a domain-incremental output space (shared 10-way head, no task ID at inference). It is not class-incremental under the standard definition (growing 100-way head, no task ID).

**Paper locations.** [paper/sections/appendix_v2.tex:566](../paper/sections/appendix_v2.tex#L566), [:572](../paper/sections/appendix_v2.tex#L572), [:631-632](../paper/sections/appendix_v2.tex#L631-L632), [:758](../paper/sections/appendix_v2.tex#L758) (claims "both domain-incremental and class-incremental"). Also check the loss-landscape experiment, described as class-incremental in [paper/sections/experiments.tex:186](../paper/sections/experiments.tex#L186).

**Plan.** Need to confirm how the "class-incremental" ImageNet runs in the table were produced (maybe a separate code path or the uncommitted changes in `htcl/data/datasets.py`). Then either (a) implement true Class-IL (global labels, 100-way head, no task ID) and rerun, or (b) relabel the results as Domain-IL and drop the Class-IL claim.

---

## C4. Disentangle permutation search, extra compute, and second-order consolidation

**Reviewers.** AC point 2 (specific design); cfM1 Q4; SA31 weakness 2, Q2.

**Required arms (AC's suggested control).** Hold the trained intra-group permutations fixed, then compare:

| Arm | Intra-group order | Consolidation |
|---|---|---|
| A | selected (best on val) | second-order Taylor (full HTCL) |
| B | random | second-order Taylor |
| C | selected | none (sequential training in selected order) |
| D | selected | first-order proximal ($H=0$, exists in [appendix_v2.tex:700](../paper/sections/appendix_v2.tex#L700)) |
| E | random | none (= baseline) |

A−B isolates search; A−C isolates consolidation; A−D isolates curvature. Since all arms reuse the same trained local models, this is cheap once C1 is fixed.

---

## C5. Compute-matched baselines

**Reviewers.** AC point 2 ("existing compute-matched evidence"); cfM1 (≈4× overhead; repeated training, model averaging, checkpoint selection, HP tuning at equal wall clock); SA31 Q2 (repeated baseline training with best/avg selection, ensembling with same number of local models).

**Current evidence.** Wall-clock in [paper/sections/experiments.tex:177](../paper/sections/experiments.tex#L177) (28.3 s SER vs. HTCL figure); break-even sweep over $k$ in [appendix_v2.tex:668-691](../paper/sections/appendix_v2.tex#L668-L691).

**Plan.** At matched wall clock (or matched gradient steps): (i) baseline trained $r$ times with different seeds, best-on-val and average reported; (ii) uniform weight averaging of the $k!$ local models; (iii) logit ensemble of the same models. Report as one compact table.

---

## C6. Eqs. (1)–(2) do not capture task-order sensitivity

**Reviewers.** tHSB weakness 1; AC.

**Paper location.** [paper/sections/method.tex:9](../paper/sections/method.tex#L9) (`eq:standCL`), [:19](../paper/sections/method.tex#L19) (`eq:tiCL`).

**Plan.** Reformulate in terms of the trajectory: $w_\sigma = \mathcal{A}(w_0; D_{\sigma(1)},\dots,D_{\sigma(t)})$ for a sequential learner $\mathcal{A}$, with order sensitivity defined as $\mathrm{Var}_\sigma[\mathcal{L}(w_\sigma)]$ or $\max_\sigma - \min_\sigma$. The sum-of-losses objective stays as the *evaluation* criterion, not the source of order dependence.

---

## C7. Theorem A.3 is overstated

**Reviewers.** All three plus AC point 3.

**Paper locations.** Main-text claim "HTCL's performance is never worse, in expectation, than encountering tasks in a uniformly random order" at [paper/sections/method.tex:154-155](../paper/sections/method.tex#L154-L155); [appendix_v2.tex:89](../paper/sections/appendix_v2.tex#L89) (`prop:group_vs_random`); abstract ("establish a performance bound on HTCL's expected accuracy, showing that it improves over random task orderings") in [paper/tnnls_main.tex](../paper/tnnls_main.tex); intro [paper/sections/introduction.tex](../paper/sections/introduction.tex).

**Options.** (a) Narrow: restate as a property of the intra-group selection rule on the validation metric, conditional on fixed incoming weights, and rewrite abstract/intro accordingly. (b) Strengthen: a bound on the final consolidated model, e.g. under a local quadratic model where the Taylor step has bounded increase in $\mathcal{L}$ on previous tasks controlled by $\lambda$ and $\|H\|$. Option (a) is mandatory regardless; (b) is a stretch goal.

---

## C8. Proposition A.4 coverage assumption

**Reviewers.** tHSB, SA31, AC.

**Paper location.** [appendix_v2.tex:122-187](../paper/sections/appendix_v2.tex#L122-L187), Assumption (ii) at line 148.

**Plan.** Either demote to a heuristic cost/coverage discussion (remove "proposition" status) or support the linear-coverage assumption empirically (accuracy gap vs. fraction of permutation space explored, using the $k$ sweep). Demotion is the honest default.

---

## C9. Novelty of the consolidation update

**Reviewers.** tHSB (standard quadratic Taylor / proximal), cfM1 (Hessian approximations, not model-agnostic), AC point 3.

**Plan.** Add a positioning paragraph and comparison table against: proximal / Lookahead-style updates, EWC / online-EWC / K-FAC Laplace (curvature-based), Natural-gradient consolidation, model merging with Fisher weighting, and order-robust CL work (identify the specific "recent prior work" tHSB alludes to). Be explicit that the update rule itself is a known quadratic-proximal form, and locate the contribution in the hierarchy + selection + consolidation combination, if C4 supports that.

---

## C10. Sensitivity to diagonal Hessian approximation and $\lambda$

**Reviewer.** cfM1 Q3.

**Plan.** $\lambda$ sweep and full vs. diagonal vs. identity Hessian on SplitMNIST and CIFAR-100. The identity case overlaps with arm D of C4.

---

## C11. Dependence on replay

**Reviewer.** cfM1 Q5 and limitations.

**Current evidence.** Buffer-size study for replay methods ([appendix_v2.tex:804-806](../paper/sections/appendix_v2.tex#L804-L806)); EWC+HTCL already has no replay buffer for the baseline, but check whether HTCL itself still uses `replay_datasets` in that case ([htcl/methods/htcl.py:641](../htcl/methods/htcl.py#L641)).

**Plan.** Extend buffer sweep (0, small, default) to all datasets; report HTCL with buffer 0.

---

## C12. Limitations section

**Reviewer.** cfM1.

**Paper location.** [appendix_v2.tex:832](../paper/sections/appendix_v2.tex#L832).

**Plan.** Move to main text. Cover extra compute, replay dependence, scope of the guarantee (after C7), $k!$ scalability, and the controlled-group vs. uncontrolled-stream gap (after C2). Write last.

---

## C14. HTCL replay buffer holds entire past-task datasets (critical)

**Found while costing experiments, 2026-11-25.** In [htcl/methods/htcl.py:700-704](../htcl/methods/htcl.py#L700-L704), `replay_datasets` collects whole task datasets (`l.dataset`), and the cap `len(replay_datasets) > buffer_size` counts datasets, not samples. Sequences have 5–10 tasks and `buffer_size` is 50–500, so the cap never triggers. Every HTCL local-training loader (line 642) and every Taylor/catch-up loader (line 666) concatenates all previous tasks' full training sets. Baselines get a 50–500-sample buffer.

**Consequences.** (i) The HTCL-vs-baseline comparison is not replay-matched, so accuracy and forgetting gains may largely reflect near-joint training. (ii) Training data grows with every group, which explains why an HTCL permutation on CIFAR-100 costs ~20× a baseline permutation (3,100–3,900 s vs. 160–180 s), not the ~4× reported.

**Present in the run commit.** Identical code at `8366fa0` ("Run Dec 25th"), the commit matching the timestamps in `results/`.

**Plan.** Replace it with a sample-level reservoir buffer of `buffer_size` examples, using the same buffer class as the baselines (`htcl/methods/buffer.py`). Then rerun everything. Nothing should be rented before this lands.

## C15. Permutation sampling and duplicated HTCL rows

`generate_canonical_permutations` draws a few base orders and then enumerates their within-group swaps. HTCL searches all within-group orders, so all swaps of one base order give the same HTCL run (`canonical_to_results` caches it). In `results/cifar100/*/csv/htcl_L*_results_CIFAR100.csv`, 40 rows hold 2 unique runs; in SplitMNIST, 40 rows hold 10. The baseline std is measured over within-group swaps, exactly the variation HTCL removes by design. So the reported variance reduction is partly a construction artifact, and the HTCL std rests on 2 samples for CIFAR-100.

**Decision (2026-11-25).** 20 permutations per experiment, drawn uniformly at random as distinct orders, or all $t!$ orders when $t! < 20$ (e.g. Cora). Before any job is scheduled, each order is mapped to its canonical form (within-group sort). HTCL runs once per canonical class, and duplicate orders inherit those metrics. **Baselines run on all 20 orders**, because they are not invariant to within-group order. Deduplication happens when the job list is generated, not at runtime, so parallel array jobs cannot race on the cache. The paper must state that HTCL std is computed over the 20 orders, with duplicates sharing results by construction, and report the number of unique HTCL runs.

## C16. HTCL results identical across baselines

`htcl_L{2..5}_results_SplitMNIST.csv` have identical per-permutation accuracies under `der/`, `er/` and `ser/`, and the CIFAR-100 files under `er/` and `ser/` match too. Either the baseline argument was not wired into HTCL when these ran, or the files were copied. The current code passes `baseline=baseline_lower` into `select_best_permutation`. Need to confirm which results the paper's tables came from.

## C17. Hessian vs. empirical Fisher

`estimate_diag_hessian` ([htcl/utils/helpers.py:156](../htcl/utils/helpers.py#L156)) returns the mean of squared mini-batch gradients, i.e. a diagonal empirical Fisher. The paper should say so, and C10 should compare it against an exact diagonal (`estimate_diag_hessian_exact`) and against identity.
