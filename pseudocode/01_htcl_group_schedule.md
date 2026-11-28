# HTCL — The Algorithm (Group Schedule)

Source: `htcl/hierarchy.py` (`learn`, `_learn_group`, `_consolidate`, `_merge_search`,
`_consolidation_data`, `_bn_data`, `_taylor`; the all-or-nothing `_guard_consolidation` is the V2
ablation) and `htcl/consolidation.py` helpers (`mean_gradient`, `fisher_diag`, `hutchinson_diag`,
`taylor_step`, `recalibrate_bn`, `average_states`). Spec: `docs/ALGORITHM.md`. Mirrors the current
implementation (default `htcl: {k: 2}`, `levels = 2`).

This documents the **group schedule** — the algorithm the paper describes. The task stream is split
into consecutive groups of $k$ tasks; each group runs an exhaustive within-group order search, then a
**step-size (line-searched) Taylor merge** cascades the result into the hierarchy. A final short group
is merged the same way (`merge_last`). The online/nudge schedule is an ablation and not covered here.

## Notation and inputs

- $T_1, T_2, \dots$ — a stream of tasks in a fixed arrival order, processed in consecutive groups of
  size $k$. One engine step = one group (`learn` dispatches to `_learn_group` when `schedule == group`).
- $k$ — group size (`cfg.k`, default $2$).
- Base continual learner $\mathcal{M}$ (finetune, EWC, SR, ER, SER, DER); its model is the **local
  model**. ER/SER/DER carry their own $N$-sample buffer. Buffer-free methods (finetune, EWC, SR) get
  HTCL's own $N$-sample **reservoir** instead, so deployed memory is always $N$ for everyone.
- Hierarchy of $L$ levels (`cfg.levels`, default $2$); the local counts as one, so there are $L-1$
  persistent hierarchical models $w_1, \dots, w_{L-1}$. The **deployed model is $w_{L-1}$** (for $L=2$,
  the single persistent $w_1$).
- $\lambda_i$ per level (`cfg.lam`, default $1/\text{lr}$); diagonal curvature $H_i$ (empirical Fisher
  by default); step scale $\eta_i$ (`cfg.step_scale`, default $1$); optional global update clip.
- $\alpha$-grid for the merge line search (`cfg.merge_grid`, default $\{0, 0.25, 0.5, 0.75, 1\}$).
- Selection uses the 10% validation split; test is for reporting only.

## Top level — process one group

**procedure** LEARN-GROUP(group of tasks $G$):  $\quad$ `_learn_group`

```
  snapshot <- copy of the base learner's full state (model + method state + own buffer)
  start    <- copy of w_1's weights, or None if the hierarchy is empty (first group)

  // ---- (1) WITHIN-GROUP ORDER SEARCH (k! permutations) ----
  sigmas <- all permutations of sorted(G)              // canonical base order
  if search == "random": sigmas <- one permutation drawn uniformly

  // Replay for the LOCAL learner: buffer-free methods replay HTCL's reservoir here so the local
  // does not forget earlier groups; ER/SER/DER already replay their own buffer inside TRAIN-TASK.
  replay <- HTCL's reservoir as a dataset, if HTCL owns one and it is non-empty, else None

  best_score <- -inf ; best_state, best_sigma <- None ; candidates <- []
  for each permutation sigma in sigmas:
      restore the base learner from snapshot
      if start is not None:                            // TrainSeq(sigma, w_1): start every run from w_1
          load start into the local model
          M.rebase()                                   // method anchors (e.g. EWC) move to w_1
      if aggregator == "fedprox": set the prox anchor to the current weights
      for task_id in sigma:
          TRAIN-TASK(M, task_id, replay=replay)        // base learner; buffer-free methods get reservoir
      clear the prox anchor
      score <- mean over G of VAL-ACCURACY(M, G)       // selection on the group's validation split
      if score > best_score:                           // strict '>' => the first permutation wins ties
          best_score, best_state, best_sigma <- score, copy(M state), sigma
      if aggregator == "diloco": append copy of local weights to candidates

  load best_state into the base learner                // adopt the best-permutation local
  seen_tasks <- seen_tasks + G
```

The exhaustive search over $sorted(G)$ with a strict `>` tie-break makes the adopted local
**bit-identical for any within-group reshuffle** of the arrival order (within-window order invariance,
§C15). Selection scores only $G$'s validation tasks.

## Merge-the-last-group rule and hierarchy update

```
  // ---- (2) HIERARCHY UPDATE ----
  if hierarchy exists AND |G| < k AND NOT merge_last:  // (ablation: merge_last == False -> skip-incomplete)
      every level w_i <- copy of the adopted local model
  else:
      CONSOLIDATE(sorted(G), candidates)               // THE algorithm: merge, incl. a short final group

  // ---- (3) grow memory AFTER consolidation (the buffer then holds earlier groups only) ----
  if HTCL owns a reservoir:
      add every (x, y) of the adopted run's tasks (in best_sigma stream order) to the reservoir

  // ---- deployed-model bookkeeping ----
  top_row <- TEST-ACCURACY(w_{L-1})                    // deployed model after this group
  mid_row <- (first group ? top_row : previous group's top_row)
  return one row per task position: [mid_row]*(|G|-1) + [top_row]
```

> `merge_last = True` is the default: a final group shorter than $k$ is **merged like any other** (the
> line search's $\alpha=0$ / local candidates absorb the overshoot a lone-task consolidation would
> cause). The skip-incomplete branch (copy the local into every level) is the `merge_last = False`
> ablation. Within a group the deployed model is still the previous group's, so the mid-group rows
> repeat the previous recorded top row; the first group reports its own result.

## Consolidation — cascade, then line search

**procedure** CONSOLIDATE(group $G$, diloco candidates):  $\quad$ `_consolidate`

```
  local <- adopted base-learner model
  n_new <- number of training samples across the tasks in G

  // First group (or cfg.consolidate == False, the V1 ablation): seed the hierarchy from the local.
  if hierarchy is empty OR not cfg.consolidate:
      first <- copy(local)
      if cfg.consolidate: RECALIBRATE-BN(first, BN-DATA(G))   // same BN treatment as later levels
      w_1 .. w_{L-1} <- (L-1) copies of first ; reset momentum ; seen_samples += n_new
      return

  // ---- the two data views (the buffer split, §5) ----
  grad_data <- CONSOLIDATION-DATA(G)   // Taylor g/H: group + the METHOD's OWN buffer only (ER/SER/DER)
  bn_data   <- BN-DATA(G)              // BN stats: group + ANY buffer (HTCL reservoir INCLUDED)

  target <- local
  if aggregator == "diloco":           // target = elementwise mean of the k! candidate locals
      target <- model with weights = AVERAGE(candidates)

  // merge_search (default, taylor): snapshot each level BEFORE the step so the cascade becomes a
  // direction to line-search along.
  old_params <- copy of every level's trainable weights   (only when merge_search and aggregator==taylor)

  // ---- cascade: level 0 is pulled toward the local; each higher level toward the level below ----
  for i, level w_i in enumerate(levels):
      if aggregator == "taylor":    TAYLOR-STEP(w_i, target, grad_data, i)
      elif aggregator == "diloco":  outer Nesterov step on pseudo-grad (w_i - target)
      elif aggregator in {fedavg, fedprox}:  weighted average toward target
      RECALIBRATE-BN(w_i, bn_data)         // BN refit after each level, on the reservoir-inclusive view
      target <- w_i                        // next level's target is this (updated) level

  if old_params is not None:               // THE algorithm
      MERGE-SEARCH(old_params, local, bn_data)
  elif cfg.guard and aggregator == "taylor":   // V2 ablation only
      GUARD-CONSOLIDATION(local, bn_data)

  seen_samples += n_new
```

After the cascade, `self.levels` already holds the **full Taylor step** ($\alpha=1$). `merge_search`
treats that step as a *direction* and line-searches it; the guard (V2) instead keeps or rejects it
whole.

### The Taylor step (the consolidation math)

**procedure** TAYLOR-STEP(level $w_i$, target, grad_data, $i$):  $\quad$ `_taylor` + `taylor_step`

```
  g_i <- MEAN-GRADIENT of the cross-entropy over grad_data, at w_i          (eval mode)
  H_i <- diagonal curvature at w_i over grad_data:
           fisher      -> mean of squared per-sample gradients (empirical Fisher), up to c samples
           hutchinson  -> E[z (.) Hz], Rademacher z, probes passes, negatives clamped to 0
           identity    -> 0   (first-order limit, H = 0)
  Delta_d_i <- target - w_i        // level 0: toward the local; level i>0: toward w_{i-1}
```

Closed-form update (eq. `hier_update`, elementwise on the diagonal):

    $w_i \leftarrow w_i + \eta_i\,(H_i + \lambda_i I)^{-1}\,(\lambda_i\,\Delta d_i - g_i)$

- With $H_i \to 0$: $w_i + \eta_i(\Delta d_i - g_i/\lambda_i)$; at $\lambda_i = 1/\text{lr}$ that is one
  SGD step from the local model, so the curvature-free limit is interpretable.
- $g_i$, $H_i$ run in **eval mode** on `grad_data` = the group's training data plus the method's own
  buffer (ER/SER/DER) — NOT HTCL's reservoir. Fisher/Hutchinson use a random subset of
  `cfg.curvature_samples` items.
- Optional global $L_2$ clip caps the update norm before applying $\eta_i$.

### The step-size merge (default)

**procedure** MERGE-SEARCH(old_params, local, bn_data):  $\quad$ `_merge_search`

```
  stepped <- current (post-cascade, alpha=1) weights of every level

  candidates <- [ ]
  for alpha in merge_grid:                      // e.g. {0, .25, .5, .75, 1}
      levels_a <- for each level: old_i + alpha * (stepped_i - old_i)
      RECALIBRATE-BN(each level in levels_a, bn_data)
      candidates.append( (rank = alpha, label = alpha, levels_a) )
  candidates.append( (rank = -1, label = "local", BN-recalibrated copies of local at every level) )

  // Score each candidate's DEPLOYED model (levels[-1]) on ALL seen tasks' validation; pick the best.
  for each candidate: score <- mean VAL-ACCURACY(levels[-1], seen_tasks)
  best <- argmax over (score, rank)             // ties -> larger alpha; the local ranks below alpha=0
  self.levels <- best candidate's levels ; reset momentum
  log the chosen alpha (or "local") and every candidate's score   // htcl_merge.alpha_hist in metrics.json
```

- The candidate set is the grid points $w_\text{old} + \alpha(w_\text{stepped} - w_\text{old})$ **plus
  the just-trained local itself** at every level.
- Each candidate is BN-recalibrated on `bn_data` (reservoir-inclusive) before scoring.
- Scoring is on **all seen tasks' validation**, not just $G$. The best wins; ties go to the larger
  $\alpha$, and the `local` candidate ranks below $\alpha=0$ (so it is adopted only when strictly best).
- With `merge_grid = [1.0]` the sole grid candidate is exactly the single all-or-nothing Taylor jump.
- The search rarely takes a full jump — it usually picks a partial step ($\alpha \in \{0.5, 0.75\}$) or
  the local, which is what makes it both higher-accuracy and safer than a lone $\alpha=1$ step.

### The all-or-nothing guard (V2 ablation only)

**procedure** GUARD-CONSOLIDATION(local, bn_data):  $\quad$ `_guard_consolidation`

```
  local_val <- mean VAL-ACCURACY of local        over ALL seen tasks
  cons_val  <- mean VAL-ACCURACY of w_{L-1}       over ALL seen tasks
  took_step <- (cons_val >= local_val)
  log took_step, cons_val, local_val
  if took_step: return                           // keep the Taylor cascade whole
  kept <- copy(local) ; RECALIBRATE-BN(kept, bn_data)
  every level w_i <- copy of kept ; reset momentum // reject the whole step -> revert to the local
```

Reached only under `merge_search = False, guard = True` (the single-jump V2 column). It keeps the full
Taylor step or reverts entirely to the local — no intermediate $\alpha$.

## The two data views (the buffer split, §5)

**CONSOLIDATION-DATA($G$)** (`_consolidation_data`) — the **Taylor gradient** view:
`ConcatDataset(G's training data) + the METHOD'S OWN buffer` if it has one (ER/SER/DER). HTCL's
injected reservoir for buffer-free methods is **excluded** — it protects the local learner only, so a
deployed no-replay model never trains on replay memory.

**BN-DATA($G$)** (`_bn_data`) — the **BatchNorm recalibration** view: `ConcatDataset(G's training data)
+ ANY available buffer` — the method's own buffer (ER/SER/DER), else HTCL's reservoir (finetune/EWC/SR).
The reservoir is included even for no-replay methods, because BN recalibration estimates deployment
statistics over seen tasks (not gradient training) and keeping it preserves earlier-task BN stats
(dropping it cost SR ~2.6 pts).

## BatchNorm recalibration

`recalibrate_bn` resets BN running stats and re-estimates them with a **cumulative average** (momentum
set to `None`) over a **shuffled** pass of the data in train mode. Shuffling mixes task blocks so the
between-task channel-mean spread enters `running_var`; `make_loader` drops the trailing incomplete
batch (equal weight per batch) and single-sample batches are skipped. It runs when seeding the first
level, after every cascaded level, and on every merge-search / guard candidate before scoring.

## Complexity

Let $k$ = group size, $P$ = trainable parameters, $L$ = levels, $m$ = samples per task,
$B$ = gradient-data size, $c$ = `curvature_samples`, $E$ = base-learner epochs,
$|G_\text{grid}|$ = `len(merge_grid) + 1` (grid points plus the local candidate).

- **Time per full group.** The order search dominates: $k!$ permutations $\times$ $k$ tasks of base
  training $= O(k!\cdot k\cdot E\cdot m)$ forward/backward, plus $k!$ validation passes. The cascade is
  $O(L(B + c)P)$ (one gradient pass over $B$ and a curvature pass over $c$ per level) plus $O(LBP)$ BN
  recalibration. The line search adds $|G_\text{grid}|$ candidates, each $O(LP)$ to form, $O(LBP)$ to
  BN-recalibrate, and one whole-seen-set validation sweep to score. $k$ is kept small ($2$–$3$): $k!$
  is the practical ceiling.
- **Space.** $O(LP)$ for the hierarchy and $O(LP)$ for the line-search candidate held while scoring,
  $O(P)$ for one diagonal curvature vector (never the full Hessian), the $N$-sample buffer, and one
  $O(P)$ group snapshot during the search. DiLoCo additionally keeps $O(LP)$ momentum and up to $k!$
  candidate copies.

## Faithfulness notes

- `merge_search = True` is the implemented default (THE algorithm); the all-or-nothing guard is reached
  only under the `merge_search = False, guard = True` (V2) ablation. The old skip-incomplete + guard
  write-up described the previous default.
- Order-search ties break to the **first permutation** of $sorted(G)$ (strict `>`), giving deterministic
  within-group order invariance.
- Consolidation always uses $sorted(G)$ so the group's arrival order never reaches BN shuffling or the
  Fisher subset selection.
- The Taylor **gradient** $(g, H)$ excludes HTCL's reservoir (`_consolidation_data`); **BN
  recalibration** includes it (`_bn_data`). Methods with their own buffer keep it in both.
- `merge_search` first applies the full cascade into `self.levels`, snapshots `old_params` **before**
  the cascade, and reconstructs each $\alpha$ candidate as $w_\text{old} + \alpha(w_\text{stepped} -
  w_\text{old})$; the local candidate is the base learner at every level.
- No code-vs-math discrepancy found: `taylor_step` implements eq. `hier_update` exactly on the
  diagonal, with $\Delta d_i = \text{target} - w_i$ and target cascading down the levels.
