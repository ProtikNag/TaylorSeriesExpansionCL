# HTCL — the algorithm (what it is and why)

This is the single source of truth for the HTCL algorithm the paper describes. The code default *is* this
algorithm: a bare `htcl: {k: 2}` runs it. Everything else in the code is an **ablation** of one of its
parts. Read this first; `docs/DESIGN.md` is the code contract, `docs/HANDOFF.md` is the current state.

## The algorithm

HTCL wraps a base continual learner (finetune, EWC, SR, ER, SER, DER) and runs the **group schedule** with
`levels = 2` (a local model + one persistent hierarchical model `w_1`; the deployed model is `w_1`), `k = 2`.
The task stream is split into consecutive **groups of k tasks**. For each group:

1. **Within-group order search.** Train a *local* learner from the persistent `w_1` on the group, trying
   every within-group permutation, and keep the one with the best **validation** accuracy on the group.
   Because the local always starts from the order-invariant `w_1` and the search is exhaustive over the
   group, the result is **bit-identical for any within-group reshuffle of the task order** — the
   within-window order invariance (see §C15).
2. **Replay-protected local.** Buffer-free base methods (finetune/EWC/SR) replay HTCL's own N-sample
   reservoir during this local training, so the local does not forget earlier groups. Methods with their
   own buffer (ER/SER/DER) use theirs. The memory budget is the same N for everyone.
3. **Step-size merge (`merge_search`).** Compute the closed-form Taylor update of `w_1` toward the local as
   a *direction*, then line-search it: score `w_old + α·(step)` for `α ∈ {0, 0.25, 0.5, 0.75, 1}` **and**
   the local itself, each on the validation set of **all tasks seen so far**, and keep the best (ties go to
   the larger α). The chosen α is logged per merge (`htcl_merge.alpha_hist` in `metrics.json`).
4. **Merge the last group too (`merge_last`).** A final group shorter than k is merged the same way, not
   skipped — the search's α=0 / local candidates absorb the overshoot a lone-task consolidation would cause.
5. **BN / gradient buffer split.** BatchNorm recalibration of the deployed model always sees the replay
   buffer (statistics over seen tasks, not training). The Taylor **gradient** (g, H) for a buffer-free
   method is the **group only** — its injected reservoir protects the local (step 2), but the deployed
   model never *trains* on it, so a deployed no-replay model carries no replay memory.

Closed-form Taylor step at level i: `w_i ← w_i + (H_i + λ_i I)^{-1} (λ_i Δd_i − g_i)`, diagonal empirical
Fisher for H, `λ = 1/lr`. Time per group O(k!·train + |grid|·eval); space O(model + N buffer).

## Why each part (SplitMNIST, 20 orders, k=2, one seed — ACC mean/std)

| method | base | V1 no-merge | V2 single-jump | **HTCL (V3)** | V4 memory-only |
|---|---|---|---|---|---|
| finetune | 62.9/5.79 | 80.2/2.93 | 79.0/5.06 | **80.4/3.92** | 79.3/2.80 |
| er | 80.8/3.16 | 81.2/2.36 | 83.6/2.53 | **86.1/2.28** | — |
| der | 74.5/4.93 | 75.6/3.10 | 79.9/2.97 | **80.8/4.22** | — |
| ser | 84.6/2.21 | 85.3/3.14 | 87.7/0.98 | **86.5/2.03** | — |
| ewc | 63.1/7.14 | 79.0/2.79 | 80.2/3.18 | **81.6/4.09** | 78.3/4.72 |
| sr | 62.0/5.47 | 78.1/1.09 | 77.8/4.58 | **79.6/4.84** | 79.6/2.91 |

- **Order search + replay-protected local** lift mean well above base and reduce std below base on **all six**
  methods (the storyline: accuracy up, order-variance down). V1 (no merge) already shows most of this.
- **The step-size merge (V3) beats the single jump (V2) on mean for 5/6 methods** (er +2.5, der +0.9,
  finetune +1.4, ewc +1.4, sr +1.8; ser is the one where V2's tighter guard wins). The search rarely takes a
  full jump — it mostly picks a partial step (0.5–0.75) or the local, which is why it is both higher and safer.
- **`merge_last`** removes an adversarial-order collapse that skip-incomplete suffers: on `[1,4,2,3,0]` sr the
  lone final task wipes task 2 (98→50) under skip, but the merge recovers it (70.5→78.9 ACC).
- **BN keeps the buffer, gradient does not** (§5): fully dropping the buffer from BN cost sr ~2.6 pts with no
  upside; the split is performance-neutral vs. keeping it and makes "the global never trains on replay" exact.
- **V4 (memory, no HTCL)** shows that for buffer-free methods the N-sample reservoir alone recovers most of the
  gain; HTCL's distinctive contribution is clearest on the **replay methods** (er/der) where there is no memory
  confound and V3 beats base by +5.3/+6.3.

## Ablations (how to turn parts off)

| flag | what it removes | table column |
|---|---|---|
| `consolidate: false` | the merge entirely — deploy the searched local | V1 |
| `merge_search: false, guard: true` | the step-size search — single Taylor jump kept all-or-nothing | V2 |
| `merge_last: false` | merging the final short group — skip-incomplete instead | — |
| `search: random` | the within-group order search | R1 |
| `curvature: identity` | the Fisher curvature — first-order step (H=0) | R1 |
| `schedule: online` | the group schedule — consolidate every task, revise windows | R1 |
| `memory: true` (method option, no htcl) | HTCL entirely — base method + N-sample replay | V4 |

## §C15 — the order-variance claim, stated honestly

Finding the best task order is NP-hard (n! orderings). HTCL does not solve it; it **removes order-variance
within small windows exactly** (a tractable O(k!) search per group) and **mitigates the rest empirically**
through the guarded step-size merge. For n=5, k=2 the 120 orderings collapse to 30 within-window classes, and
HTCL is bit-identical within each class (verified). Scope every variance claim to *within-window* order and
report `n_unique_runs`; the merge is what keeps the distinct classes tight, and that part is empirical.

## Where the numbers live

Code + this doc: branch `tnnls-release`. Per-run metrics and the summary CSVs are on branch `cluster/results`
(`results/cluster/`); the laptop draws figures from there. No figures are made on the cluster.
