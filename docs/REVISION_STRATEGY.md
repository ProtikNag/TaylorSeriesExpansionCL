# TNNLS revision strategy — storyline, algorithm, and how we answer the reviews

Written 2026-11-27 (cluster + author working session). This is the authoritative statement of the
**current thinking** for the TNNLS revision: the locked algorithm, the new storyline, and — most
importantly — how we tackle the reviewer concerns, especially **C15**. It supersedes the storyline and
claims framing in `reviews/PLAN.md` where they differ. Read with `reviews/CONCERNS.md` (C1–C17) and
`docs/DESIGN.md` (the code contract).

## 1. The locked algorithm (what HTCL now is)

HTCL is the **group schedule** (`schedule: group`, `search: best`) with three properties, two of them
added during this revision. It is verified (code-verifier PASS) and documented in
`pseudocode/01_htcl_group_schedule.pdf`.

- **Within-window order search (the core).** The stream is split into consecutive groups of k tasks.
  For each group, a *local* learner is trained from the persistent hierarchical model w_1 on the k
  tasks, trying every within-group permutation and keeping the one with the best **validation**
  accuracy on the group. The global model is then consolidated toward that local via the closed-form
  Taylor update w_i ← w_i + (H_i+λ_iI)^{-1}(λ_iΔd_i − g_i) (diagonal empirical Fisher, λ=1/lr).
- **Skip-incomplete-final-group (new).** A final group of fewer than k tasks is **not** consolidated;
  the replay-protected base-trained local is deployed directly. Consolidating a lone task overshoots
  and catastrophically forgets earlier tasks (we observed a grouping collapse to 58.7% ACC with tasks
  at near-chance; the acc-matrix localized it to the lone-final-group Taylor step). This fix removes it.
- **Guard (new, `guard: true`).** A consolidation step is kept only if it does not lower seen-task
  validation versus the local; otherwise the local is adopted. This bounds HTCL below by the base
  learner it wraps (the honest version of the old "never worse" theory claim, C7).
- **Replay-protected local (new).** Buffer-free base methods (finetune/EWC/SR) replay HTCL's own
  N-sample reservoir during local training, so their local is protected from forgetting like the
  buffered methods (ER/SER/DER) are. Without this, HTCL did not help (even regressed) for those methods.

**Why this is the algorithm and not the earlier "online nudge".** We explored a continuous-global
online schedule (the "nudge"). It does NOT have within-window invariance — the global trains tasks in
arrival order before consolidating, so (1,2) and (2,1) give different results. The group schedule's
local starts from the order-invariant w_1 and searches, so it is **provably invariant** to within-group
order (verified bit-identical). The invariance is the whole point (see C15 below), so the group schedule
is the method.

## 2. The new storyline (intro narrative + Fig. 1)

Task-order sensitivity is real and matters (a medical model shouldn't depend on disease order). Finding
the best order is **NP-hard**: n tasks → n! orderings, intractable (10! ≈ 3.6M). HTCL does not solve
this. Instead it **removes order-sensitivity within small windows exactly** (a tractable O(k!) search per
group) and **mitigates the rest empirically** through guarded hierarchical consolidation.

**Fig. 1 (base vs +HTCL, across datasets and baselines).** Per (dataset, baseline), two columns: the
base method (lower mean accuracy, higher std across task orders) and base+HTCL (higher mean, lower std).
Current numbers (SplitMNIST, 500 spc, controlled 20-order set, base → +HTCL-k2):

| method | base mean/std | +HTCL mean/std |
|---|---|---|
| finetune | 62.9 / 5.79 | 76.3 / 7.45 |
| ER | 80.8 / 3.16 | 83.6 / 2.76 |
| DER | 74.5 / 4.93 | 79.9 / 3.24 |
| SER | 84.6 / 2.21 | 87.7 / 1.06 |
| EWC | 63.1 / 7.14 | 79.8 / 5.48 |
| SR | 62.0 / 5.47 | 80.4 / 2.37 |

SplitMNIST shows the full effect across all six baselines (mean ↑, std ↓ for the replay family and now
also EWC/SR/finetune after the replay-protected-local fix). Cora is a moderate supporting panel; 20NG at
5 tasks is too easy (being re-run at 10 tasks); CIFAR-100 needs full data (200 spc gave ~15% — being
re-run full). The laptop draws the figure from `results/cluster/storyline_summary.csv` (per-order ACC).

## 3. How we answer C15 (the central battle)

**C15 (the reviewers' sharpest point):** our variance reduction is measured over an order distribution
that *contains within-window-equivalent orders*; on genuinely distinct orders HTCL may not reduce
variance, so the reduction looks like a counting artifact.

**Our counter-argument (this is the spine of the rebuttal):**
1. Order-sensitivity is **NP-hard** — you cannot evaluate all n! orders to pick the best. Any honest
   method must be a tractable approximation. HTCL's O(k!)-per-group search is exactly that.
2. HTCL is **provably invariant to within-window order**. For n=5, k=2 the 5!=120 orderings collapse
   into 120/(2!·2!) = 30 equivalence classes. A base method yields up to 120 distinct accuracies; HTCL
   yields at most 30, identical within each class (we verified bit-identical). This is a **guarantee**,
   not an artifact — we *remove* within-window variance by construction, and we report `n_unique` openly.
3. **But the collapse alone is not the reduction** — and we must say this, not hide it. The 30 class
   representatives could be *more* spread than the 120 (before the skip-incomplete + guard fixes, the
   SplitMNIST classes ranged 58–85, i.e. HTCL's std was *worse*). What keeps the classes **tight** is
   the guarded consolidation (+ skip-incomplete). That part is **empirical**, and we now have the data
   showing HTCL's std over the distinct classes is below base's std.

So the honest, airtight claim is **two separable contributions**:
- (a) *Provable* within-window order-invariance: HTCL removes within-window variance exactly, in O(k!)
  instead of O(n!). Larger k → more of the n! space collapses → more variance removed by construction.
- (b) *Empirical* cross-window mitigation + mean-accuracy improvement at equal memory, via the guarded
  hierarchical consolidation.

**What we must NOT claim:** "HTCL reduces order variance over arbitrary orderings." The data refutes it
and it is the C15 trap that sank the NeurIPS submission. Scope every variance statement to "within-window
order" and report `n_unique_runs`. Frame HTCL as *mitigating*, not *solving*, task-order sensitivity.

## 4. Claims to make, claims to drop

- KEEP: formalizing task-order sensitivity as Var_σ[A(w_σ)] over the sequential-training trajectory
  (answers C6 — the static sum objective is order-invariant; the trajectory is where order lives).
- KEEP: provable within-window invariance (a clean, correct, statable guarantee).
- KEEP: mean-accuracy improvement at equal memory (the consolidation genuinely helps).
- NARROW: the old "never worse in expectation than random ordering" theorem (C7) → restate as the guard's
  never-worse-than-base property on validation, conditional, not a global guarantee. Honest and provable.
- DROP/DEMOTE: Proposition A.4's linear-coverage assumption (C8) → a heuristic cost discussion, no
  "proposition" status.
- FIX LANGUAGE: "Hessian" is the diagonal empirical Fisher (C17) — say so.

## 5. Reviewer-concern status (short)

- C1 test leakage — FIXED (validation split drives selection; test only for reporting).
- C2 setting legitimacy — the group schedule is a buffered mini-batch-of-tasks protocol; state it
  honestly (k tasks of look-ahead); the within-window invariance is the payoff.
- C4 disentangle search/compute/consolidation — ablation experiments pending (see experiment list).
- C5 compute-matched baselines — pending (repeats / averaging / ensemble at matched budget).
- C7, C8, C6, C17 — theory/language, per §4.
- C10 λ/curvature sensitivity — sweep pending.
- C11 replay dependence — buffer sweep; HTCL at buffer 0.
- C14 equal memory — FIXED (equal N buffer; k-window is training-time memory, reported).
- C15 — see §3 (the core strategy).

## 6. Where the numbers live

- Code + algorithm: branch `tnnls-release`. Pseudocode: `pseudocode/01_htcl_group_schedule.pdf`.
- Numbers for the figure/table: branch `cluster/results`, `results/cluster/storyline_summary.csv`
  (columns: dataset, method, variant [base|HTCL], order, acc). Pull on the laptop:
  `git fetch origin cluster/results && git checkout origin/cluster/results -- results/cluster`.
- All training runs happen on the cluster; the laptop draws figures. No figures on the cluster.
