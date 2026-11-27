# Full Rerun Plan and Cost: UofSC Hyperion + Rented GPUs

Revised 2026-11-25. Every experiment in the paper is rerun on audited code, plus the rebuttal experiments.

**Protocol.**
- 20 distinct task orders per experiment, drawn uniformly at random, or all $t!$ orders when $t! < 20$.
- Orders are canonicalized (sorted within each group) when the job list is generated.
- HTCL runs once per canonical class, and duplicate orders reuse its metrics. Baselines run on every order.
- Single seed.

## 0. Preconditions

1. **Code audit and fixes.** At minimum C14 (replay buffer), C1 (validation split), C16 (baseline wiring), and the permutation and job generator, plus anything the audit finds. No compute is spent before this.
2. **Missing code.** These experiments have no implementation in the repo: FedAvg/FedProx comparison, first-order/DiLoCo variants, loss landscape, group-size sweep driver, and the motivation box plots (MER, BCL, sine regression). Each needs to be found or rewritten.
3. **Calibration (~$5).** Measure post-fix per-permutation time for one baseline and HTCL on CIFAR-100 (Hyperion) and ImageNet-100 (one 4090 pod). **Every number below scales with these two measurements.**

## 1. Unit cost per permutation (A100-class, post-fix, mostly assumed)

| Dataset | Tasks | Distinct orders used | Unique HTCL runs (k=2) | Baseline | HTCL-L2 |
|---|---|---|---|---|---|
| SplitMNIST | 5 | 20 | ~15 | 25 s (measured) | ~125 s |
| 20 Newsgroups | 5 | 20 | ~15 | ~25 s (assumed) | ~125 s |
| Cora | 3 | 6 (all) | 3 | ~25 s (assumed) | ~125 s |
| CIFAR-100 | 10 | 20 | ~20 | 180 s (measured) | ~900 s (assumed 5×) |
| ImageNet-100 | 10 | 20 | ~20 | 0.5–1 h (estimate) | 2.5–5 h (estimate) |

With $t=10$, collisions among 20 random orders are negligible: there are $10!/2^5 = 113{,}400$ canonical classes. Canonical reuse therefore saves compute mainly on 5-task datasets.

## 2. Experiment inventory

| ID | Paper item | Configuration | A100-h |
|---|---|---|---|
| E1 | Fig. SER results (a–e) | SplitMNIST, SER, HTCL-L2…L5 | ~2 |
| E2 | Fig. long sequences | CIFAR-100, DER, HTCL-L3 | (in E4) |
| E3 | Fig. loss landscape | SplitMNIST Class-IL, 3 tasks | ~0.5 |
| E4 | Main table | 7 baselines (ER, SER, DER, EWC, SR, iCaRL, DualNet) × {base, L2, L3} × 4 datasets | MNIST 9, 20NG 9, Cora 2, **CIFAR 81** |
| E5 | Motivation (BCL table, forgetting vs. #tasks, box plots) | EWC, MER, BCL, ER × image/text/graph/sine, 50 sequences | ~5 (code missing) |
| E6 | Group-size break-even | SplitMNIST, SER, k = 1…5 | ~5 |
| E7 | First-order / DiLoCo table | SplitMNIST, 5 baselines × 2 variants | ~4 |
| E8 | ImageNet-100 table | 5 baselines × {base, L2} × {Domain-IL, Class-IL} | **600–1,200** |
| E9 | Buffer-size figure | ER, DER, SER × {MNIST, CIFAR} × 4 extra buffer sizes × {base, L2} | MNIST 8, **CIFAR 72** |
| E10 | Federated comparison | SplitMNIST, SER & DER × {HTCL, FedAvg, FedProx} | ~2 |
| R1 | C4 ablation arms | SER & DER × {MNIST, CIFAR} × 3 arms | MNIST 2.5, CIFAR 18 |
| R2 | C5 compute-matched baselines | SER & DER × {MNIST, CIFAR}, 5 repeats | MNIST 1.4, CIFAR 10 |
| R3 | C10/C17 λ and curvature sweep | SER × {MNIST, CIFAR} × 7 configs | MNIST 3.6, CIFAR 35 |

| Group | A100-h |
|---|---|
| Small: SplitMNIST, 20NG, Cora, E1, E3, E5–E7, E10, small parts of R1–R3 | ~55 |
| CIFAR-100: E4, E9, R1–R3 | ~215 |
| ImageNet-100, full table (E8) | 600–1,200 |
| ImageNet-100, Domain-IL only | 300–600 |

## 3. Split

| Where | What | Job shape |
|---|---|---|
| **Hyperion** (free) | All small-dataset work (~55 h) and all CIFAR-100 work (~215 h) | One array task per (config, permutation): ≤ 2 min small, ≤ 20 min CIFAR. Short jobs backfill in a slow queue. |
| **Rented GPUs** | ImageNet-100 (E8), plus any CIFAR not finished by a cutoff date | One pod per (baseline, setting), each running permutations in parallel |
| **Mac** | Audit, fixes, smoke tests, missing code, writing | n/a |

CIFAR fallback: if Hyperion has not finished a whole CIFAR table by the cutoff, rerun that table on 4090s (~$0.34 per A100-h equivalent, ~$73 for all CIFAR work). Don't mix hardware within a table.

## 4. Rental cost for ImageNet-100

RTX 4090 assumed ~1.25× slower than an A100 for ViT-S. Prices are on-demand, Nov 2026: 4090 $0.34/h, A100 PCIe $1.19/h (RunPod Community).

| ImageNet scope | A100-h | RTX 4090 | A100 PCIe |
|---|---|---|---|
| Full table: 5 baselines × Domain-IL + Class-IL | 600–1,200 | **$255–510** | $714–1,428 |
| Domain-IL only, 5 baselines | 300–600 | **$128–255** | $357–714 |
| Domain-IL only, SER + EWC | 120–240 | **$51–102** | $143–286 |

Wall-clock: 20 parallel 4090 pods finish the full table in ~1.5–3 days, Domain-IL only in ~1–1.5 days. Community 4090 availability can cap how many pods you get.

## 5. Total spend (with 20% contingency)

| Scenario | ImageNet | CIFAR overflow | Storage | **Total** |
|---|---|---|---|---|
| Full ImageNet table, Hyperion delivers CIFAR | $255–510 | $0 | ~$5 | **$310–620** |
| Full ImageNet table, CIFAR on 4090s | $255–510 | $73 | ~$5 | **$400–705** |
| Domain-IL only (5 baselines), Hyperion delivers CIFAR | $128–255 | $0 | ~$5 | **$160–315** |
| Domain-IL only (SER + EWC), Hyperion delivers CIFAR | $51–102 | $0 | ~$5 | **$67–130** |

The range is wide because the post-fix ImageNet time is unmeasured. The $5 calibration in §0 narrows it to about ±20%.

## 6. Cost levers (your call)

| Lever | Saves |
|---|---|
| Drop ImageNet Class-IL (the current code doesn't implement standard Class-IL anyway; C3) | ~50% of ImageNet |
| ImageNet with SER + EWC only (as the original script did) | another ~60% |
| Buffer sweep on CIFAR with 2 extra sizes instead of 4 | ~36 A100-h of Hyperion time |
| λ sweep with 3 values instead of 5 | ~10 A100-h of Hyperion time |

## Sources

- Measured: `results/**/csv/*.csv` per-permutation `Time_s`; Mac calibration runs 2026-11-25; paper Fig. (e).
- [RunPod pricing](https://www.runpod.io/pricing); [RunPod GPU prices](https://www.runpod.io/gpu-prices); [Cloud GPU prices Nov 2026](https://altstreet.investments/tools/gpu/prices)
- [USC Research Computing resources](https://www.sc.edu/about/offices_and_divisions/division_of_information_technology/rc/resources/)
