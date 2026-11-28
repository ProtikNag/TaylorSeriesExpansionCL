# Guard-acceptance reconciliation (cluster vs laptop)

Both counts are correct over their own set of run dirs. The 47% (cluster) and 42% (laptop) differ only
because the cluster set included two dirs that were not on `cluster/results`; now pushed.

| run dir | dataset | num_tasks | k | guarded steps | accepted | on cluster/results before |
|---|---|---|---|---|---|---|
| _intended | SplitMNIST | 5 | 2 | 15 (er/der/ser) | 3 | yes |
| _intended | SplitMNIST | 5 | 4 | 0 | 0 | yes |
| _scale_mnist_nr | SplitMNIST | 5 | 2 | 15 (ewc/ft/sr) | 1 | yes |
| _scale_20ng | 20NG | 5 | 2 | 15 (er/der/ser) | 15 | yes |
| _scale_cora | Cora | 3 | 2 | 0 | 0 | yes |
| **_scale_cifar** | CIFAR-100 | 10 | 2 | **40 (er/der)** | **23** | **NO → pushed now** |
| **_scale_mnist** | SplitMNIST | 5 | 2 | **15 (ewc/ft/sr)** | **5** | **NO → pushed now** |

- Laptop set = the first five rows = 45 steps, 19 accepted = **42%**.
- Cluster set = all seven rows = 100 steps, 47 accepted = **47%**.
- Difference = `_scale_cifar` (40 steps, 23 acc) + `_scale_mnist` (15 steps, 5 acc).

**Caveats on the two pushed dirs:**
- `_scale_cifar` is reduced-data CIFAR (`samples_per_class=200`, ACC ~15%) — a code-path validation run,
  NOT paper numbers. It is the only long-sequence (10-task) guard data we have, hence the higher accept
  rate (~57%): a 10-task k=2 stream makes 4 Taylor steps per job, vs 1 per job on the 5-task sets.
- `_scale_mnist` is the **superseded pre-replay-fix** SplitMNIST run for ewc/ft/sr. It was replaced by
  `_scale_mnist_nr` after commit c1919a0 (replay-protected local). Do not use it for paper numbers; it is
  pushed only so the guard count reconciles.

**Algorithm-of-record accept rate** (drop superseded `_scale_mnist`, keep CIFAR as the long-sequence
witness): `_intended` + `_scale_mnist_nr` + `_scale_20ng` + `_scale_cifar` = 85 steps, 42 accepted = **49%**.
Pushed-numbers-only (no CIFAR) = 45 steps, 19 = **42%**.
