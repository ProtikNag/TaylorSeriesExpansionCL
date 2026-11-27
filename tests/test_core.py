"""One runnable check per non-trivial piece of core logic: `.venv/bin/python -m pytest -q tests/test_core.py`."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from htcl.buffer import ReservoirBuffer
from htcl.consolidation import diloco_step, taylor_step
from htcl.engine import evaluate, run_job
from htcl.metrics import compute_metrics, group_basis
from htcl.permutations import canonical, sample_orders
from htcl.run import plan_jobs


def test_hier_update_quadratic_closed_form() -> None:
    """For J(w) = 1/2 w^T A w + b^T w (A diagonal), eq. hier_update with g = A w + b and H = A equals
    argmin_w J(w) + lam/2 ||w - w_local||^2 = (A + lam I)^{-1} (lam w_local - b)."""
    a, b, lam = torch.tensor([2.0, 0.5]), torch.tensor([-1.0, 3.0]), 0.7
    w_prev, w_local = torch.tensor([0.3, -2.0]), torch.tensor([1.5, 0.25])
    g = a * w_prev + b
    (w_new,) = taylor_step([w_prev], [w_local], [g], [a], lam)
    exact = torch.linalg.solve(torch.diag(a) + lam * torch.eye(2), lam * w_local - b)
    assert torch.allclose(w_new, exact, atol=1e-6)


def test_identity_curvature_is_first_order_rule() -> None:
    """H = 0 gives eq. firstorder: w_prev + Delta d - g / lam."""
    w_prev, w_local, g, lam = torch.randn(5), torch.randn(5), torch.randn(5), 3.0
    (w_new,) = taylor_step([w_prev], [w_local], [g], [torch.zeros(5)], lam)
    assert torch.allclose(w_new, w_prev + (w_local - w_prev) - g / lam, atol=1e-6)


def test_reservoir_capacity_and_uniformity() -> None:
    """Capacity is never exceeded, and every stream item is kept with probability capacity / n."""
    torch.manual_seed(0)
    n, capacity, trials = 500, 50, 400
    counts = np.zeros(n)
    for _ in range(trials):
        buffer = ReservoirBuffer(capacity)
        for start in range(0, n, 25):
            items = torch.arange(start, start + 25)
            buffer.add(items.float().unsqueeze(1), items, task_id=0)
        assert len(buffer) == capacity and buffer.seen == n
        counts[buffer.y.numpy()] += 1
    freq = counts / trials  # expected 0.1 for every item
    deciles = freq.reshape(10, -1).mean(1)
    assert np.all(np.abs(deciles - 0.1) < 0.015), deciles


def test_canonical_and_aliases() -> None:
    """Groups are sorted internally; group-schedule HTCL jobs with an already-planned canonical form alias
    it, baselines never alias."""
    assert canonical((3, 1, 0, 2, 4), 2) == (1, 3, 0, 2, 4)
    assert canonical((2, 0, 1), 2) == (0, 2, 1)
    orders = sample_orders(3, 20, 0)
    assert len(orders) == 6 and len(set(orders)) == 6  # 3! <= 20: all orders
    assert sample_orders(6, 5, 1) == sample_orders(6, 9, 1)[:5]  # reduced runs are a prefix
    cfg = {"experiment": "t", "orders": 6, "datasets": [{"name": "cora", "lr": 0.01, "batch_size": "full",
                                                          "epochs": 1, "buffer_size": 5}],
           "methods": [{"name": "er"}, {"name": "er", "htcl": {"levels": 2, "schedule": "group"}}]}
    jobs = plan_jobs(cfg, {})
    base = [j for j in jobs if "htcl" not in j["method"]]
    htcl = [j for j in jobs if "htcl" in j["method"]]
    assert all(not j.get("alias_of") for j in base)
    runners = {j["job_id"]: j for j in htcl if not j["alias_of"]}
    assert len(runners) == 3  # 3 canonical forms {0,1}|2, {0,2}|1, {1,2}|0
    for j in htcl:
        if j["alias_of"]:
            assert canonical(runners[j["alias_of"]]["order"], 2) == canonical(j["order"], 2)


def test_method_defaults_enter_job_spec() -> None:
    """Method defaults are written into the planned spec (and so the job id); SER's depend on the dataset."""
    cfg = {"experiment": "t", "orders": 1, "datasets": [
        {"name": n, "lr": 0.1, "batch_size": 8, "epochs": 1, "buffer_size": 5} for n in ("splitmnist", "cifar100")],
        "methods": [{"name": "ser"}, {"name": "ewc", "lam": 7}]}
    specs = {(j["dataset"]["name"], j["method"]["name"]): j["method"] for j in plan_jobs(cfg, {})}
    assert specs["splitmnist", "ser"]["alpha"] == 0.2 and specs["cifar100", "ser"]["alpha"] == 0.5
    assert specs["splitmnist", "ewc"] == {"name": "ewc", "lam": 7, "fisher_samples": 512}


def test_metrics_hand_made_matrix() -> None:
    """ACC, forgetting and BWT on a 3x3 matrix with order (2, 0, 1); columns are task ids."""
    order = [2, 0, 1]
    matrix = [[0, 0, 90],   # after task 2
              [80, 0, 70],  # after task 0
              [60, 95, 50]]  # after task 1
    m = compute_metrics(matrix, order)
    assert np.isclose(m["acc"], (60 + 95 + 50) / 3)
    # task 2: max(90, 70) - 50 = 40; task 0: max(80) - 60 = 20
    assert np.isclose(m["forgetting"], 30.0)
    assert np.isclose(m["bwt"], ((50 - 90) + (60 - 80)) / 2)
    # HTCL group semantics: tasks 2 and 0 learned at row 1 (end of group {2, 0})
    h = compute_metrics(matrix, order, learned_pos={2: 1, 0: 1, 1: 2})
    assert np.isclose(h["forgetting"], ((70 - 50) + (80 - 60)) / 2)
    # A baseline's group basis for k = 2 is exactly HTCL-k2's definition on the same matrix.
    assert group_basis(matrix, order, [2])["2"] == {"forgetting": h["forgetting"], "bwt": h["bwt"]}


def test_diloco_defaults_give_the_mean() -> None:
    """With the default outer lr 1 and momentum 0, the DiLoCo step lands exactly on the candidates' mean."""
    from htcl.hierarchy import DEFAULTS
    w, mean = [torch.randn(3, 2)], [torch.randn(3, 2)]
    new, _ = diloco_step(w, mean, [torch.zeros(3, 2)], DEFAULTS["diloco_lr"], DEFAULTS["diloco_momentum"])
    assert torch.allclose(new[0], mean[0])


def test_no_buffer_methods_and_replay_label() -> None:
    """NO_BUFFER_METHODS lists exactly the methods without a replay buffer; their HTCL rows say +replay=N."""
    from htcl.engine import load_data, make_method
    from htcl.hierarchy import NO_BUFFER_METHODS
    names = ["finetune", "er", "ser", "der", "ewc", "sr", "icarl", "dualnet"]
    ds = {"name": "splitmnist", "samples_per_class": 6, "lr": 0.01, "batch_size": 8, "epochs": 1, "buffer_size": 10}
    cfg = {"experiment": "t", "orders": 1, "datasets": [ds],
           "methods": [{"name": n} for n in names] + [{"name": "ewc", "htcl": {}}, {"name": "er", "htcl": {}}]}
    jobs = plan_jobs(cfg, {})
    data = load_data(jobs[0]["dataset"])
    for job in jobs[:len(names)]:
        method = make_method(job, data, torch.device("cpu"))
        assert (getattr(method, "buffer", None) is None) == (job["method"]["name"] in NO_BUFFER_METHODS)
    assert "+replay=10" in jobs[-2]["method_label"] and "+replay" not in jobs[-1]["method_label"]


def test_columns_by_task_id_under_permuted_order() -> None:
    """Row l of the matrix is indexed by task id: a predictor that only knows the learned tasks scores
    100 on exactly those columns, whatever their position in the order."""
    num_tasks, order = 4, [2, 0, 3, 1]
    # Task t's test inputs are the value t; its labels are all t (class scenario, head of size 4).
    tasks = [SimpleNamespace(test=[(torch.tensor([float(t)]), t)] * 3) for t in range(num_tasks)]
    data = SimpleNamespace(tasks=tasks)
    learned: set[int] = set()

    def predict(x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(len(x), num_tasks)
        for i, t in enumerate(x[:, 0].long().tolist()):
            logits[i, t if t in learned else (t + 1) % num_tasks] = 1
        return logits

    rows = []
    for task_id in order:
        learned.add(task_id)
        rows.append(evaluate(predict, data, "test", torch.device("cpu")))
    for pos, row in enumerate(rows):
        assert row == [100.0 if t in order[:pos + 1] else 0.0 for t in range(num_tasks)]


def test_resume_equals_uninterrupted(tmp_path) -> None:
    """A tiny ER job on CPU interrupted after task 2 and resumed ends with the same metrics."""
    cfg = {"experiment": "resume_test", "orders": 1,
           "datasets": [{"name": "splitmnist", "samples_per_class": 6, "lr": 0.01, "batch_size": 8, "epochs": 1,
                         "buffer_size": 10}],
           "methods": [{"name": "er"}]}
    (job,) = plan_jobs(cfg, {})
    cpu = torch.device("cpu")
    assert run_job(job, tmp_path / "full", cpu)
    assert not run_job(job, tmp_path / "resumed", cpu, max_steps=2)
    assert (tmp_path / "resumed" / "checkpoint.pt").exists()
    assert run_job(job, tmp_path / "resumed", cpu)
    full = json.loads((tmp_path / "full" / "acc_matrix.json").read_text())
    resumed = json.loads((tmp_path / "resumed" / "acc_matrix.json").read_text())
    assert full == resumed
    assert (tmp_path / "resumed" / "DONE").exists()


def test_diverged_job_fails_loudly(tmp_path) -> None:
    """With an absurd lr the weights blow up; the job must raise instead of writing chance-level metrics."""
    import pytest
    cfg = {"experiment": "nan_test", "orders": 1,
           "datasets": [{"name": "splitmnist", "samples_per_class": 6, "lr": 1e30, "batch_size": 8, "epochs": 1,
                         "buffer_size": 10}],
           "methods": [{"name": "finetune"}]}
    (job,) = plan_jobs(cfg, {})
    with pytest.raises(FloatingPointError):
        run_job(job, tmp_path, torch.device("cpu"))
    assert not (tmp_path / "DONE").exists()


def test_make_loader_drops_incomplete_trailing_batch() -> None:
    """Shuffled loaders with more than one batch drop the partial last batch; single-batch and eval loaders keep all."""
    from torch.utils.data import TensorDataset
    from htcl.data import make_loader
    data = lambda n: TensorDataset(torch.arange(n))  # noqa: E731
    assert [len(b[0]) for b in make_loader(data(66), 64, shuffle=True)] == [64]
    assert [len(b[0]) for b in make_loader(data(66), 64, shuffle=False)] == [64, 2]
    assert [len(b[0]) for b in make_loader(data(40), 64, shuffle=True)] == [40]
    assert [len(b[0]) for b in make_loader(data(40), "full", shuffle=True)] == [40]


def test_htcl_ewc_own_buffer_mid_rows_and_resume(tmp_path) -> None:
    """Group schedule: EWC has no replay buffer, so HTCL keeps its own (filled after each group, saved for resume);
    mid-group rows repeat the previous group's top row; a resumed job matches an uninterrupted one."""
    from htcl.engine import load_data, make_learner
    cfg = {"experiment": "htcl_ewc_test", "orders": 1,
           "datasets": [{"name": "splitmnist", "samples_per_class": 6, "lr": 0.01, "batch_size": 8, "epochs": 1,
                         "buffer_size": 10}],
           "methods": [{"name": "ewc", "htcl": {"k": 2, "curvature_samples": 8, "schedule": "group"}}]}
    (job,) = plan_jobs(cfg, {})
    cpu = torch.device("cpu")
    torch.manual_seed(0)
    learner = make_learner(job, load_data(job["dataset"]), cpu)
    assert learner.buffer is not None and len(learner.buffer) == 0
    learner.learn(job["order"][:2])
    assert len(learner.buffer) == 10 and set(learner.buffer.task.tolist()) <= set(job["order"][:2])
    # First group: w_1 gets recalibrated BN statistics, not the local model's training EMA.
    level, local = learner.levels[0].state_dict(), learner.method.model.state_dict()
    assert any(not torch.equal(level[k], local[k]) for k in level if k.endswith("running_var"))

    assert run_job(job, tmp_path / "full", cpu)
    assert not run_job(job, tmp_path / "resumed", cpu, max_steps=1)
    assert run_job(job, tmp_path / "resumed", cpu)
    full = json.loads((tmp_path / "full" / "acc_matrix.json").read_text())
    assert full == json.loads((tmp_path / "resumed" / "acc_matrix.json").read_text())
    rows = full["matrices"]["main"]
    assert rows[2] == rows[1]  # position 2 is inside group 2: deployed model = group 1's top model


def test_classes_per_task_prefix() -> None:
    """classes_per_task: shorter sequences use a prefix of the same class shuffle."""
    from htcl.data import _class_tasks
    short = _class_tasks("cifar100", 100, 5, 0, 5)
    long = _class_tasks("cifar100", 100, 20, 0, 5)
    assert all(len(t) == 5 for t in long) and len(long) == 20
    assert {c for t in short for c in t} <= {c for t in long[:5] for c in t} | set()
    assert short == long[:5]



def test_one_seed_for_every_job() -> None:
    """Every job of an experiment shares the paper seed; the seed is part of the job id."""
    from htcl.run import PAPER_SEED, plan_jobs
    ds = {"name": "splitmnist", "lr": 0.01, "batch_size": 32, "epochs": 1, "buffer_size": 50}
    cfg = {"experiment": "x", "orders": 3, "datasets": [ds], "methods": [{"name": "er"}, {"name": "ewc"}]}
    jobs = plan_jobs(cfg, {})
    assert {j["seed"] for j in jobs} == {PAPER_SEED} and len({j["job_id"] for j in jobs}) == len(jobs)
    other = plan_jobs({**cfg, "seed": PAPER_SEED + 1}, {})
    assert not {j["job_id"] for j in jobs} & {j["job_id"] for j in other}
