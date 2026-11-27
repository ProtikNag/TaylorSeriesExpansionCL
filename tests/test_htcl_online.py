"""Checks of HTCL's online schedule (DESIGN.md, HTCL): `.venv/bin/python -m pytest -q tests/test_htcl_online.py`."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from htcl.engine import load_data, make_learner, run_job
from htcl.run import plan_jobs

CPU = torch.device("cpu")
TINY = {"name": "splitmnist", "samples_per_class": 6, "lr": 0.01, "batch_size": 8, "epochs": 1, "buffer_size": 10}


def _job(htcl: dict, name: str = "er") -> dict:
    """The single planned job of a tiny SplitMNIST experiment with one order."""
    cfg = {"experiment": "online_test", "orders": 1, "datasets": [TINY],
           "methods": [{"name": name, "htcl": {"curvature_samples": 8, **htcl}}]}
    (job,) = plan_jobs(cfg, {})
    return job


def _states_equal(a, b) -> bool:
    """Recursive exact equality of nested state (tensors compared with torch.equal)."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_revision_never_worse_than_arrival_order() -> None:
    """Every revision adopts a candidate whose validation score is at least the arrival order's."""
    job = _job({"k": 3, "levels": 3})
    torch.manual_seed(job["seed"])
    learner = make_learner(job, load_data(job["dataset"]), CPU)
    for task_id in job["order"]:
        learner.learn([task_id])
    (rev,) = learner.revisions  # 5 tasks, k = 3: one full window, the last two are never revised
    assert len(rev["scores"]) == 6  # arrival + the 5 other orders of 3 tasks
    key = lambda order: ",".join(map(str, order))  # noqa: E731
    assert rev["scores"][key(rev["adopted"])] >= rev["scores"][key(rev["arrival"])]
    assert rev["scores"][key(rev["adopted"])] == max(rev["scores"].values())
    assert learner.window == job["order"][3:] and learner.snapshot is not None


def test_arrival_replay_reproduces_online_state() -> None:
    """Replaying the arrival order from the window snapshot with the seeding rule gives the online state
    bit for bit, so the arrival candidate never needs retraining. EWC also covers HTCL's own buffer."""
    for name in ("er", "ewc"):
        job = _job({"k": 3}, name)
        torch.manual_seed(job["seed"])
        learner = make_learner(job, load_data(job["dataset"]), CPU)
        learner.learn([job["order"][0]])  # first window: starts from no hierarchy
        learner.learn([job["order"][1]])
        online = copy.deepcopy(learner._core_state())
        learner._load_core(copy.deepcopy(learner.snapshot))
        for p in range(len(learner.window)):
            learner._online_task(learner.window[:p + 1])
        assert _states_equal(online, learner._core_state())


def test_k1_never_revises(tmp_path) -> None:
    """k = 1: no snapshot, no revision, one row per task, standard per-task learned positions."""
    job = _job({"k": 1})
    assert run_job(job, tmp_path, CPU)
    rows = json.loads((tmp_path / "acc_matrix.json").read_text())["matrices"]["main"]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(rows) == len(job["order"])
    assert metrics["htcl_revisions"] == [] and metrics["htcl_window_tasks"] == 1
    assert metrics["learned_pos"] == {str(t): p for p, t in enumerate(job["order"])}


def test_resume_mid_window(tmp_path) -> None:
    """A job killed after a mid-window step (snapshot in the checkpoint) resumes to the same matrix."""
    job = _job({"k": 3})
    assert run_job(job, tmp_path / "full", CPU)
    assert not run_job(job, tmp_path / "resumed", CPU, max_steps=2)
    ckpt = torch.load(tmp_path / "resumed" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["learner"]["window"] == job["order"][:2] and ckpt["learner"]["snapshot"] is not None
    assert run_job(job, tmp_path / "resumed", CPU)
    full = json.loads((tmp_path / "full" / "acc_matrix.json").read_text())
    assert full == json.loads((tmp_path / "resumed" / "acc_matrix.json").read_text())
    assert len(full["matrices"]["main"]) == len(job["order"])


def test_online_jobs_never_aliased() -> None:
    """Online rows depend on the arrival order: no aliasing. The group schedule still aliases."""
    cfg = {"experiment": "t", "orders": 6, "datasets": [{"name": "cora", "lr": 0.01, "batch_size": "full",
                                                          "epochs": 1, "buffer_size": 5}],
           "methods": [{"name": "er", "htcl": {}}, {"name": "er", "htcl": {"schedule": "group"}}]}
    jobs = plan_jobs(cfg, {})
    online = [j for j in jobs if j["method"]["htcl"]["schedule"] == "online"]
    group = [j for j in jobs if j["method"]["htcl"]["schedule"] == "group"]
    assert len(online) == 6 and not any(j.get("alias_of") for j in online)
    assert sum(1 for j in group if not j["alias_of"]) == 3


def test_group_schedule_invariant_inside_group(tmp_path) -> None:
    """Group schedule, search best: two orders with the same canonical form give identical matrices, which
    is what justifies aliasing them in run.plan_jobs."""
    job = _job({"k": 2, "schedule": "group"})
    results = []
    for order in ([0, 1, 2, 3, 4], [1, 0, 3, 2, 4]):
        assert run_job({**job, "order": order}, tmp_path / "".join(map(str, order)), CPU)
        results.append(json.loads((tmp_path / "".join(map(str, order)) / "acc_matrix.json").read_text())["matrices"])
    assert results[0] == results[1]


def test_lam_default_is_inverse_lr() -> None:
    """lam None resolves to 1/lr at planning time, so it is in the job spec and id; explicit lam is kept."""
    default, explicit = _job({}), _job({"lam": 5.0})
    assert default["method"]["htcl"]["lam"] == 1.0 / TINY["lr"] == 100.0
    assert explicit["method"]["htcl"]["lam"] == 5.0 and default["job_id"] != explicit["job_id"]


def test_catchup_rejected_online() -> None:
    """catchup belongs to the group schedule only."""
    with pytest.raises(ValueError, match="catchup"):
        _job({"catchup": {"iters": 1, "lr": 0.01}})
    assert _job({"catchup": {"iters": 1, "lr": 0.01}, "schedule": "group"})["method"]["htcl"]["catchup"]
