"""Sequential continual-learning engine: training, evaluation, accuracy matrix, checkpoint/resume.

A job runs a *learner* over the steps of its order. A step is one task (baselines, repeats, HTCL's
online schedule) or one group of k tasks (HTCL's group schedule). After every step the learner returns accuracy-matrix rows (one per task
in the step, columns indexed by task id) for each variant, and the engine writes an atomic
checkpoint with the learner state, the partial matrices and all RNG states.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .consolidation import average_states, recalibrate_bn
from .data import CLData, build_data, make_loader
from .methods import get_method
from .methods.base import Method
from .metrics import compute_metrics, group_basis, learned_positions
from .models import build_model
from .permutations import groups
from .utils import atomic_save, atomic_write_json, derive_seed, rng_state, seed_everything, set_rng_state, stable_hash

EVAL_BATCH = 256
DATA_KEYS = ("name", "scenario", "num_tasks", "classes_per_task", "samples_per_class", "test_samples_per_class", "seed", "data_root",
             "imagenet_root")
_DATA_CACHE: OrderedDict[str, CLData] = OrderedDict()


def load_data(spec: dict) -> CLData:
    """build_data with a small in-process cache keyed by the data-relevant part of the spec."""
    key = stable_hash({k: spec.get(k) for k in DATA_KEYS})
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = build_data(spec)
        while len(_DATA_CACHE) > 2:
            _DATA_CACHE.popitem(last=False)
    return _DATA_CACHE[key]


def train_task(method: Method, data: CLData, task_id: int, hp: dict, device: torch.device,
               replay: Dataset | None = None) -> None:
    """Train one task: begin_task, `epochs` passes of observe, end_task (epoch loop inside the task).

    `replay` (optional): a dataset interleaved into the task's training stream. HTCL passes its own
    reservoir here for buffer-free base methods (finetune/EWC/SR), so the local learner replays past
    tasks and is protected from forgetting them -- the replay protection the buffered methods get for
    free. Buffered methods (ER/SER/DER) replay their own buffer inside `observe`, so `replay` is None.
    """
    train = data.tasks[task_id].train
    source = ConcatDataset([train, replay]) if replay is not None and len(replay) > 0 else train
    loader = make_loader(source, hp["batch_size"], shuffle=True)
    method.model.train()
    method.begin_task(task_id, loader)
    for _ in range(int(hp["epochs"])):
        for x, y in loader:
            method.observe(x.to(device), y.to(device), task_id)
    method.end_task(task_id, loader)


@torch.no_grad()
def evaluate(predict: Callable[[torch.Tensor], torch.Tensor], data: CLData, split: str, device: torch.device,
             task_ids: list[int] | None = None) -> list[float]:
    """Accuracy (percent) of argmax(predict(x)) on each requested task's split, in task_ids order."""
    accs = []
    for t in task_ids if task_ids is not None else range(len(data.tasks)):
        correct = total = 0
        for x, y in make_loader(getattr(data.tasks[t], split), EVAL_BATCH, shuffle=False):
            correct += int((predict(x.to(device)).argmax(1).cpu() == y).sum())
            total += len(y)
        accs.append(100.0 * correct / max(total, 1))
    return accs


def evaluate_method(method: Method, data: CLData, split: str, device: torch.device,
                    task_ids: list[int] | None = None, weights: dict | None = None) -> list[float]:
    """Evaluate a method, optionally with other weights loaded temporarily into its model."""
    saved = copy.deepcopy(method.model.state_dict()) if weights is not None else None
    if weights is not None:
        method.model.load_state_dict(weights)
    was_training = method.model.training
    method.model.eval()
    method.prepare_eval()
    accs = evaluate(method.predict, data, split, device, task_ids)
    method.model.train(was_training)
    if saved is not None:
        method.model.load_state_dict(saved)
    return accs


def make_method(job: dict, data: CLData, device: torch.device) -> Method:
    """Instantiate the job's method on a fresh model."""
    spec = job["method"]
    cfg = {**job["dataset"], "num_outputs": data.num_outputs, "num_tasks": len(data.tasks),
           **{k: v for k, v in spec.items() if k not in ("name", "htcl", "repeats")}}
    return get_method(spec["name"])(build_model(data), cfg, device)


class Sequential:
    """Plain baseline: one task per step, the trained model is the deployed model."""

    def __init__(self, method: Method, data: CLData, hp: dict, device: torch.device):
        """Wrap a method."""
        self.method, self.data, self.hp, self.device = method, data, hp, device

    def learn(self, step: list[int]) -> dict[str, list[list[float]]]:
        """Train the step's task and evaluate on every task's test set."""
        train_task(self.method, self.data, step[0], self.hp, self.device)
        return {"main": [evaluate_method(self.method, self.data, "test", self.device)]}

    def finalize(self, metrics: dict) -> dict:
        """metrics.json content."""
        return metrics["main"]

    def final_model_state(self) -> dict:
        """Deployed weights."""
        return self.method.model.state_dict()

    def state_dict(self) -> dict:
        """Method state."""
        return {"method": self.method.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore method state."""
        self.method.load_state_dict(state["method"])


class Repeats:
    """R2 compute-matched control: n copies of the baseline from the same init, trained in lockstep.

    Repeat r trains task t with its own seed derive_seed(job seed, r, t), so the repeats differ only
    in their random streams. Variants: each repeat (r0..), weight average (wavg) and logit ensemble
    (ens); finalize adds best-on-validation (best_val) and the mean of the repeats' metrics (mean).
    """

    def __init__(self, method: Method, data: CLData, hp: dict, device: torch.device, n: int, seed: int):
        """n deep copies of a freshly initialised method."""
        self.methods = [method] + [copy.deepcopy(method) for _ in range(n - 1)]
        self.data, self.hp, self.device, self.seed = data, hp, device, seed
        self.wavg_state: dict | None = None  # weight average with recalibrated BN stats, after the last task

    def learn(self, step: list[int]) -> dict[str, list[list[float]]]:
        """Train every repeat on the task, then evaluate repeats, weight average and ensemble."""
        task_id, rows = step[0], {}
        for r, method in enumerate(self.methods):
            seed_everything(derive_seed(self.seed, r, task_id))
            train_task(method, self.data, task_id, self.hp, self.device)
            rows[f"r{r}"] = [evaluate_method(method, self.data, "test", self.device)]
        self.wavg_state = self._weight_average(task_id)
        rows["wavg"] = [evaluate_method(self.methods[0], self.data, "test", self.device, weights=self.wavg_state)]
        for m in self.methods:
            m.model.eval()
            m.prepare_eval()
        rows["ens"] = [evaluate(lambda x: torch.stack([m.predict(x) for m in self.methods]).mean(0),
                                self.data, "test", self.device)]
        for m in self.methods:
            m.model.train()
        return rows

    def _weight_average(self, task_id: int) -> dict:
        """Mean of the repeats' weights with BN stats recalibrated as HTCL does: shuffled train-mode passes
        over the current task's training data plus repeat 0's replay buffer (if any)."""
        from torch.utils.data import ConcatDataset
        from .hierarchy import BufferDataset  # lazy: hierarchy imports this module
        model = copy.deepcopy(self.methods[0].model)
        model.load_state_dict(average_states([m.model.state_dict() for m in self.methods]))
        parts = [self.data.tasks[task_id].train]
        buffer = getattr(self.methods[0], "buffer", None)
        if buffer is not None and len(buffer) > 0:
            parts.append(BufferDataset(buffer))
        recalibrate_bn(model, ConcatDataset(parts), self.device, self.hp["batch_size"])
        return model.state_dict()

    def finalize(self, metrics: dict) -> dict:
        """Pick the repeat with the best mean validation accuracy and average the repeats' metrics."""
        n = len(self.methods)
        val = [float(np.mean(evaluate_method(m, self.data, "val", self.device))) for m in self.methods]
        best = int(np.argmax(val))
        repeats = [metrics[f"r{r}"] for r in range(n)]
        mean = {k: float(np.mean([m[k] for m in repeats])) for k in ("acc", "forgetting", "bwt")}
        mean["final_per_task"] = np.mean([m["final_per_task"] for m in repeats], axis=0).tolist()
        mean["group_basis"] = {k: {key: float(np.mean([m["group_basis"][k][key] for m in repeats]))
                                   for key in ("forgetting", "bwt")} for k in repeats[0]["group_basis"]}
        variants = {"best_val": repeats[best], "mean": mean, "wavg": metrics["wavg"], "ens": metrics["ens"]}
        return {"variants": variants, "repeats": repeats, "best_repeat": best, "val_acc": val}

    def final_model_state(self) -> dict:
        """Weight average of the repeats, BN recalibrated (metrics.json names the best-on-validation repeat)."""
        return self.wavg_state

    def state_dict(self) -> dict:
        """All repeats' method states and the current weight average."""
        return {"methods": [m.state_dict() for m in self.methods], "wavg": self.wavg_state}

    def load_state_dict(self, state: dict) -> None:
        """Restore all repeats."""
        for m, s in zip(self.methods, state["methods"]):
            m.load_state_dict(s)
        self.wavg_state = state["wavg"]


def make_learner(job: dict, data: CLData, device: torch.device):
    """Sequential, Repeats or HTCL learner for a job."""
    method = make_method(job, data, device)
    spec = job["method"]
    if "htcl" in spec:
        from .hierarchy import HTCL
        return HTCL(method, data, job["dataset"], spec["htcl"], device, job["seed"])
    if spec.get("repeats"):
        return Repeats(method, data, job["dataset"], device, int(spec["repeats"]), job["seed"])
    return Sequential(method, data, job["dataset"], device)


def group_k(job: dict) -> int | None:
    """k of an HTCL job on the group schedule, else None (baselines, online HTCL: one task per step)."""
    htcl = job["method"].get("htcl")
    return htcl["k"] if htcl and htcl["schedule"] == "group" else None


def job_steps(job: dict) -> list[list[int]]:
    """Steps of the order: groups of k for the HTCL group schedule, otherwise single tasks."""
    return groups(job["order"], group_k(job) or 1)


def run_job(job: dict, out_dir: str | Path, device: torch.device, max_steps: int | None = None) -> bool:
    """Run (or resume) one job into out_dir. Returns True when the job is complete.

    `max_steps` stops after that many steps without finishing (used to test resume).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        return True
    atomic_write_json(job, out / "job.json")
    seed_everything(job["seed"])  # seeded at job start: independent of jobs run before in this process
    data = load_data(job["dataset"])
    learner = make_learner(job, data, device)
    steps = job_steps(job)
    rows: dict[str, list[list[float]]] = {}
    start, step_seconds = 0, []
    ckpt_path = out / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        learner.load_state_dict(ckpt["learner"])
        rows, start, step_seconds = ckpt["rows"], ckpt["next_step"], ckpt["step_seconds"]
        set_rng_state(ckpt["rng"])
    for s in range(start, len(steps)):
        if max_steps is not None and s - start >= max_steps:
            return False
        t0 = time.perf_counter()
        for variant, new_rows in learner.learn(steps[s]).items():
            rows.setdefault(variant, []).extend(new_rows)
        if device.type == "mps":
            torch.mps.synchronize()
        step_seconds.append(time.perf_counter() - t0)
        # A diverged model still "evaluates" (argmax of NaN logits is class 0), so fail loudly instead of
        # writing chance-level metrics. The previous step's checkpoint is kept.
        if not all(torch.isfinite(v).all() for v in learner.final_model_state().values() if v.is_floating_point()):
            raise FloatingPointError(f"non-finite weights after step {s} ({steps[s]}): training diverged")
        atomic_save({"learner": learner.state_dict(), "rows": rows, "next_step": s + 1,
                     "step_seconds": step_seconds, "rng": rng_state()}, ckpt_path)

    # A task counts as learned at the last position of its step (HTCL group schedule: end of its group).
    learned_pos = learned_positions(steps)
    metrics = {v: compute_metrics(m, job["order"], learned_pos) for v, m in rows.items()}
    # Common basis with the group schedule (DESIGN, Metrics): group HTCL-k reports its own k, every other
    # job (baselines, online HTCL: standard per-task rows) every k.
    ks = [group_k(job)] if group_k(job) else range(2, len(job["order"]))
    for v, m in rows.items():
        metrics[v]["group_basis"] = group_basis(m, job["order"], ks)
    result = learner.finalize(metrics)
    result["learned_pos"] = learned_pos
    atomic_write_json({"order": job["order"], "matrices": rows}, out / "acc_matrix.json")
    atomic_write_json(result, out / "metrics.json")
    atomic_write_json({"total_seconds": sum(step_seconds), "step_seconds": step_seconds, "device": str(device)},
                      out / "timing.json")
    atomic_save(learner.final_model_state(), out / "final_model.pt")
    (out / "DONE").touch()
    ckpt_path.unlink(missing_ok=True)  # the final state lives in final_model.pt; checkpoints are large
    return True
