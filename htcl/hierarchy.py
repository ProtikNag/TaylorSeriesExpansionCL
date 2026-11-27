"""HTCL learner: online consolidation with windowed order revision (default) or the group schedule.

Levels: `levels = L` counts the local model, so there are L-1 hierarchical models w_1..w_{L-1};
the deployed model is w_{L-1}. See docs/DESIGN.md, HTCL section.

`schedule: online` (default, one engine step per task): local <- w_1 (with the method state), train the
arriving task, cascade the Taylor step w_1 <- local, w_i <- w_{i-1}, recalibrate BN; methods without a
replay buffer add the task to HTCL's own N-sample reservoir. Every k tasks (a window) the learner restores
the window's start snapshot, replays every other order of the window and adopts the best one by mean
validation accuracy of the deployed model on the window's tasks (ties keep the arrival order). Every task's
local training is seeded from (job seed, window, position, task, window prefix), so replaying the arrival
order reproduces the online state exactly and is never retrained.

`schedule: group` (ablation, one engine step per group of k): 1. snapshot the method state; 2. train every
permutation of the group from w_1 and the snapshot, score by mean validation accuracy on the group (first
permutation in lexicographic order wins ties); 3. adopt the winner's model and method state; 4. first
group: every level = winner; 5. otherwise consolidate level by level (Taylor / DiLoCo / FedAvg / FedProx);
6. recalibrate BN (first group included, unless consolidate is false); 7. methods without a replay buffer
(finetune, EWC, SR): add the group's training data to HTCL's own N-sample reservoir.
"""

from __future__ import annotations

import copy
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

from . import consolidation as C
from .buffer import ReservoirBuffer
from .data import CLData, make_loader
from .engine import evaluate_method, train_task
from .methods.base import Method
from .utils import derive_seed, seed_everything

DEFAULTS = {
    "levels": 2,               # L, local model included: L2 = local + w_1
    "schedule": "online",      # online (consolidate every task, revise every k) | group (ablation)
    "k": 2,                    # revision window (online) or group size (group)
    "lam": None,               # lambda, scalar or one per hierarchical level; None = 1/lr of the dataset
    "curvature": "fisher",     # fisher | hutchinson | identity (H = 0, first-order rule)
    "curvature_samples": 512,
    "hutchinson_probes": 4,
    "aggregator": "taylor",    # taylor | diloco | fedavg | fedprox
    "search": "best",          # best | random
    "consolidate": True,       # False: deploy the selected local model
    "step_scale": 1.0,         # eta, scalar or one per level
    "clip_norm": None,         # global L2 cap on each Taylor update
    "catchup": None,           # {iters, lr}
    "fedprox_mu": 0.01,
    "diloco_lr": 1.0,          # with momentum 0: w_1 <- mean of the k! candidates (paper's DiLoCo variant)
    "diloco_momentum": 0.0,
}

# Base methods without a replay buffer: HTCL gives them its own N-sample reservoir for the consolidation
# data, so their HTCL rows carry a `+replay=N` label tag (run.plan_jobs); tests check this list.
NO_BUFFER_METHODS = ("finetune", "ewc", "sr")


def resolve(spec: dict, lr: float | None = None) -> dict:
    """HTCL spec with defaults filled in (unknown keys are rejected).

    lam None becomes 1/lr when the dataset lr is given: with H ~ 0 the update is Delta d - g/lam, so the
    curvature-free limit is then one SGD step from the local model.
    """
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown htcl keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **spec}
    if cfg["schedule"] not in ("online", "group"):
        raise ValueError(f"unknown htcl schedule {cfg['schedule']!r}")
    if cfg["schedule"] == "online" and cfg["catchup"] is not None:
        raise ValueError("catchup is not part of the online schedule (use schedule: group)")
    if cfg["lam"] is None and lr is not None:
        cfg["lam"] = 1.0 / float(lr)
    return cfg


def _per_level(value, level: int) -> float:
    """Scalar or per-level list value."""
    return float(value[level] if isinstance(value, (list, tuple)) else value)


class BufferDataset(Dataset):
    """The replay buffer's (x, y) as a dataset."""

    def __init__(self, buffer):
        """Wrap a ReservoirBuffer (read-only)."""
        self.buffer = buffer

    def __len__(self) -> int:
        """Stored samples."""
        return len(self.buffer)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """(x, label)."""
        return self.buffer.x[i], int(self.buffer.y[i])


class HTCL:
    """Hierarchical learner wrapping a base method (see module docstring)."""

    def __init__(self, method: Method, data: CLData, hp: dict, spec: dict, device: torch.device, seed: int):
        """method: the base CL method (its model is the local model); hp: dataset hyperparameters;
        seed: the job seed (online schedule: per-task training seeds derive from it)."""
        self.method, self.data, self.hp, self.device, self.seed = method, data, hp, device, seed
        self.cfg = resolve(spec, hp.get("lr"))
        self.levels: list[nn.Module] = []
        self.momentum: list[list[torch.Tensor]] = []  # DiLoCo outer momentum per level
        self.seen_samples = 0                           # FedAvg weight of the levels
        self.seen_tasks: list[int] = []
        # HTCL's own N-sample reservoir for methods without a replay buffer (EWC, SR, finetune), so g and
        # H always cover earlier groups; replay methods keep using theirs, so the memory budget stays N.
        self.buffer = None if getattr(method, "buffer", None) is not None else ReservoirBuffer(hp["buffer_size"])
        self.last_top_row: list[float] | None = None  # deployed model's test row after the previous group
        # Online schedule: tasks of the open window, its index, the snapshot taken when it opened, and a
        # log of every revision (scores of the candidates and the adopted order), written to metrics.json.
        self.window: list[int] = []
        self.window_index = 0
        self.snapshot: dict | None = None
        self.revisions: list[dict] = []

    def learn(self, step: list[int]) -> dict[str, list[list[float]]]:
        """One engine step: a task (online schedule) or a group (group schedule); one row per task."""
        if self.cfg["schedule"] == "group":
            return self._learn_group(step)
        (task_id,) = step
        k = int(self.cfg["k"])
        if not self.window and k > 1:  # the window opens: everything a revision may roll back to
            self.snapshot = copy.deepcopy(self._core_state())
        self.window.append(task_id)
        self._online_task(self.window)
        if len(self.window) == k:
            if k > 1:
                self._revise()
            self.window, self.snapshot = [], None
            self.window_index += 1
        return {"main": [evaluate_method(self.method, self.data, "test", self.device,
                                         weights=self.levels[-1].state_dict())]}

    # ---------------------------------------------------------- online schedule
    def _online_task(self, prefix: list[int]) -> None:
        """Steps 2-4 for the last task of `prefix` (the window order up to and including it): seed, local
        <- w_1, train, cascade, own reservoir. The seed depends only on the window and its prefix, so a
        revision replaying the arrival order from the snapshot reproduces the online state exactly."""
        task_id = prefix[-1]
        seed_everything(derive_seed(self.seed, self.window_index, len(prefix) - 1, task_id, list(prefix)))
        if self.levels:
            self.method.model.load_state_dict(self.levels[0].state_dict())
            self.method.rebase()  # weight anchors (EWC) move to w_1 as well
        if self.cfg["aggregator"] == "fedprox":
            self.method.prox = (self.cfg["fedprox_mu"], [p.detach().clone() for p in C.trainable(self.method.model)])
        train_task(self.method, self.data, task_id, self.hp, self.device)
        self.method.prox = None
        self.seen_tasks = sorted(self.seen_tasks + [task_id])
        # DiLoCo with one local candidate per step: its mean is the local model itself.
        candidates = [copy.deepcopy(self.method.model.state_dict())] if self.cfg["aggregator"] == "diloco" else []
        self._consolidate([task_id], candidates)
        if self.buffer is not None:  # after consolidation, so the buffer holds earlier tasks only
            for x, y in make_loader(self.data.tasks[task_id].train, self.hp["batch_size"], shuffle=False):
                self.buffer.add(x, y, task_id)

    def _window_score(self, tasks: list[int]) -> float:
        """Mean validation accuracy of the deployed model w_{L-1} on the window's tasks."""
        return float(np.mean(evaluate_method(self.method, self.data, "val", self.device, list(tasks),
                                             weights=self.levels[-1].state_dict())))

    def _revise(self) -> None:
        """Replay the closed window in every other order from its snapshot and adopt the best state
        (ties keep the arrival order, then the first in permutations order). search: random adopts one
        uniformly drawn order instead."""
        arrival = tuple(self.window)
        online = copy.deepcopy(self._core_state())
        scores = {arrival: self._window_score(arrival)}
        orders = list(itertools.permutations(arrival))  # orders[0] is the arrival order
        if self.cfg["search"] == "random":
            rng = random.Random(derive_seed(self.seed, self.window_index, "random", list(arrival)))  # varies with the order
            orders = [orders[rng.randrange(len(orders))]]
        else:
            orders = orders[1:]
        best, best_state = arrival, online
        for sigma in orders:
            if sigma == arrival:  # random draw of the arrival order: the online state is that candidate
                continue
            self._load_core(copy.deepcopy(self.snapshot))
            for p in range(len(sigma)):
                self._online_task(list(sigma[:p + 1]))
            scores[sigma] = self._window_score(arrival)
            if self.cfg["search"] == "random" or scores[sigma] > scores[best]:
                best, best_state = sigma, copy.deepcopy(self._core_state())
        self._load_core(best_state)
        self.revisions.append({"window": self.window_index, "arrival": list(arrival), "adopted": list(best),
                               "scores": {",".join(map(str, s)): v for s, v in scores.items()}})

    # ----------------------------------------------------------- group schedule
    def _learn_group(self, group: list[int]) -> dict[str, list[list[float]]]:
        """Process one group; returns one accuracy row per task position in the group."""
        cfg = self.cfg
        snapshot = copy.deepcopy(self.method.state_dict())
        start = copy.deepcopy(self.levels[0].state_dict()) if self.levels else None
        sigmas = list(itertools.permutations(sorted(group)))
        if cfg["search"] == "random":
            sigmas = [sigmas[int(torch.randint(len(sigmas), (1,)))]]
        best_score, best_state, best_sigma, candidates = -np.inf, None, None, []
        for sigma in sigmas:
            self.method.load_state_dict(copy.deepcopy(snapshot))
            if start is not None:
                self.method.model.load_state_dict(start)
                self.method.rebase()  # TrainSeq(sigma, w_1): weight anchors move to w_1 as well
            if cfg["aggregator"] == "fedprox":
                self.method.prox = (cfg["fedprox_mu"], [p.detach().clone() for p in C.trainable(self.method.model)])
            for task_id in sigma:
                train_task(self.method, self.data, task_id, self.hp, self.device)
            self.method.prox = None
            score = float(np.mean(evaluate_method(self.method, self.data, "val", self.device, list(group))))
            if score > best_score:  # strict: the first permutation wins ties
                best_score, best_state, best_sigma = score, copy.deepcopy(self.method.state_dict()), sigma
            if cfg["aggregator"] == "diloco":
                candidates.append(copy.deepcopy(self.method.model.state_dict()))
        self.method.load_state_dict(best_state)

        self.seen_tasks = sorted(self.seen_tasks + list(group))
        # Canonical (sorted) group: the consolidation data's block order drives BN recalibration's shuffle and
        # the Fisher subset, so the arrival order inside a group must not reach it (canonical aliasing).
        self._consolidate(sorted(group), candidates)
        if self.buffer is not None:  # after consolidation, so the buffer holds earlier groups only
            for task_id in best_sigma:  # stream order of the adopted run
                for x, y in make_loader(self.data.tasks[task_id].train, self.hp["batch_size"], shuffle=False):
                    self.buffer.add(x, y, task_id)
        top_row = evaluate_method(self.method, self.data, "test", self.device, weights=self.levels[-1].state_dict())
        # Within a group the deployed model is the previous group's, so its mid rows are that group's
        # recorded top row (re-evaluating would pair the old weights with the new method state, e.g.
        # iCaRL's exemplars). The first group deploys its own result.
        mid_row = top_row if self.last_top_row is None else self.last_top_row
        self.last_top_row = top_row
        return {"main": [mid_row] * (len(group) - 1) + [top_row]}

    # ----------------------------------------------------------- consolidation
    def _consolidate(self, group: list[int], candidates: list[dict]) -> None:
        """Update w_1..w_{L-1} from the adopted local model."""
        cfg, local = self.cfg, self.method.model
        n_new = sum(len(self.data.tasks[t].train) for t in group)
        if not self.levels or not cfg["consolidate"]:
            first = copy.deepcopy(local)
            if cfg["consolidate"]:  # w_1^(1) gets the same BN treatment as every later level (step 6)
                C.recalibrate_bn(first, self._consolidation_data(group), self.device, self.hp["batch_size"])
            self.levels = [copy.deepcopy(first) for _ in range(cfg["levels"] - 1)]
            self.momentum = [[torch.zeros_like(p) for p in C.trainable(local)] for _ in self.levels]
            self.seen_samples += n_new
            return
        data = self._consolidation_data(group)
        target = local
        if cfg["aggregator"] == "diloco":  # target = mean of all k! local candidates
            target = copy.deepcopy(local)
            target.load_state_dict(C.average_states(candidates))
        for i, level in enumerate(self.levels):
            if cfg["aggregator"] == "taylor":
                self._taylor(level, target, data, i)
                if i == 0 and cfg["catchup"]:
                    self._catchup(level, data)
            elif cfg["aggregator"] == "diloco":
                new, self.momentum[i] = C.diloco_step(
                    [p.detach() for p in C.trainable(level)], [p.detach() for p in C.trainable(target)],
                    self.momentum[i], cfg["diloco_lr"], cfg["diloco_momentum"])
                C.assign(level, new)
            elif cfg["aggregator"] in ("fedavg", "fedprox"):
                from .methods.federated import fedavg_update
                fedavg_update(level, target, self.seen_samples, n_new)
            else:
                raise ValueError(f"unknown aggregator {cfg['aggregator']}")
            C.recalibrate_bn(level, data, self.device, self.hp["batch_size"])
            target = level  # Delta d_i = w_{i-1}^{(t)} - w_i^{(t-1)} for i > 1
        self.seen_samples += n_new

    def _consolidation_data(self, group: list[int]) -> Dataset:
        """Current group's training data plus the replay buffer (the method's, else HTCL's own)."""
        parts: list[Dataset] = [self.data.tasks[t].train for t in group]
        buffer = self.buffer if self.buffer is not None else self.method.buffer
        if len(buffer) > 0:
            parts.append(BufferDataset(buffer))
        return ConcatDataset(parts)

    def _taylor(self, level: nn.Module, target: nn.Module, data: Dataset, i: int) -> None:
        """eq. hier_update at level i: g and diagonal H at w_i^{(t-1)}, then the closed-form step."""
        cfg = self.cfg
        g = C.mean_gradient(level, data, self.device, self.hp["batch_size"])
        if cfg["curvature"] == "fisher":
            h = C.fisher_diag(level, data, self.device, cfg["curvature_samples"])
        elif cfg["curvature"] == "hutchinson":
            h = C.hutchinson_diag(level, data, self.device, cfg["curvature_samples"], self.hp["batch_size"],
                                  cfg["hutchinson_probes"])
        elif cfg["curvature"] == "identity":
            h = [torch.zeros_like(p) for p in C.trainable(level)]
        else:
            raise ValueError(f"unknown curvature {cfg['curvature']}")
        new = C.taylor_step([p.detach() for p in C.trainable(level)], [p.detach() for p in C.trainable(target)],
                            g, h, _per_level(cfg["lam"], i), _per_level(cfg["step_scale"], i), cfg["clip_norm"])
        C.assign(level, new)

    def _catchup(self, level: nn.Module, data: Dataset) -> None:
        """Optional (off by default): `iters` rounds of one SGD epoch on the consolidation data followed
        by a Taylor step toward it; keep the best iterate by validation accuracy on all seen tasks."""
        iters, lr = int(self.cfg["catchup"]["iters"]), float(self.cfg["catchup"]["lr"])

        def score(state: dict) -> float:
            return float(np.mean(evaluate_method(self.method, self.data, "val", self.device, self.seen_tasks,
                                                 weights=state)))

        C.recalibrate_bn(level, data, self.device, self.hp["batch_size"])  # score the Taylor iterate fairly
        best_state = copy.deepcopy(level.state_dict())
        best = score(best_state)
        for _ in range(iters):
            sgd = copy.deepcopy(level)
            opt = torch.optim.SGD(sgd.parameters(), lr=lr, momentum=self.hp.get("momentum", 0.9),
                                  weight_decay=self.hp.get("weight_decay", 1e-4))
            sgd.train()
            for x, y in make_loader(data, self.hp["batch_size"], shuffle=True):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(sgd(x.to(self.device)), y.to(self.device)).backward()
                opt.step()
            self._taylor(level, sgd, data, 0)
            C.recalibrate_bn(level, data, self.device, self.hp["batch_size"])
            current = score(level.state_dict())
            if current > best:
                best, best_state = current, copy.deepcopy(level.state_dict())
        level.load_state_dict(best_state)

    # ------------------------------------------------------------- engine API
    def finalize(self, metrics: dict) -> dict:
        """metrics.json content. htcl_window_tasks: past tasks' training data the local learner holds during
        a window or group (training-time memory only; the deployed replay buffer stays N)."""
        return {**metrics["main"], "htcl_window_tasks": int(self.cfg["k"]), "htcl_revisions": self.revisions}

    def final_model_state(self) -> dict:
        """Deployed model w_{L-1}."""
        return self.levels[-1].state_dict()

    def _core_state(self) -> dict:
        """Learner state a revision rolls back: method state, levels, DiLoCo momentum, FedAvg counter,
        seen tasks, own buffer. References live tensors: deep-copy before keeping it."""
        return {"method": self.method.state_dict(), "levels": [lv.state_dict() for lv in self.levels],
                "momentum": self.momentum, "seen_samples": self.seen_samples, "seen_tasks": self.seen_tasks,
                "buffer": self.buffer.state_dict() if self.buffer is not None else None}

    def state_dict(self) -> dict:
        """Core state plus the group schedule's last top row and the online window (snapshot included)."""
        return {**self._core_state(), "last_top_row": self.last_top_row, "window": self.window,
                "window_index": self.window_index, "snapshot": self.snapshot, "revisions": self.revisions}

    def load_state_dict(self, state: dict) -> None:
        """Restore (tensors arrive on the CPU)."""
        self._load_core(state)
        self.last_top_row = state["last_top_row"]
        self.window, self.window_index = list(state["window"]), state["window_index"]
        self.snapshot, self.revisions = state["snapshot"], list(state["revisions"])

    def _load_core(self, state: dict) -> None:
        """Restore `_core_state` output (the buffers may alias `state`'s tensors: pass a copy to keep it)."""
        self.method.load_state_dict(state["method"])
        self.levels = []
        for level_state in state["levels"]:
            level = copy.deepcopy(self.method.model)
            level.load_state_dict(level_state)
            self.levels.append(level)
        self.momentum = [[m.to(self.device) for m in ms] for ms in state["momentum"]]
        self.seen_samples = state["seen_samples"]
        self.seen_tasks = list(state["seen_tasks"])
        if self.buffer is not None:
            self.buffer.load_state_dict(state["buffer"])
