"""HTCL learner: the group schedule (the algorithm the paper describes), with the online schedule as an
ablation. See docs/ALGORITHM.md for the algorithm and why, and docs/DESIGN.md for the code contract.

Levels: `levels = L` counts the local model, so there are L-1 hierarchical models w_1..w_{L-1};
the deployed model is w_{L-1}. The paper uses L=2 (local + w_1), k=2.

`schedule: group` (default, THE algorithm; one engine step per group of k tasks):
  1. snapshot the method state; 2. train every within-group permutation from w_1 and the snapshot, score by
     mean validation accuracy on the group, adopt the best (first permutation wins ties) -- this gives the
     within-window order invariance; 3. first group: every level = the local; otherwise merge level by level
     with the closed-form Taylor step, line-searched over alpha in {0,.25,.5,.75,1} plus the local itself and
     kept on seen-task validation (`merge_search`); 4. a final group shorter than k is merged the same way
     (`merge_last`); 5. BN recalibration always sees the replay buffer, but the Taylor *gradient* for a
     buffer-free method (finetune/EWC/SR) is the group only -- its injected N-sample reservoir protects the
     local window learner (step 2), not the deployed model.
  Ablations: `consolidate: false` deploys the searched local with no merge (V1); `merge_search: false,
  guard: true` is the single-jump all-or-nothing merge (V2).

`schedule: online` (ablation, one engine step per task): local <- w_1, train the arriving task, cascade the
Taylor step, recalibrate BN; every k tasks (a window) replay every other order from the window's start
snapshot and adopt the best by validation. Used only in the R1 schedule-comparison ablation.
"""

from __future__ import annotations

import copy
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset

from . import consolidation as C
from .buffer import ReservoirBuffer
from .data import CLData, make_loader
from .engine import evaluate_method, train_task
from .methods.base import Method
from .utils import derive_seed, seed_everything

DEFAULTS = {
    "levels": 2,               # L, local model included: L2 = local + w_1
    "schedule": "group",       # group (THE algorithm: a group of k tasks per step) | online (R1 ablation)
    "k": 2,                    # group size (group schedule) or revision window (online ablation)
    "lam": None,               # lambda, scalar or one per hierarchical level; None = 1/lr of the dataset
    "curvature": "fisher",     # fisher | hutchinson | identity (H = 0, first-order rule)
    "curvature_samples": 512,
    "hutchinson_probes": 4,
    "aggregator": "taylor",    # taylor | diloco | fedavg | fedprox
    "search": "best",          # best | random
    "consolidate": True,       # False: deploy the selected local model (ablation V1)
    "guard": False,            # True: all-or-nothing guard instead of the step-size search (ablation V2)
    "merge_search": True,      # THE algorithm: treat the Taylor step as a direction and line-search alpha
    "merge_grid": [0.0, 0.25, 0.5, 0.75, 1.0],  # alpha grid for the step-size search (plus the local model)
    "merge_last": True,        # THE algorithm: merge a final group shorter than k too (via the step-size search)
    "step_scale": 1.0,         # eta, scalar or one per level
    "clip_norm": None,         # global L2 cap on each Taylor update
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
    if cfg["schedule"] not in ("group", "online"):
        raise ValueError(f"unknown htcl schedule {cfg['schedule']!r}")
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
        self.guard_log: list[dict] = []  # per guarded consolidation: did the Taylor step beat the base model?

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
        # Buffer-free base methods (finetune/EWC/SR) replay HTCL's own reservoir during local training, so
        # the local learner is protected from forgetting past groups -- the protection ER/SER/DER get for free.
        replay = BufferDataset(self.buffer) if self.buffer is not None and len(self.buffer) > 0 else None
        best_score, best_state, best_sigma, candidates = -np.inf, None, None, []
        for sigma in sigmas:
            self.method.load_state_dict(copy.deepcopy(snapshot))
            if start is not None:
                self.method.model.load_state_dict(start)
                self.method.rebase()  # TrainSeq(sigma, w_1): weight anchors move to w_1 as well
            if cfg["aggregator"] == "fedprox":
                self.method.prox = (cfg["fedprox_mu"], [p.detach().clone() for p in C.trainable(self.method.model)])
            for task_id in sigma:
                train_task(self.method, self.data, task_id, self.hp, self.device, replay=replay)
            self.method.prox = None
            score = float(np.mean(evaluate_method(self.method, self.data, "val", self.device, list(group))))
            if score > best_score:  # strict: the first permutation wins ties
                best_score, best_state, best_sigma = score, copy.deepcopy(self.method.state_dict()), sigma
            if cfg["aggregator"] == "diloco":
                candidates.append(copy.deepcopy(self.method.model.state_dict()))
        self.method.load_state_dict(best_state)

        self.seen_tasks = sorted(self.seen_tasks + list(group))
        # A final group of fewer than k tasks is never consolidated (the consolidation step on a lone task
        # overshoots and wipes earlier tasks); instead the base-trained local -- which replay keeps healthy
        # -- is deployed directly. Only a full group of k tasks is consolidated into the hierarchy.
        if self.levels and len(group) < int(cfg["k"]) and not cfg["merge_last"]:
            self.levels = [copy.deepcopy(self.method.model) for _ in self.levels]
        else:
            # Canonical (sorted) group: the consolidation data's block order drives BN recalibration's shuffle
            # and the Fisher subset, so the arrival order inside a group must not reach it (canonical aliasing).
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
                C.recalibrate_bn(first, self._bn_data(group), self.device, self.hp["batch_size"])
            self.levels = [copy.deepcopy(first) for _ in range(cfg["levels"] - 1)]
            self.momentum = [[torch.zeros_like(p) for p in C.trainable(local)] for _ in self.levels]
            self.seen_samples += n_new
            return
        data = self._consolidation_data(group)   # gradient data: group + method's own buffer only
        bn = self._bn_data(group)                # BN-stat data: group + any buffer (reservoir included)
        target = local
        if cfg["aggregator"] == "diloco":  # target = mean of all k! local candidates
            target = copy.deepcopy(local)
            target.load_state_dict(C.average_states(candidates))
        # merge_search: snapshot each level's weights before the step, so the cascade result becomes a
        # direction we line-search along (old + alpha * (stepped - old)).
        old_params = ([[p.detach().clone() for p in C.trainable(lv)] for lv in self.levels]
                      if cfg["merge_search"] and cfg["aggregator"] == "taylor" else None)
        for i, level in enumerate(self.levels):
            if cfg["aggregator"] == "taylor":
                self._taylor(level, target, data, i)
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
            C.recalibrate_bn(level, bn, self.device, self.hp["batch_size"])  # BN sees the buffer
            target = level  # Delta d_i = w_{i-1}^{(t)} - w_i^{(t-1)} for i > 1
        if old_params is not None:
            self._merge_search(old_params, local, bn)
        elif cfg["guard"] and cfg["aggregator"] == "taylor":
            self._guard_consolidation(local, bn)
        self.seen_samples += n_new

    def _guard_consolidation(self, local: nn.Module, bn_data: Dataset) -> None:
        """Non-destructive consolidation: if the cascade left the deployed model worse on seen-task
        validation than the just-trained local model, reject it and adopt the local model at every
        level instead (BN recalibrated). This bounds HTCL below by the base learner it wraps, so a
        single overshooting Taylor step cannot wipe earlier tasks (the C7 "never worse" property)."""
        def val(state: dict) -> float:
            return float(np.mean(evaluate_method(self.method, self.data, "val", self.device,
                                                 self.seen_tasks, weights=state)))
        local_score, cons_score = val(local.state_dict()), val(self.levels[-1].state_dict())
        took_step = cons_score >= local_score  # keep the Taylor step only when it is not worse
        self.guard_log.append({"window": self.window_index, "n_seen": len(self.seen_tasks),
                               "took_step": took_step, "cons_val": cons_score, "local_val": local_score})
        if took_step:
            return
        kept = copy.deepcopy(local)  # reject: adopt the base learner as every level
        C.recalibrate_bn(kept, bn_data, self.device, self.hp["batch_size"])
        self.levels = [copy.deepcopy(kept) for _ in self.levels]
        self.momentum = [[torch.zeros_like(p) for p in C.trainable(local)] for _ in self.levels]

    def _merge_search(self, old_params: list[list[torch.Tensor]], local: nn.Module, bn_data: Dataset) -> None:
        """Step-size search for the merge. The cascade (already applied, held in self.levels) is a
        direction: candidate levels are old_i + alpha (stepped_i - old_i) for alpha in merge_grid, plus
        the local model itself at every level. Each candidate's BatchNorm is recalibrated, then its
        deployed model (levels[-1]) is scored on seen-task validation. The best wins; ties go to the
        larger alpha (the local candidate ranks below alpha=0). With merge_grid=[1.0] the alpha=1
        candidate is exactly the single Taylor jump. The chosen alpha (or "local") is logged per merge."""
        stepped = [[p.detach().clone() for p in C.trainable(lv)] for lv in self.levels]

        def deployed_val(levels: list[nn.Module]) -> float:
            return float(np.mean(evaluate_method(self.method, self.data, "val", self.device,
                                                 self.seen_tasks, weights=levels[-1].state_dict())))

        def recal(m: nn.Module) -> nn.Module:
            C.recalibrate_bn(m, bn_data, self.device, self.hp["batch_size"])
            return m

        cands = []  # (rank, label, [level modules])
        for alpha in self.cfg["merge_grid"]:
            a = float(alpha)
            levels = []
            for lv, old_p, st_p in zip(self.levels, old_params, stepped):
                m = copy.deepcopy(lv)
                C.assign(m, [o * (1.0 - a) + s * a for o, s in zip(old_p, st_p)])
                levels.append(recal(m))
            cands.append((a, a, levels))
        cands.append((-1.0, "local", [recal(copy.deepcopy(local)) for _ in self.levels]))  # local ranks lowest

        scored = [(deployed_val(levels), rank, label, levels) for rank, label, levels in cands]
        _, _, best_label, best_levels = max(scored, key=lambda t: (t[0], t[1]))  # tie -> larger alpha
        self.levels = [copy.deepcopy(m) for m in best_levels]
        self.momentum = [[torch.zeros_like(p) for p in C.trainable(local)] for _ in self.levels]
        self.guard_log.append({"window": self.window_index, "n_seen": len(self.seen_tasks), "alpha": best_label,
                               "scores": {str(lab): round(sc, 4) for sc, _, lab, _ in scored}})

    def _consolidation_data(self, group: list[int]) -> Dataset:
        """Current group's training data for the GLOBAL consolidation (Taylor g/H and BN recalibration),
        plus the method's OWN replay buffer (ER/SER/DER). HTCL's injected N-sample reservoir for
        buffer-free methods (finetune/EWC/SR) is a local-window-only device: it protects the local learner
        (see `_learn_group`), so the global's Taylor step takes no replay-gradient of its own -- it reaches
        the reservoir's effect only through the merge toward that local (`_merge_search`)."""
        parts: list[Dataset] = [self.data.tasks[t].train for t in group]
        buffer = getattr(self.method, "buffer", None)  # method's own buffer only; not HTCL's reservoir
        if buffer is not None and len(buffer) > 0:
            parts.append(BufferDataset(buffer))
        return ConcatDataset(parts)

    def _bn_data(self, group: list[int]) -> Dataset:
        """Data for BatchNorm recalibration of the consolidated/deployed model: the group plus ANY
        available replay -- the method's own buffer (ER/SER/DER), else HTCL's reservoir (finetune/EWC/SR).
        BN recalibration estimates deployment statistics over seen tasks, not gradient training, so the
        reservoir is included even for no-replay methods (whose Taylor gradient step, `_consolidation_data`,
        excludes it). Keeping the buffer here preserves earlier-task BN statistics in the deployed model."""
        parts: list[Dataset] = [self.data.tasks[t].train for t in group]
        buffer = self.buffer if self.buffer is not None else getattr(self.method, "buffer", None)
        if buffer is not None and len(buffer) > 0:
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

    # ------------------------------------------------------------- engine API
    def finalize(self, metrics: dict) -> dict:
        """metrics.json content. htcl_window_tasks: past tasks' training data the local learner holds during
        a window or group (training-time memory only; the deployed replay buffer stays N)."""
        out = {**metrics["main"], "htcl_window_tasks": int(self.cfg["k"]), "htcl_revisions": self.revisions}
        if self.cfg["merge_search"]:  # step-size search: which alpha (or the local model) each merge chose
            alphas = [g["alpha"] for g in self.guard_log if "alpha" in g]
            hist: dict[str, int] = {}
            for a in alphas:
                hist[str(a)] = hist.get(str(a), 0) + 1
            out["htcl_merge"] = {"merges": len(alphas), "alpha_hist": hist, "log": self.guard_log}
        elif self.cfg["guard"]:  # V2 ablation: how often the all-or-nothing Taylor step beat the local
            taken = sum(g["took_step"] for g in self.guard_log)
            n = len(self.guard_log)
            out["htcl_guard"] = {"steps": n, "taken": taken,
                                 "accept_pct": round(100.0 * taken / n, 1) if n else None, "log": self.guard_log}
        return out

    def final_model_state(self) -> dict:
        """Deployed model w_{L-1}; the base learner before the first consolidation."""
        return self.levels[-1].state_dict() if self.levels else self.method.model.state_dict()

    def _core_state(self) -> dict:
        """Learner state a revision rolls back: method state, levels, DiLoCo momentum, FedAvg counter,
        seen tasks, own buffer. References live tensors: deep-copy before keeping it."""
        return {"method": self.method.state_dict(), "levels": [lv.state_dict() for lv in self.levels],
                "momentum": self.momentum, "seen_samples": self.seen_samples, "seen_tasks": self.seen_tasks,
                "buffer": self.buffer.state_dict() if self.buffer is not None else None}

    def state_dict(self) -> dict:
        """Core state plus the group schedule's last top row and the online window (snapshot included)."""
        return {**self._core_state(), "last_top_row": self.last_top_row, "window": self.window,
                "window_index": self.window_index, "snapshot": self.snapshot, "revisions": self.revisions,
                "guard_log": self.guard_log}

    def load_state_dict(self, state: dict) -> None:
        """Restore (tensors arrive on the CPU)."""
        self._load_core(state)
        self.last_top_row = state["last_top_row"]
        self.window, self.window_index = list(state["window"]), state["window_index"]
        self.snapshot, self.revisions = state["snapshot"], list(state["revisions"])
        self.guard_log = list(state.get("guard_log", []))

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
