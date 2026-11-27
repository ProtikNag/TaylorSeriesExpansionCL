"""iCaRL (Rebuffi et al., CVPR 2017) adapted to the domain and class scenarios.

- Exemplar memory of total capacity K = buffer_size, one exemplar set per class keyed by
  (task_id, label): in the domain scenario labels are remapped to 0..C-1 in every task, so the
  label alone would let a new task overwrite an earlier task's exemplars.
- Training set of a task = its data U the exemplars (Alg. 2). Each batch is completed with
  ceil(|M| / n_batches) exemplars drawn without replacement, so one epoch visits every exemplar once.
- Loss (Alg. 3, sigmoid BCE): one-hot targets on the classification units, the previous model's
  sigmoid outputs q = sigma(f_old(x)) on the distillation units, on new and exemplar samples alike.
  Class scenario: classification units = current task's classes, distillation units = classes seen
  before (Rebuffi's split). Domain scenario: every task uses all C units, so all units get both terms.
  Normalisation: sum of the BCE terms / (batch x units used), as Mammoth's iCaRL (mean BCE).
- Exemplars are chosen by herding (Alg. 4) on L2-normalised penultimate features model.features(x),
  over all of the task's training samples under the plain (un-augmented) transform, and stored in that view.
  Memory is rebalanced at end_task to m = K // (#classes seen) per class (Alg. 5: keep the first m).
- Inference: nearest mean of exemplars (Alg. 1). Class means are the re-normalised means of the
  exemplars' normalised features, recomputed in prepare_eval with the current weights. The score of
  output unit y is max over classes with label y of -||phi(x) - mu_class||, i.e. argmax picks the
  label of the nearest class mean (domain: max-pooled over the tasks that share label y; class:
  one class per label). Units without exemplars score -inf.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..data import make_loader
from .base import Method

FEATURE_BATCH = 256


class ExemplarMemory:
    """Exemplars as flat CPU tensors (x, y, task); rows of a class are contiguous in herding order.

    Exposes `x`, `y` and `len()`, which is what HTCL's BufferDataset reads for its consolidation data.
    """

    def __init__(self):
        """Empty memory."""
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.task: torch.Tensor | None = None

    def __len__(self) -> int:
        """Number of stored exemplars."""
        return 0 if self.y is None else len(self.y)

    def classes(self) -> list[tuple[int, int]]:
        """(task_id, label) keys present, in storage order."""
        if len(self) == 0:
            return []
        return list(dict.fromkeys(zip(self.task.tolist(), self.y.tolist())))

    def keep(self, rows: torch.Tensor) -> None:
        """Keep only the given rows (in the given order)."""
        self.x, self.y, self.task = self.x[rows], self.y[rows], self.task[rows]

    def append(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        """Add exemplars of one task."""
        task = torch.full_like(y, task_id)
        if len(self) == 0:
            self.x, self.y, self.task = x, y, task
        else:
            self.x, self.y, self.task = torch.cat([self.x, x]), torch.cat([self.y, y]), torch.cat([self.task, task])

    def state_dict(self) -> dict:
        """Tensors (None when empty)."""
        return {"x": self.x, "y": self.y, "task": self.task}

    def load_state_dict(self, state: dict) -> None:
        """Restore from `state_dict` output."""
        self.x, self.y, self.task = state["x"], state["y"], state["task"]


def herding(features: torch.Tensor, m: int) -> list[int]:
    """Rebuffi Alg. 4: greedily pick rows whose running mean best approximates the class mean.

    features: (n, d) L2-normalised. Returns min(m, n) distinct row indices in selection order.
    """
    mu = features.mean(0)
    running = torch.zeros_like(mu)
    available = torch.ones(len(features), dtype=torch.bool)
    chosen: list[int] = []
    for k in range(1, min(m, len(features)) + 1):
        dist = (mu - (running + features) / k).norm(dim=1)
        dist[~available] = float("inf")
        i = int(dist.argmin())
        chosen.append(i)
        available[i] = False
        running += features[i]
    return chosen


class ICaRL(Method):
    """iCaRL: herding exemplars keyed by (task_id, label), BCE + distillation, NME classifier."""

    def __init__(self, model, cfg: dict, device: torch.device):
        """Method with an empty exemplar memory of `buffer_size` samples in total."""
        super().__init__(model, cfg, device)
        self.capacity = int(cfg["buffer_size"])
        self.num_outputs = int(cfg["num_outputs"])
        self.class_scenario = cfg.get("scenario", "domain") == "class"
        self.buffer = ExemplarMemory()  # named `buffer` so HTCL replays it during consolidation
        self.old_model: torch.nn.Module | None = None
        self.cls_units = self.kd_units = None
        self.replay_per_step, self.stream = 0, torch.empty(0, dtype=torch.long)
        self.prototypes: torch.Tensor | None = None
        self.proto_labels: torch.Tensor | None = None

    # ------------------------------------------------------------------ training
    def begin_task(self, task_id: int, train_loader) -> None:
        """Fresh optimizer, frozen copy of the current model for distillation, unit masks, replay rate."""
        super().begin_task(task_id, train_loader)
        seen = self.buffer.classes()
        self.old_model = None
        if seen:
            self.old_model = copy.deepcopy(self.model).eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)
        units = torch.ones(self.num_outputs, dtype=torch.bool, device=self.device)
        if self.class_scenario:
            # Rebuffi's split: one-hot targets on the new classes, distillation on the old ones.
            self.cls_units = torch.zeros_like(units)
            self.cls_units[torch.as_tensor(np.unique(train_loader.dataset.labels), device=self.device)] = True
            self.kd_units = torch.zeros_like(units)
            if seen:
                self.kd_units[torch.tensor([label for _, label in seen], device=self.device)] = True
        else:  # domain: every task uses all units
            self.cls_units, self.kd_units = units, units if seen else torch.zeros_like(units)
        self.replay_per_step = math.ceil(len(self.buffer) / max(1, len(train_loader)))
        self.stream = torch.empty(0, dtype=torch.long)

    def _next_exemplars(self) -> torch.Tensor:
        """Next replay_per_step exemplar rows from a reshuffled pass over the memory."""
        if len(self.stream) < self.replay_per_step:
            self.stream = torch.cat([self.stream, torch.randperm(len(self.buffer))])
        rows, self.stream = self.stream[:self.replay_per_step], self.stream[self.replay_per_step:]
        return rows

    def observe(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        """BCE on the classification units + BCE against sigma(old logits) on the distillation units."""
        if self.replay_per_step > 0:
            rows = self._next_exemplars()
            x = torch.cat([x, self.buffer.x[rows].to(self.device)])
            y = torch.cat([y, self.buffer.y[rows].to(self.device)])
        logits = self.model(x)
        terms = F.binary_cross_entropy_with_logits(logits, F.one_hot(y, self.num_outputs).float(),
                                                   reduction="none") * self.cls_units
        if self.old_model is not None:
            with torch.no_grad():
                q = torch.sigmoid(self.old_model(x))
            terms = terms + F.binary_cross_entropy_with_logits(logits, q, reduction="none") * self.kd_units
        loss = terms.sum() / (len(x) * int((self.cls_units | self.kd_units).sum()))
        return self.optimize(loss)

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised penultimate features of CPU inputs, computed in batches on the device."""
        return torch.cat([F.normalize(self.model.features(x[i:i + FEATURE_BATCH].to(self.device)), dim=1)
                          for i in range(0, len(x), FEATURE_BATCH)])

    @torch.no_grad()
    def end_task(self, task_id: int, train_loader) -> None:
        """Rebalance: shrink old exemplar sets to m = K // #classes, herd m exemplars per new class."""
        self.old_model = None
        # Herd over every training sample of the task under the plain transform (Rebuffi et al.): the
        # train loader is shuffled, augmented and drops its last batch. TaskDataset.plain holds that view.
        dataset = getattr(train_loader.dataset, "plain", None) or train_loader.dataset
        xs, ys = zip(*[(x.cpu(), y.cpu()) for x, y in make_loader(dataset, FEATURE_BATCH, shuffle=False)])
        x_all, y_all = torch.cat(xs), torch.cat(ys)
        new_labels = sorted(set(y_all.tolist()))
        old = self.buffer.classes()
        m = self.capacity // (len(old) + len(new_labels))
        if old:
            self.buffer.keep(torch.cat([torch.nonzero((self.buffer.task == t) & (self.buffer.y == lab)).flatten()[:m]
                                        for t, lab in old]))
        was_training = self.model.training
        self.model.eval()
        for label in new_labels:
            x_class = x_all[y_all == label]
            picked = herding(self._features(x_class).cpu(), m)
            self.buffer.append(x_class[picked], torch.full((len(picked),), label, dtype=torch.long), task_id)
        self.model.train(was_training)

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def prepare_eval(self) -> None:
        """Class means of the exemplars under the current weights (engine has set eval mode)."""
        self.prototypes = self.proto_labels = None
        classes = self.buffer.classes()
        if not classes:
            return
        feats = self._features(self.buffer.x)
        task, y = self.buffer.task.to(self.device), self.buffer.y.to(self.device)
        self.prototypes = F.normalize(torch.stack([feats[(task == t) & (y == lab)].mean(0) for t, lab in classes]),
                                      dim=1)
        self.proto_labels = torch.tensor([lab for _, lab in classes], device=self.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """NME scores: per output unit, max over its classes of -||phi(x) - mu||; logits if no memory."""
        if self.prototypes is None:
            return self.model(x)
        neg_dist = -torch.cdist(F.normalize(self.model.features(x), dim=1), self.prototypes)
        scores = torch.full((len(x), self.num_outputs), float("-inf"), device=neg_dist.device)
        return scores.scatter_reduce(1, self.proto_labels.expand(len(x), -1), neg_dist, reduce="amax")

    # --------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        """Model and exemplar memory (old model, masks and prototypes are rebuilt from these)."""
        return {**super().state_dict(), "memory": self.buffer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore model and memory (memory stays on the CPU)."""
        super().load_state_dict(state)
        self.buffer.load_state_dict(state["memory"])
        self.prototypes = self.proto_labels = None
