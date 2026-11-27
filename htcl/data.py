"""Continual-learning datasets: class-to-task split, subsampling, validation split, scenarios.

Every dataset is reduced to the same raw form (augmented train view, plain train view, train
targets, plain test view, test targets), and `build_data` turns it into tasks:

1. classes go to tasks in a fixed seeded shuffle (dataset seed), independent of the order;
2. `samples_per_class` draws a seeded random subset per class (before the validation split);
3. a stratified, seeded 10% of each class becomes validation (plain transforms);
4. labels are remapped per task (`domain`) or kept global (`class`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VAL_FRACTION = 0.1
MNIST_STATS = ((0.1307,), (0.3081,))
CIFAR100_STATS = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
# Fixed task split of the loss-landscape experiment (E4).
CLASSIL3_TASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
DEFAULT_TASKS = {"splitmnist": 5, "cifar100": 10, "news20": 5, "cora": 3, "imagenet100": 10, "mnist_classil3": 3}


class ImageTensorDataset(Dataset):
    """uint8 images (N, C, H, W) in memory, normalised per item, optional crop-pad-4 + flip."""

    def __init__(self, images: torch.Tensor, stats: tuple, augment: bool = False):
        """Store images; `stats` is (mean, std) per channel."""
        self.images = images
        self.mean = torch.tensor(stats[0]).view(-1, 1, 1)
        self.std = torch.tensor(stats[1]).view(-1, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        """Number of images."""
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Return (normalised image, 0); labels come from TaskDataset."""
        x = self.images[i].float() / 255.0
        if self.augment:  # RandomCrop(32, padding=4) + RandomHorizontalFlip, torch global RNG
            size = x.shape[-1]
            x = F.pad(x, (4, 4, 4, 4))
            top, left = torch.randint(0, 9, (2,)).tolist()
            x = x[:, top:top + size, left:left + size]
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
        return (x - self.mean) / self.std, 0


class SparseRowDataset(Dataset):
    """Rows of a scipy CSR matrix, densified per item (TF-IDF features)."""

    def __init__(self, matrix):
        """Wrap a CSR matrix."""
        self.matrix = matrix

    def __len__(self) -> int:
        """Number of rows."""
        return self.matrix.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Return (dense float row, 0)."""
        return torch.from_numpy(self.matrix[i].toarray()[0]).float(), 0


class IndexDataset(Dataset):
    """Items are their own indices (graph nodes: the model holds the full graph)."""

    def __init__(self, n: int):
        """n items."""
        self.n = n

    def __len__(self) -> int:
        """Number of nodes."""
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Return (node index, 0)."""
        return torch.tensor(i), 0


class TaskDataset(Dataset):
    """A subset of a base dataset with task-specific labels."""

    def __init__(self, base: Dataset, indices: np.ndarray, labels: np.ndarray, plain: "TaskDataset | None" = None):
        """indices into `base`, labels aligned with indices; `plain`: the same samples under the plain
        (un-augmented) transform, set on training sets (iCaRL herds on it)."""
        self.base, self.indices, self.labels, self.plain = base, indices, labels, plain

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Return (input, remapped label)."""
        return self.base[int(self.indices[i])][0], int(self.labels[i])


@dataclass
class Task:
    """One task: its classes (original ids) and train/val/test datasets."""
    classes: list[int]
    train: TaskDataset
    val: TaskDataset
    test: TaskDataset


@dataclass
class CLData:
    """All tasks of a dataset plus what the model factory needs."""
    name: str
    scenario: str
    tasks: list[Task]
    num_outputs: int
    input_dim: int | None = None
    graph: tuple[torch.Tensor, torch.Tensor] | None = field(default=None, repr=False)


def make_loader(dataset: Dataset, batch_size: int | str, shuffle: bool) -> DataLoader:
    """DataLoader in the main process (MPS-friendly); batch_size 'full' means one batch.

    A shuffled loader that holds more than one batch drops the incomplete trailing batch: a batch of
    2-4 samples blows up BatchNorm at 1x1 spatial size (SplitMNIST ResNet), and reshuffling each epoch
    still visits every sample. A dataset of at most one batch is kept whole.
    """
    size = max(1, len(dataset) if batch_size in ("full", None) else int(batch_size))
    drop_last = shuffle and len(dataset) > size
    return DataLoader(dataset, batch_size=size, shuffle=shuffle, num_workers=0, drop_last=drop_last)


def _per_class_subsample(targets: np.ndarray, cap: int | None, rng: np.random.RandomState) -> np.ndarray:
    """Random subset of at most `cap` indices per class (all indices if cap is None)."""
    if cap is None:
        return np.arange(len(targets))
    keep = [rng.choice(np.flatnonzero(targets == c), min(cap, int((targets == c).sum())), replace=False)
            for c in np.unique(targets)]
    return np.sort(np.concatenate(keep))


def _stratified_split(indices: np.ndarray, targets: np.ndarray, fraction: float,
                      rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Split indices per class into (rest, held-out fraction); at least one held out when a class has >= 2."""
    rest, held = [], []
    for c in np.unique(targets[indices]):
        idx = rng.permutation(indices[targets[indices] == c])
        n_held = max(1, int(round(fraction * len(idx)))) if len(idx) >= 2 else 0
        held.append(idx[:n_held])
        rest.append(idx[n_held:])
    return np.sort(np.concatenate(rest)), np.sort(np.concatenate(held))


def _class_tasks(name: str, num_classes: int, num_tasks: int, seed: int,
                 classes_per_task: int | None = None) -> list[list[int]]:
    """Fixed seeded shuffle of classes split into tasks (np.array_split sizes, e.g. Cora 3/2/2).

    With classes_per_task only the first num_tasks * classes_per_task classes of the shuffle are used,
    so shorter sequences are prefixes of longer ones and per-task difficulty does not change with T.
    """
    if name == "mnist_classil3":
        return CLASSIL3_TASKS
    perm = np.random.RandomState(seed).permutation(num_classes)
    if classes_per_task:
        if num_tasks * classes_per_task > num_classes:
            raise ValueError(f"{num_tasks} tasks x {classes_per_task} classes exceeds {num_classes} classes")
        perm = perm[:num_tasks * classes_per_task]
    return [sorted(chunk.tolist()) for chunk in np.array_split(perm, num_tasks)]


def _load_mnist(root: Path) -> tuple:
    """MNIST under data/MNIST (downloaded on first use)."""
    from torchvision.datasets import MNIST
    train, test = MNIST(root, train=True, download=True), MNIST(root, train=False, download=True)
    tr_img, te_img = train.data.unsqueeze(1), test.data.unsqueeze(1)
    return (ImageTensorDataset(tr_img, MNIST_STATS), ImageTensorDataset(tr_img, MNIST_STATS), train.targets.numpy(),
            ImageTensorDataset(te_img, MNIST_STATS), test.targets.numpy(), {})


def _load_cifar100(root: Path) -> tuple:
    """CIFAR-100 under data/cifar-100-python (downloaded on first use); train view: crop pad 4 + flip."""
    from torchvision.datasets import CIFAR100
    train, test = CIFAR100(root, train=True, download=True), CIFAR100(root, train=False, download=True)
    tr_img = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    te_img = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
    return (ImageTensorDataset(tr_img, CIFAR100_STATS, augment=True), ImageTensorDataset(tr_img, CIFAR100_STATS),
            np.asarray(train.targets), ImageTensorDataset(te_img, CIFAR100_STATS), np.asarray(test.targets), {})


def _load_news20(root: Path) -> tuple:
    """20 Newsgroups, headers/footers/quotes removed, TF-IDF (20k features) fit on the train split only."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    strip = ("headers", "footers", "quotes")
    train = fetch_20newsgroups(subset="train", remove=strip)
    test = fetch_20newsgroups(subset="test", remove=strip)
    vectorizer = TfidfVectorizer(max_features=20000)
    x_train = vectorizer.fit_transform(train.data).tocsr().astype(np.float32)
    x_test = vectorizer.transform(test.data).tocsr().astype(np.float32)
    return (SparseRowDataset(x_train), SparseRowDataset(x_train), np.asarray(train.target),
            SparseRowDataset(x_test), np.asarray(test.target), {"input_dim": x_train.shape[1]})


def _load_cora(root: Path, seed: int) -> tuple:
    """Cora full graph; stratified 80/20 node split (seeded). Items are node indices."""
    from torch_geometric.datasets import Planetoid
    graph = Planetoid(str(root / "Planetoid"), name="Cora")[0]
    labels = graph.y.numpy()
    train_nodes, test_nodes = _stratified_split(np.arange(len(labels)), labels, 0.2, np.random.RandomState(seed + 10))
    nodes = IndexDataset(len(labels))
    # Train/test "views" index the node set, so their targets are the labels of the chosen nodes.
    return (_Reindexed(nodes, train_nodes), _Reindexed(nodes, train_nodes), labels[train_nodes],
            _Reindexed(nodes, test_nodes), labels[test_nodes],
            {"graph": (graph.x, graph.edge_index), "input_dim": graph.x.shape[1]})


class _Reindexed(Dataset):
    """View of a base dataset through an index array."""

    def __init__(self, base: Dataset, indices: np.ndarray):
        """Base dataset and the positions it exposes."""
        self.base, self.indices = base, indices

    def __len__(self) -> int:
        """Number of exposed items."""
        return len(self.indices)

    def __getitem__(self, i: int):
        """Item `indices[i]` of the base."""
        return self.base[int(self.indices[i])]


def _load_imagenet100(root: Path) -> tuple:
    """ImageFolder root with train/ and val/ subfolders, timm transforms of vit_small_patch16_224."""
    import timm
    from timm.data import create_transform, resolve_data_config
    from torchvision.datasets import ImageFolder
    data_cfg = resolve_data_config(pretrained_cfg=timm.get_pretrained_cfg("vit_small_patch16_224").to_dict())
    train_aug = ImageFolder(str(root / "train"), transform=create_transform(**data_cfg, is_training=True))
    train_plain = copy.copy(train_aug)
    train_plain.transform = create_transform(**data_cfg, is_training=False)
    test = ImageFolder(str(root / "val"), transform=train_plain.transform)
    if test.classes != train_aug.classes:
        raise ValueError(f"train/ and val/ class folders differ under {root}")
    return train_aug, train_plain, np.asarray(train_aug.targets), test, np.asarray(test.targets), {}


def build_data(spec: dict) -> CLData:
    """Build all tasks from a resolved dataset spec.

    Keys used: name, scenario ('domain'|'class'), num_tasks, classes_per_task, samples_per_class,
    test_samples_per_class, seed (dataset seed, default 0), data_root, imagenet_root.
    """
    name, seed = spec["name"], int(spec.get("seed", 0))
    root = Path(spec.get("data_root", "data"))
    loaders = {"splitmnist": _load_mnist, "mnist_classil3": _load_mnist, "cifar100": _load_cifar100,
               "news20": _load_news20, "imagenet100": lambda r: _load_imagenet100(Path(spec["imagenet_root"]))}
    if name == "cora":
        raw = _load_cora(root, seed)
    elif name in loaders:
        raw = loaders[name](root)
    else:
        raise ValueError(f"unknown dataset {name}")
    train_aug, train_plain, train_targets, test_base, test_targets, extra = raw

    num_classes = int(train_targets.max()) + 1
    task_classes = _class_tasks(name, num_classes, int(spec.get("num_tasks", DEFAULT_TASKS[name])), seed,
                                spec.get("classes_per_task"))
    spc = spec.get("samples_per_class")
    test_spc = spec.get("test_samples_per_class", spc)
    train_idx = _per_class_subsample(train_targets, spc, np.random.RandomState(seed + 1))
    test_idx = _per_class_subsample(test_targets, test_spc, np.random.RandomState(seed + 3))
    fit_idx, val_idx = _stratified_split(train_idx, train_targets, VAL_FRACTION, np.random.RandomState(seed + 2))

    scenario = spec.get("scenario", "domain")
    tasks = []
    for classes in task_classes:
        # domain: position of the class within its task; class: global id.
        remap = {c: (i if scenario == "domain" else c) for i, c in enumerate(classes)}

        def subset(base: Dataset, idx: np.ndarray, targets: np.ndarray) -> TaskDataset:
            mine = idx[np.isin(targets[idx], classes)]
            return TaskDataset(base, mine, np.array([remap[int(c)] for c in targets[mine]], dtype=np.int64))

        train = subset(train_aug, fit_idx, train_targets)
        train.plain = subset(train_plain, fit_idx, train_targets)
        tasks.append(Task(classes, train, subset(train_plain, val_idx, train_targets),
                          subset(test_base, test_idx, test_targets)))
    num_outputs = max(len(c) for c in task_classes) if scenario == "domain" else num_classes
    return CLData(name, scenario, tasks, num_outputs, extra.get("input_dim"), extra.get("graph"))
