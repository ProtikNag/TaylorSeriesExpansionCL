import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch_geometric.datasets import Planetoid
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class ContinualCIFAR100:
    def __init__(self, data_dir="./data", num_tasks=10, batch_size=64, seed=42,
                 debug=False, samples_per_class=5):
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.debug = debug
        self.samples_per_class = samples_per_class

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=self.test_transform
        )

        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self.task_train_loaders, self.task_test_loaders = self._build_task_loaders()

    def _create_task_splits(self):
        np.random.seed(self.seed)
        all_classes = np.arange(100)
        np.random.shuffle(all_classes)
        return np.array_split(all_classes, self.num_tasks)

    def _build_label_maps(self):
        return [
            {original: new for new, original in enumerate(class_subset)}
            for class_subset in self.task_splits
        ]

    def _remap_labels(self, dataset, class_subset, task_id):
        mapping = self.label_maps[task_id]
        for i in range(len(dataset.dataset.targets)):
            old_label = dataset.dataset.targets[i]
            if old_label in mapping:
                dataset.dataset.targets[i] = mapping[old_label]
        return dataset

    @staticmethod
    def _select_subset_indices(dataset, class_subset, max_per_class):
        """Return indices from dataset where each class appears at most max_per_class times."""
        class_counts = {cls: 0 for cls in class_subset}
        selected_indices = []

        for idx, (_, label) in enumerate(dataset):
            if label in class_subset and class_counts[label] < max_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            if all(c >= max_per_class for c in class_counts.values()):
                break

        return selected_indices

    def _build_task_loaders(self):
        task_train_loaders = []
        task_test_loaders = []

        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = self._select_subset_indices(self.train_dataset, class_subset, self.samples_per_class)
                test_indices = self._select_subset_indices(self.test_dataset, class_subset, self.samples_per_class)
            else:
                train_indices = [i for i, (_, label) in enumerate(self.train_dataset) if label in class_subset]
                test_indices = [i for i, (_, label) in enumerate(self.test_dataset) if label in class_subset]

            train_subset = Subset(self.train_dataset, train_indices)
            test_subset = Subset(self.test_dataset, test_indices)

            # Remap labels to 0–9
            train_subset = self._remap_labels(train_subset, class_subset, task_id)
            test_subset = self._remap_labels(test_subset, class_subset, task_id)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            task_train_loaders.append(train_loader)
            task_test_loaders.append(test_loader)

        return task_train_loaders, task_test_loaders

    def get_task_loaders(self):
        return self.task_train_loaders, self.task_test_loaders

    def get_task_class_mapping(self):
        """
        Returns: list of original CIFAR-100 class IDs for each task
        """
        return self.task_splits

    def get_label_maps(self):
        """
        Returns: list of dicts mapping original class labels to [0–9] per task
        """
        return self.label_maps


class ContinualSplitMNIST:
    def __init__(self, data_dir="./data", num_tasks=5, batch_size=64, seed=42,
                 debug=False, samples_per_class=5):
        """
        SplitMNIST with domain incremental setup.
        - 10 classes (digits 0–9)
        - num_tasks splits (default=5), each task has 2 classes
        - Labels are remapped within each task
        """
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.debug = debug
        self.samples_per_class = samples_per_class

        self.train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=self.test_transform
        )

        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self.task_train_loaders, self.task_test_loaders = self._build_task_loaders()

    def _create_task_splits(self):
        """
        Split MNIST into tasks (e.g., 0–1, 2–3, ..., 8–9)
        """
        np.random.seed(self.seed)
        all_classes = np.arange(10)
        return np.array_split(all_classes, self.num_tasks)

    def _build_label_maps(self):
        """
        Build a mapping per task: original_class -> new_class (0,1)
        """
        return [
            {original: new for new, original in enumerate(class_subset)}
            for class_subset in self.task_splits
        ]

    def _remap_labels(self, dataset, class_subset, task_id):
        mapping = self.label_maps[task_id]
        for i in range(len(dataset.dataset.targets)):
            old_label = int(dataset.dataset.targets[i])
            if old_label in mapping:
                dataset.dataset.targets[i] = mapping[old_label]
        return dataset

    @staticmethod
    def _select_subset_indices(dataset, class_subset, max_per_class):
        """Return indices from dataset where each class appears at most max_per_class times."""
        class_counts = {cls: 0 for cls in class_subset}
        selected_indices = []

        for idx, (_, label) in enumerate(dataset):
            label = int(label)
            if label in class_subset and class_counts[label] < max_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            if all(c >= max_per_class for c in class_counts.values()):
                break

        return selected_indices

    def _build_task_loaders(self):
        task_train_loaders = []
        task_test_loaders = []

        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = self._select_subset_indices(self.train_dataset, class_subset, self.samples_per_class)
                test_indices = self._select_subset_indices(self.test_dataset, class_subset, self.samples_per_class)
            else:
                train_indices = [i for i, (_, label) in enumerate(self.train_dataset) if int(label) in class_subset]
                test_indices = [i for i, (_, label) in enumerate(self.test_dataset) if int(label) in class_subset]

            train_subset = Subset(self.train_dataset, train_indices)
            test_subset = Subset(self.test_dataset, test_indices)

            # Remap labels to 0,1 for each task
            train_subset = self._remap_labels(train_subset, class_subset, task_id)
            test_subset = self._remap_labels(test_subset, class_subset, task_id)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            task_train_loaders.append(train_loader)
            task_test_loaders.append(test_loader)

        return task_train_loaders, task_test_loaders

    def get_task_loaders(self):
        return self.task_train_loaders, self.task_test_loaders

    def get_task_class_mapping(self):
        """
        Returns: list of original MNIST class IDs for each task
        """
        return self.task_splits

    def get_label_maps(self):
        """
        Returns: list of dicts mapping original class labels to [0–1] per task
        """
        return self.label_maps


class NodeDataset(Dataset):
    """
    Thin Dataset wrapper that exposes node-wise (x, y) pairs for a single PyG Data object.
    Expects x: [num_nodes, feat_dim], y: [num_nodes]
    """

    def __init__(self, x, y):
        # store as tensors
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])


class ContinualCora:
    def __init__(self, data_dir="./data", num_tasks=3, batch_size=64, seed=42,
                 debug=False, samples_per_class=5):
        """
        Cora with domain incremental setup (structured to match ContinualSplitMNIST interface).
        - 7 classes (Cora)
        - num_tasks splits (default=3)
        - Labels are remapped within each task
        NOTE: requires torch-geometric (Planetoid) to be installed and available.
        """
        if Planetoid is None:
            raise ImportError(
                "torch-geometric Planetoid dataset not available. "
                "Please install torch-geometric to use ContinualCora."
            )

        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.debug = debug
        self.samples_per_class = samples_per_class

        # keep same attributes as MNIST class for compatibility (no-op transforms)
        self.train_transform = lambda x: x
        self.test_transform = lambda x: x

        # Load Cora (single graph). We'll expose nodes as a node-wise dataset.
        self.raw_dataset = Planetoid(root=data_dir, name="Cora")
        # Planetoid returns a dataset with one graph (dataset[0])
        data = self.raw_dataset[0]

        # Node features and labels
        x = data.x.clone()
        y = data.y.clone()

        # Create a NodeDataset that holds the whole graph nodes (we will copy per-task when needed)
        self.full_node_dataset = NodeDataset(x, y)

        # Build tasks by splitting class ids into num_tasks parts
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self.task_train_loaders, self.task_test_loaders = self._build_task_loaders()

    def _create_task_splits(self):
        """
        Split Cora classes into tasks.
        """
        np.random.seed(self.seed)
        all_classes = np.arange(7)  # Cora has 7 classes
        return np.array_split(all_classes, self.num_tasks)

    def _build_label_maps(self):
        """
        Build a mapping per task: original_class -> new_class (0..(task_classes-1))
        """
        return [
            {original: new for new, original in enumerate(class_subset)}
            for class_subset in self.task_splits
        ]

    def _remap_labels(self, dataset, class_subset, task_id):
        """
        Remap labels in-place for dataset.dataset (matches behavior of SplitMNIST implementation).
        Expects dataset to be a Subset whose .dataset is a NodeDataset instance.
        """
        mapping = self.label_maps[task_id]
        # dataset is a torch.utils.data.Subset; dataset.dataset is NodeDataset
        base_dataset = dataset.dataset
        # We need to remap only labels for indices present in subset
        for idx in dataset.indices:
            old_label = int(base_dataset.y[idx])
            if old_label in mapping:
                base_dataset.y[idx] = mapping[old_label]
        return dataset

    @staticmethod
    def _select_subset_indices(dataset, class_subset, max_per_class):
        """
        Return indices from dataset where each class appears at most max_per_class times.
        dataset here can be NodeDataset or any dataset whose __getitem__ returns (x, label).
        """
        class_counts = {int(cls): 0 for cls in class_subset}
        selected_indices = []

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label = int(label)
            if label in class_counts and class_counts[label] < max_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            if all(c >= max_per_class for c in class_counts.values()):
                break

        return selected_indices

    def _build_task_loaders(self):
        task_train_loaders = []
        task_test_loaders = []

        # For Cora (single graph), we'll simulate train/test splits by splitting nodes into
        # train and test sets (common simple approach): we'll use 80% nodes as "train" and 20% as "test"
        num_nodes = len(self.full_node_dataset)
        rng = np.random.RandomState(self.seed)
        all_indices = np.arange(num_nodes)
        rng.shuffle(all_indices)
        split = int(0.8 * num_nodes)
        train_node_indices = set(all_indices[:split])
        test_node_indices = set(all_indices[split:])

        for task_id, class_subset in enumerate(self.task_splits):
            # For debug mode, pick up to samples_per_class nodes per class for train/test respectively
            if self.debug:
                # create a small train subset selecting up to samples_per_class from train_node_indices
                train_indices = []
                class_counts = {int(cls): 0 for cls in class_subset}
                for idx in train_node_indices:
                    _, label = self.full_node_dataset[idx]
                    label = int(label)
                    if label in class_subset and class_counts[label] < self.samples_per_class:
                        train_indices.append(idx)
                        class_counts[label] += 1
                    if all(c >= self.samples_per_class for c in class_counts.values()):
                        break

                # same for test
                test_indices = []
                class_counts = {int(cls): 0 for cls in class_subset}
                for idx in test_node_indices:
                    _, label = self.full_node_dataset[idx]
                    label = int(label)
                    if label in class_subset and class_counts[label] < self.samples_per_class:
                        test_indices.append(idx)
                        class_counts[label] += 1
                    if all(c >= self.samples_per_class for c in class_counts.values()):
                        break
            else:
                train_indices = [int(idx) for idx in train_node_indices if int(self.full_node_dataset[idx][1]) in class_subset]
                test_indices = [int(idx) for idx in test_node_indices if int(self.full_node_dataset[idx][1]) in class_subset]

            # For each task, create copies of the base NodeDataset so remapping doesn't bleed between tasks
            task_node_dataset_for_train = NodeDataset(self.full_node_dataset.x.clone(), self.full_node_dataset.y.clone())
            task_node_dataset_for_test = NodeDataset(self.full_node_dataset.x.clone(), self.full_node_dataset.y.clone())

            train_subset = Subset(task_node_dataset_for_train, train_indices)
            test_subset = Subset(task_node_dataset_for_test, test_indices)

            # Remap labels to 0..(k-1) for each task
            train_subset = self._remap_labels(train_subset, class_subset, task_id)
            test_subset = self._remap_labels(test_subset, class_subset, task_id)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            task_train_loaders.append(train_loader)
            task_test_loaders.append(test_loader)

        return task_train_loaders, task_test_loaders

    def get_task_loaders(self):
        return self.task_train_loaders, self.task_test_loaders

    def get_task_class_mapping(self):
        """
        Returns: list of original Cora class IDs for each task
        """
        return self.task_splits

    def get_label_maps(self):
        """
        Returns: list of dicts mapping original class labels to [0..k-1] per task
        """
        return self.label_maps


class TextDataset(Dataset):
    """
    Simple Dataset wrapper for vectorized text data.
    - data: list or numpy array of feature vectors (torch tensors)
    - targets: numpy array of integer labels (mutable so remapping can modify in-place)
    """

    def __init__(self, data_vectors, targets):
        # store as list of tensors and numpy array of targets to allow in-place assignment
        self.data = list(data_vectors)
        self.targets = np.array(targets, dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


class Continual20Newsgroups:
    def __init__(self, data_dir="./data", num_tasks=5, batch_size=64, seed=42,
                 debug=False, samples_per_class=5):
        """
        20 Newsgroups with domain incremental setup (matching the ContinualSplitMNIST interface).
        - 20 classes (20 Newsgroups)
        - num_tasks splits (default=5), each task will have ~4 classes (np.array_split)
        - Labels are remapped within each task
        """
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.debug = debug
        self.samples_per_class = samples_per_class

        # No-op "transforms" attributes for compatibility with the SplitMNIST class structure
        self.train_transform = lambda x: x
        self.test_transform = lambda x: x

        # Load 20 Newsgroups train/test raw text and labels
        train_raw = fetch_20newsgroups(subset="train", remove=())
        test_raw = fetch_20newsgroups(subset="test", remove=())

        # Vectorize text (fit on train, transform both)
        vectorizer = TfidfVectorizer(max_features=20000)  # limit vocab size for memory
        train_vectors = vectorizer.fit_transform(train_raw.data)
        test_vectors = vectorizer.transform(test_raw.data)

        # Convert sparse vectors to dense torch tensors (could be memory heavy; adjust max_features if needed)
        # Keep them as float32 tensors to match typical model inputs
        train_tensors = [torch.from_numpy(train_vectors[i].toarray().squeeze().astype(np.float32)) for i in range(train_vectors.shape[0])]
        test_tensors = [torch.from_numpy(test_vectors[i].toarray().squeeze().astype(np.float32)) for i in range(test_vectors.shape[0])]

        # Create dataset objects with .targets attribute to allow remapping (mutable numpy array)
        self.train_dataset = TextDataset(train_tensors, train_raw.target)
        self.test_dataset = TextDataset(test_tensors, test_raw.target)

        # Build task splits, label maps and loaders (mirrors SplitMNIST class behavior)
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self.task_train_loaders, self.task_test_loaders = self._build_task_loaders()

    def _create_task_splits(self):
        """
        Split 20 classes into num_tasks groups (np.array_split).
        """
        np.random.seed(self.seed)
        all_classes = np.arange(20)
        return np.array_split(all_classes, self.num_tasks)

    def _build_label_maps(self):
        """
        Build a mapping per task: original_class -> new_class (0..k-1 for that task)
        """
        return [
            {original: new for new, original in enumerate(class_subset)}
            for class_subset in self.task_splits
        ]

    def _remap_labels(self, dataset, class_subset, task_id):
        """
        Remap labels in-place for dataset.dataset (keeps the same behavior as your SplitMNIST implementation).
        Expects dataset to be a Subset whose .dataset is a TextDataset instance.
        """
        mapping = self.label_maps[task_id]
        for i in range(len(dataset.dataset.targets)):
            old_label = int(dataset.dataset.targets[i])
            if old_label in mapping:
                dataset.dataset.targets[i] = mapping[old_label]
        return dataset

    @staticmethod
    def _select_subset_indices(dataset, class_subset, max_per_class):
        """Return indices from dataset where each class appears at most max_per_class times."""
        class_counts = {int(cls): 0 for cls in class_subset}
        selected_indices = []

        for idx, (_, label) in enumerate(dataset):
            label = int(label)
            if label in class_subset and class_counts[label] < max_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            if all(c >= max_per_class for c in class_counts.values()):
                break

        return selected_indices

    def _build_task_loaders(self):
        task_train_loaders = []
        task_test_loaders = []

        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = self._select_subset_indices(self.train_dataset, class_subset, self.samples_per_class)
                test_indices = self._select_subset_indices(self.test_dataset, class_subset, self.samples_per_class)
            else:
                train_indices = [i for i, (_, label) in enumerate(self.train_dataset) if int(label) in class_subset]
                test_indices = [i for i, (_, label) in enumerate(self.test_dataset) if int(label) in class_subset]

            train_subset = Subset(self.train_dataset, train_indices)
            test_subset = Subset(self.test_dataset, test_indices)

            # Remap labels to 0..(k-1) for each task
            train_subset = self._remap_labels(train_subset, class_subset, task_id)
            test_subset = self._remap_labels(test_subset, class_subset, task_id)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            task_train_loaders.append(train_loader)
            task_test_loaders.append(test_loader)

        return task_train_loaders, task_test_loaders

    def get_task_loaders(self):
        return self.task_train_loaders, self.task_test_loaders

    def get_task_class_mapping(self):
        """
        Returns: list of original 20 Newsgroups class IDs for each task
        """
        return self.task_splits

    def get_label_maps(self):
        """
        Returns: list of dicts mapping original class labels to [0..k-1] per task
        """
        return self.label_maps
