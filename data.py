import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
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
