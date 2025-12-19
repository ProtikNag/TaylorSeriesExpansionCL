"""
Continual Learning Dataset implementations.
Provides a unified interface for various CL benchmarks.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional


class BaseContinualDataset(ABC):
    """Abstract base class for continual learning datasets."""
    
    def __init__(
        self,
        data_dir: str = "./data",
        num_tasks: int = 5,
        batch_size: int = 64,
        seed: int = 42,
        debug: bool = False,
        samples_per_class: int = 60,
        num_workers: int = 2,
    ):
        self.data_dir = data_dir
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.seed = seed
        self.debug = debug
        self.samples_per_class = samples_per_class
        self.num_workers = num_workers
        
        self.task_splits: List[np.ndarray] = []
        self.label_maps: List[Dict[int, int]] = []
        self.task_train_loaders: List[DataLoader] = []
        self.task_test_loaders: List[DataLoader] = []
    
    @abstractmethod
    def _load_data(self):
        """Load the raw dataset."""
        pass
    
    @abstractmethod
    def _create_task_splits(self) -> List[np.ndarray]:
        """Create class splits for each task."""
        pass
    
    def _build_label_maps(self) -> List[Dict[int, int]]:
        """Build label remapping for each task."""
        return [
            {original: new for new, original in enumerate(class_subset)}
            for class_subset in self.task_splits
        ]
    
    @staticmethod
    def _select_subset_indices(
        dataset: Dataset,
        class_subset: np.ndarray,
        max_per_class: int,
        label_accessor=lambda x: x[1]
    ) -> List[int]:
        """Select indices with at most max_per_class samples per class."""
        class_counts = {cls: 0 for cls in class_subset}
        selected_indices = []
        
        for idx in range(len(dataset)):
            label = int(label_accessor(dataset[idx]))
            if label in class_subset and class_counts[label] < max_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            if all(c >= max_per_class for c in class_counts.values()):
                break
        
        return selected_indices
    
    def get_task_loaders(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Return train and test loaders for all tasks."""
        return self.task_train_loaders, self.task_test_loaders
    
    def get_task_class_mapping(self) -> List[np.ndarray]:
        """Return original class IDs for each task."""
        return self.task_splits
    
    def get_label_maps(self) -> List[Dict[int, int]]:
        """Return label remapping dictionaries."""
        return self.label_maps
    
    @property
    def num_classes_per_task(self) -> int:
        """Return number of classes per task (assumes uniform split)."""
        if self.task_splits:
            return len(self.task_splits[0])
        return 0


class ContinualSplitMNIST(BaseContinualDataset):
    """SplitMNIST dataset for continual learning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self._build_task_loaders()
    
    def _load_data(self):
        self.train_transform = transforms.Compose([transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])
        
        self.train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True,
            transform=self.train_transform
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, download=True,
            transform=self.test_transform
        )
    
    def _create_task_splits(self) -> List[np.ndarray]:
        np.random.seed(self.seed)
        all_classes = np.arange(10)
        return list(np.array_split(all_classes, self.num_tasks))
    
    def _build_task_loaders(self):
        for task_id, class_subset in enumerate(self.task_splits):
            # Get indices for this task
            if self.debug:
                train_indices = self._select_subset_indices(
                    self.train_dataset, class_subset, self.samples_per_class
                )
                test_indices = self._select_subset_indices(
                    self.test_dataset, class_subset, self.samples_per_class
                )
            else:
                train_indices = [
                    i for i, (_, label) in enumerate(self.train_dataset)
                    if int(label) in class_subset
                ]
                test_indices = [
                    i for i, (_, label) in enumerate(self.test_dataset)
                    if int(label) in class_subset
                ]
            
            # Create wrapped datasets with remapped labels
            train_subset = RemappedSubset(
                self.train_dataset, train_indices,
                self.label_maps[task_id]
            )
            test_subset = RemappedSubset(
                self.test_dataset, test_indices,
                self.label_maps[task_id]
            )
            
            self.task_train_loaders.append(
                DataLoader(train_subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
            )
            self.task_test_loaders.append(
                DataLoader(test_subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
            )


class ContinualCIFAR100(BaseContinualDataset):
    """CIFAR-100 split into continual learning tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self._build_task_loaders()
    
    def _load_data(self):
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([transforms.ToTensor()])
        
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True,
            transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, download=True,
            transform=self.test_transform
        )
    
    def _create_task_splits(self) -> List[np.ndarray]:
        np.random.seed(self.seed)
        all_classes = np.arange(100)
        np.random.shuffle(all_classes)
        return list(np.array_split(all_classes, self.num_tasks))
    
    def _build_task_loaders(self):
        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = self._select_subset_indices(
                    self.train_dataset, class_subset, self.samples_per_class
                )
                test_indices = self._select_subset_indices(
                    self.test_dataset, class_subset, self.samples_per_class
                )
            else:
                train_indices = [
                    i for i, (_, label) in enumerate(self.train_dataset)
                    if label in class_subset
                ]
                test_indices = [
                    i for i, (_, label) in enumerate(self.test_dataset)
                    if label in class_subset
                ]
            
            train_subset = RemappedSubset(
                self.train_dataset, train_indices,
                self.label_maps[task_id]
            )
            test_subset = RemappedSubset(
                self.test_dataset, test_indices,
                self.label_maps[task_id]
            )
            
            self.task_train_loaders.append(
                DataLoader(train_subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
            )
            self.task_test_loaders.append(
                DataLoader(test_subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
            )


class RemappedSubset(Dataset):
    """Dataset wrapper that remaps labels according to a mapping dict."""
    
    def __init__(self, dataset: Dataset, indices: List[int], label_map: Dict[int, int]):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        return data, self.label_map.get(int(label), int(label))


class TextDataset(Dataset):
    """Dataset wrapper for vectorized text data."""
    
    def __init__(self, data_vectors: List[torch.Tensor], targets: np.ndarray):
        self.data = list(data_vectors)
        self.targets = np.array(targets, dtype=int)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


class Continual20Newsgroups(BaseContinualDataset):
    """20 Newsgroups dataset for continual learning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self._build_task_loaders()
    
    def _load_data(self):
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        train_raw = fetch_20newsgroups(subset="train", remove=())
        test_raw = fetch_20newsgroups(subset="test", remove=())
        
        vectorizer = TfidfVectorizer(max_features=20000)
        train_vectors = vectorizer.fit_transform(train_raw.data)
        test_vectors = vectorizer.transform(test_raw.data)
        
        train_tensors = [
            torch.from_numpy(train_vectors[i].toarray().squeeze().astype(np.float32))
            for i in range(train_vectors.shape[0])
        ]
        test_tensors = [
            torch.from_numpy(test_vectors[i].toarray().squeeze().astype(np.float32))
            for i in range(test_vectors.shape[0])
        ]
        
        self.train_dataset = TextDataset(train_tensors, train_raw.target)
        self.test_dataset = TextDataset(test_tensors, test_raw.target)
    
    def _create_task_splits(self) -> List[np.ndarray]:
        np.random.seed(self.seed)
        all_classes = np.arange(20)
        return list(np.array_split(all_classes, self.num_tasks))
    
    def _build_task_loaders(self):
        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = self._select_subset_indices(
                    self.train_dataset, class_subset, self.samples_per_class
                )
                test_indices = self._select_subset_indices(
                    self.test_dataset, class_subset, self.samples_per_class
                )
            else:
                train_indices = [
                    i for i, (_, label) in enumerate(self.train_dataset)
                    if int(label) in class_subset
                ]
                test_indices = [
                    i for i, (_, label) in enumerate(self.test_dataset)
                    if int(label) in class_subset
                ]
            
            train_subset = RemappedSubset(
                self.train_dataset, train_indices,
                self.label_maps[task_id]
            )
            test_subset = RemappedSubset(
                self.test_dataset, test_indices,
                self.label_maps[task_id]
            )
            
            self.task_train_loaders.append(
                DataLoader(train_subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
            )
            self.task_test_loaders.append(
                DataLoader(test_subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
            )


class NodeDataset(Dataset):
    """Dataset for graph node classification."""
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])


class ContinualCora(BaseContinualDataset):
    """Cora dataset for continual learning (node classification)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()
        self.task_splits = self._create_task_splits()
        self.label_maps = self._build_label_maps()
        self._build_task_loaders()
    
    def _load_data(self):
        from torch_geometric.datasets import Planetoid
        
        cora = Planetoid(root=self.data_dir, name="Cora")
        self.graph_data = cora[0]
        self.full_node_dataset = NodeDataset(
            self.graph_data.x.clone(),
            self.graph_data.y.clone()
        )
        
        # Split nodes into train/test
        n_nodes = self.graph_data.num_nodes
        np.random.seed(self.seed)
        perm = np.random.permutation(n_nodes)
        split = int(0.8 * n_nodes)
        self.train_node_indices = perm[:split]
        self.test_node_indices = perm[split:]
    
    def _create_task_splits(self) -> List[np.ndarray]:
        np.random.seed(self.seed)
        all_classes = np.arange(7)  # Cora has 7 classes
        return list(np.array_split(all_classes, self.num_tasks))
    
    def _build_task_loaders(self):
        for task_id, class_subset in enumerate(self.task_splits):
            if self.debug:
                train_indices = []
                class_counts = {cls: 0 for cls in class_subset}
                for idx in self.train_node_indices:
                    label = int(self.full_node_dataset[idx][1])
                    if label in class_subset and class_counts[label] < self.samples_per_class:
                        train_indices.append(int(idx))
                        class_counts[label] += 1
                
                test_indices = []
                class_counts = {cls: 0 for cls in class_subset}
                for idx in self.test_node_indices:
                    label = int(self.full_node_dataset[idx][1])
                    if label in class_subset and class_counts[label] < self.samples_per_class:
                        test_indices.append(int(idx))
                        class_counts[label] += 1
            else:
                train_indices = [
                    int(idx) for idx in self.train_node_indices
                    if int(self.full_node_dataset[idx][1]) in class_subset
                ]
                test_indices = [
                    int(idx) for idx in self.test_node_indices
                    if int(self.full_node_dataset[idx][1]) in class_subset
                ]
            
            # Create separate node datasets for each task
            task_train_dataset = NodeDataset(
                self.full_node_dataset.x.clone(),
                self.full_node_dataset.y.clone()
            )
            task_test_dataset = NodeDataset(
                self.full_node_dataset.x.clone(),
                self.full_node_dataset.y.clone()
            )
            
            train_subset = RemappedSubset(
                task_train_dataset, train_indices,
                self.label_maps[task_id]
            )
            test_subset = RemappedSubset(
                task_test_dataset, test_indices,
                self.label_maps[task_id]
            )
            
            self.task_train_loaders.append(
                DataLoader(train_subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
            )
            self.task_test_loaders.append(
                DataLoader(test_subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
            )


# Dataset registry
DATASET_REGISTRY = {
    "SplitMNIST": ContinualSplitMNIST,
    "CIFAR100": ContinualCIFAR100,
    "20Newsgroups": Continual20Newsgroups,
    "Cora": ContinualCora,
}


def get_dataset(name: str, **kwargs) -> BaseContinualDataset:
    """Factory function to get dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
