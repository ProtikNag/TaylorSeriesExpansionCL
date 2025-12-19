"""
Configuration classes for HTCL experiments.
Uses dataclasses for clean, type-hinted configuration management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    name: str = "SplitMNIST"
    data_dir: str = "./data"
    num_tasks: int = 5
    batch_size: int = 32
    seed: int = 42
    debug: bool = False
    samples_per_class: int = 60
    
    # Dataset-specific defaults
    @classmethod
    def for_dataset(cls, name: str, debug: bool = False) -> "DataConfig":
        """Factory method for dataset-specific configurations."""
        configs = {
            "SplitMNIST": cls(
                name="SplitMNIST",
                num_tasks=5,
                batch_size=32,
                debug=debug,
                samples_per_class=60
            ),
            "CIFAR100": cls(
                name="CIFAR100",
                num_tasks=10,
                batch_size=32,
                debug=debug,
                samples_per_class=60
            ),
            "Cora": cls(
                name="Cora",
                num_tasks=3,
                batch_size=32,
                debug=debug,
                samples_per_class=60
            ),
            "20Newsgroups": cls(
                name="20Newsgroups",
                num_tasks=5,
                batch_size=32,
                debug=debug,
                samples_per_class=60
            ),
        }
        return configs.get(name, cls(name=name, debug=debug))


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    num_classes_per_task: int = 2
    freeze_backbone: bool = False
    hidden_dim: int = 128
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    num_epochs: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    device: str = "cuda"


@dataclass
class HTCLConfig:
    """Configuration for HTCL-specific parameters."""
    # Hierarchy settings
    num_levels: int = 2  # L-level hierarchy (2 = local + global)
    group_size: int = 2  # k tasks per group for permutation search
    
    # Buffer settings
    buffer_size: int = 100
    buffer_beta: float = 0.5  # weight for replay loss
    
    # Taylor update parameters
    eta: float = 1.0  # step size for Taylor update
    max_norm: float = 1.0  # gradient clipping
    lambda_reg: float = 1000.0  # regularization toward local model
    
    # Global model catch-up settings
    catchup_enabled: bool = True
    catchup_epochs: int = 2  # extra epochs for global model on recent tasks
    catchup_lr_factor: float = 0.1  # reduced LR for catch-up phase
    
    # Number of permutations to sample (if None, use all k!)
    num_permutations: Optional[int] = 20  # Default to 20 canonical permutations


@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    htcl: HTCLConfig = field(default_factory=HTCLConfig)
    
    # Experiment settings
    experiment_name: str = "htcl_experiment"
    output_dir: str = "./results"
    save_models: bool = True
    verbose: bool = True
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "htcl": self.htcl.__dict__,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "save_models": self.save_models,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            htcl=HTCLConfig(**data.get("htcl", {})),
            experiment_name=data.get("experiment_name", "htcl_experiment"),
            output_dir=data.get("output_dir", "./results"),
            save_models=data.get("save_models", True),
            verbose=data.get("verbose", True),
        )


# Predefined experiment configurations
def get_mnist_config(debug: bool = False, num_levels: int = 2) -> ExperimentConfig:
    """Get configuration for SplitMNIST experiments."""
    return ExperimentConfig(
        data=DataConfig.for_dataset("SplitMNIST", debug=debug),
        model=ModelConfig(num_classes_per_task=2),
        training=TrainingConfig(num_epochs=5, learning_rate=0.01),
        htcl=HTCLConfig(
            num_levels=num_levels,
            group_size=2,
            buffer_size=50,
            catchup_enabled=True,
            catchup_epochs=2
        ),
        experiment_name=f"mnist_L{num_levels}",
    )


def get_cifar100_config(debug: bool = False, num_levels: int = 2) -> ExperimentConfig:
    """Get configuration for CIFAR-100 experiments."""
    return ExperimentConfig(
        data=DataConfig.for_dataset("CIFAR100", debug=debug),
        model=ModelConfig(num_classes_per_task=10),
        training=TrainingConfig(num_epochs=5, learning_rate=0.1),
        htcl=HTCLConfig(
            num_levels=num_levels,
            group_size=2,
            buffer_size=500,
            catchup_enabled=True,
            catchup_epochs=3
        ),
        experiment_name=f"cifar100_L{num_levels}",
    )


def get_newsgroups_config(debug: bool = False, num_levels: int = 2) -> ExperimentConfig:
    """Get configuration for 20Newsgroups experiments."""
    return ExperimentConfig(
        data=DataConfig.for_dataset("20Newsgroups", debug=debug),
        model=ModelConfig(num_classes_per_task=4),
        training=TrainingConfig(num_epochs=5, learning_rate=1e-3),
        htcl=HTCLConfig(
            num_levels=num_levels,
            group_size=2,
            buffer_size=100,
            catchup_enabled=True,
            catchup_epochs=2
        ),
        experiment_name=f"newsgroups_L{num_levels}",
    )


def get_cora_config(debug: bool = False, num_levels: int = 2) -> ExperimentConfig:
    """Get configuration for Cora experiments."""
    return ExperimentConfig(
        data=DataConfig.for_dataset("Cora", debug=debug),
        model=ModelConfig(num_classes_per_task=3),
        training=TrainingConfig(num_epochs=5, learning_rate=0.01),
        htcl=HTCLConfig(
            num_levels=num_levels,
            group_size=2,
            buffer_size=50,
            catchup_enabled=True,
            catchup_epochs=2
        ),
        experiment_name=f"cora_L{num_levels}",
    )
