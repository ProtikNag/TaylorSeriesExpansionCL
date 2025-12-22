"""
HTCL: Hierarchical Taylor Series Continual Learning

A modular framework for continual learning experiments with:
- Multi-level hierarchy for knowledge consolidation
- Taylor series-based global model updates
- Global model catch-up mechanism
- Professional visualizations

Usage:
    from htcl import run_hierarchy_experiment, get_mnist_config
    
    config = get_mnist_config(debug=True)
    results = run_hierarchy_experiment(config, hierarchy_levels=[2, 3, 4])
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

# Import main components for convenient access
from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    HTCLConfig,
    get_mnist_config,
    get_cifar100_config,
    get_newsgroups_config,
    get_cora_config,
)

from .data import get_dataset, DATASET_REGISTRY

from .models import get_model, MODEL_REGISTRY

from .methods import (
    run_er_experiments,
    run_ser_experiments,
    train_htcl,
    ReplayBuffer,
    generate_canonical_permutations,
    BASELINE_REGISTRY,
    get_baseline_runner,
)

from .utils import (
    set_seed,
    get_device,
    evaluate,
    evaluate_all_tasks,
    ResultsManager,
)

from .experiments import (
    run_hierarchy_experiment,
    list_baselines,
)

from .visualization import create_all_visualizations

__all__ = [
    # Config
    "ExperimentConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "HTCLConfig",
    "get_mnist_config",
    "get_cifar100_config",
    "get_newsgroups_config",
    "get_cora_config",
    # Data
    "get_dataset",
    "DATASET_REGISTRY",
    # Models
    "get_model",
    "MODEL_REGISTRY",
    # Methods
    "run_er_experiments",
    "run_ser_experiments",
    "train_htcl",
    "ReplayBuffer",
    "generate_canonical_permutations",
    "BASELINE_REGISTRY",
    "get_baseline_runner",
    # Utils
    "set_seed",
    "get_device",
    "evaluate",
    "evaluate_all_tasks",
    "ResultsManager",
    # Experiments
    "run_hierarchy_experiment",
    "list_baselines",
    # Visualization
    "create_all_visualizations",
]
