from .buffer import ReplayBuffer, ClassBalancedBuffer, ReplayDataset
from .er import (
    train_er_single_epoch,
    train_er_on_task,
    train_er_model,
    run_er_experiments,
)
from .ser import (
    train_ser_single_epoch,
    train_ser_on_task,
    train_ser_model,
    run_ser_experiments,
)
from .der import (
    train_der_single_epoch,
    train_der_on_task,
    train_der_model,
    run_der_experiments,
)
from .htcl import (
    HierarchicalModel,
    taylor_update,
    global_catchup,
    select_best_permutation,
    train_htcl,
    generate_canonical_permutations,
)
from .dualnet import (
    train_dualnet_single_epoch,
    train_dualnet_on_task,
    train_dualnet_model,
    run_dualnet_experiments
)
from .spectral_reg import (
    train_spectral_single_epoch,
    train_spectral_on_task,
    train_spectral_model,
    run_spectral_experiments
)
from .fedavg import (
    fedavg_update,
    FedAvgHierarchy,
    train_fedavg,
    run_fedavg_ser_experiments,
    run_fedavg_der_experiments,
)
from .fedprox import (
    fedprox_update,
    FedProxHierarchy,
    proximal_loss,
    train_fedprox,
    run_fedprox_ser_experiments,
    run_fedprox_der_experiments,
)

# Registry of available baseline methods
BASELINE_REGISTRY = {
    "er": run_er_experiments,
    "ser": run_ser_experiments,
    "der": run_der_experiments,
    "dualnet": run_dualnet_experiments,
    "spectral": run_spectral_experiments,
}


def get_baseline_runner(name: str):
    """Get the experiment runner for a baseline method."""
    name_lower = name.lower()
    if name_lower not in BASELINE_REGISTRY:
        available = list(BASELINE_REGISTRY.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    return BASELINE_REGISTRY[name_lower]


__all__ = [
    # Buffer
    "ReplayBuffer",
    "ClassBalancedBuffer",
    "ReplayDataset",
    # ER
    "train_er_single_epoch",
    "train_er_on_task",
    "train_er_model",
    "run_er_experiments",
    # SER
    "train_ser_single_epoch",
    "train_ser_on_task",
    "train_ser_model",
    "run_ser_experiments",
    # DER
    "train_der_single_epoch",
    "train_der_on_task",
    "train_der_model",
    "run_der_experiments",
    # DualNet
    "train_dualnet_single_epoch",
    "train_dualnet_on_task",
    "train_dualnet_model",
    "run_dualnet_experiments",
    # Spectral Regularization
    "train_spectral_single_epoch",
    "train_spectral_on_task",
    "train_spectral_model",
    "run_spectral_experiments",
    # HTCL
    "HierarchicalModel",
    "taylor_update",
    "global_catchup",
    "select_best_permutation",
    "train_htcl",
    "generate_canonical_permutations",
    # Registry
    "BASELINE_REGISTRY",
    "get_baseline_runner",
]
