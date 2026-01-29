"""
Methods subpackage for HTCL.

Contains all continual learning baseline implementations and the HTCL framework.
"""

from .buffer import ReplayBuffer, ClassBalancedBuffer, ReplayDataset
from .er import (
    train_er_model,
    train_er_on_task,
    train_er_single_epoch,
    run_er_experiments,
)
from .ser import (
    train_ser_model,
    train_ser_on_task,
    train_ser_single_epoch,
    run_ser_experiments,
)
from .der import (
    train_der_model,
    train_der_single_epoch,
    run_der_experiments,
    run_derpp_experiments,
)
from .ewc import (
    train_ewc_model,
    train_ewc_on_task,
    train_ewc_single_epoch,
    run_ewc_experiments,
    compute_fisher_information,
    ewc_penalty,
)
from .icarl import (
    train_icarl_model,
    train_icarl_on_task,
    train_icarl_single_epoch,
    run_icarl_experiments,
    ExemplarManager,
    distillation_loss,
)
from .htcl import (
    train_htcl,
    taylor_update,
    global_catchup,
    select_best_permutation,
    generate_canonical_permutations,
    HierarchicalModel,
    BASELINE_TRAIN_REGISTRY,
    get_baseline_train_fn,
    list_available_baselines,
)

# Registry mapping baseline names to their experiment runner functions
BASELINE_REGISTRY = {
    "er": run_er_experiments,
    "ser": run_ser_experiments,
    "der": run_der_experiments,
    "ewc": run_ewc_experiments,
    "icarl": run_icarl_experiments,
}

__all__ = [
    # Buffer
    "ReplayBuffer",
    "ClassBalancedBuffer",
    "ReplayDataset",
    # ER
    "train_er_model",
    "train_er_on_task",
    "train_er_single_epoch",
    "run_er_experiments",
    # SER
    "train_ser_model",
    "train_ser_on_task",
    "train_ser_single_epoch",
    "run_ser_experiments",
    # DER
    "train_der_model",
    "train_der_single_epoch",
    "run_der_experiments",
    "run_derpp_experiments",
    # EWC
    "train_ewc_model",
    "train_ewc_on_task",
    "train_ewc_single_epoch",
    "run_ewc_experiments",
    "compute_fisher_information",
    "ewc_penalty",
    # iCaRL
    "train_icarl_model",
    "train_icarl_on_task",
    "train_icarl_single_epoch",
    "run_icarl_experiments",
    "ExemplarManager",
    "distillation_loss",
    # HTCL
    "train_htcl",
    "taylor_update",
    "global_catchup",
    "select_best_permutation",
    "generate_canonical_permutations",
    "HierarchicalModel",
    "BASELINE_TRAIN_REGISTRY",
    "get_baseline_train_fn",
    "list_available_baselines",
    # Registry
    "BASELINE_REGISTRY",
]
