"""FedAvg / FedProx consolidation for HTCL's `aggregator` flag (task = client).

Each group's selected local model plays the role of a client update and the hierarchical level
plays the global model (McMahan et al. 2017; Li et al. 2020). Ported from the intent of
`origin/federated-comp:htcl/methods/{fedavg,fedprox}.py`, with two deliberate changes:
- the old fixed interpolation W_g <- (1 - alpha) W_g + alpha W_l (alpha = 0.5 / (level + 1)) is
  replaced by a sample-weighted running average across groups (cumulative FedAvg). This is not the
  literal FedAvg rule: each round (group) has a single client, so FedAvg would set w_1 to the selected
  local model, which is the `consolidate: false` ablation (E8 carries it as a reference row). The
  running average keeps the old global model and gives group t weight n_t / (sum of samples so far);
- integer buffers (BN's num_batches_tracked) are copied instead of being averaged as floats.

Callers:
- `htcl.methods.base.Method.optimize` adds `fedprox_penalty(...)` to every local loss when
  HTCL's aggregator is 'fedprox' (it sets `method.prox = (mu, anchor)` before local training).
- `htcl.hierarchy.HTCL` calls `fedavg_update(...)` once per hierarchical level after each group
  for aggregators 'fedavg' and 'fedprox' (level 1 aggregates the selected local model, level i
  aggregates the freshly updated level i-1). BN recalibration happens afterwards in hierarchy.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fedprox_penalty(model: nn.Module, anchor: list[torch.Tensor], mu: float) -> torch.Tensor:
    """FedProx proximal term (mu / 2) * ||w - w_1||^2.

    Args:
        model: the local model being trained.
        anchor: detached copies of w_1's trainable parameters, in `model.parameters()` order
            (only parameters with requires_grad), on the model's device.
        mu: proximal coefficient.

    Returns:
        Scalar tensor that is differentiable w.r.t. the model parameters.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == len(anchor), "anchor does not match the model's trainable parameters"
    return 0.5 * mu * sum((p - a).pow(2).sum() for p, a in zip(params, anchor))


@torch.no_grad()
def fedavg_update(global_model: nn.Module, local_model: nn.Module, seen_samples: int, new_samples: int) -> None:
    """In-place cumulative (running-average) FedAvg of the selected local models, weighted by samples.

    global <- (seen_samples * global + new_samples * local) / (seen_samples + new_samples)
    for every floating-point parameter and buffer (integer buffers such as BN's
    num_batches_tracked are copied from `local_model`).

    Args:
        global_model: hierarchical level being updated (modified in place).
        local_model: model being aggregated into it (not modified).
        seen_samples: training samples already represented in `global_model` (> 0).
        new_samples: training samples of the current group.
    """
    assert seen_samples > 0 and new_samples > 0, (seen_samples, new_samples)
    weight_new = new_samples / (seen_samples + new_samples)
    local_state = local_model.state_dict()
    # state_dict tensors share storage with the module, so in-place ops update global_model.
    for name, value in global_model.state_dict().items():
        if value.is_floating_point():
            value.lerp_(local_state[name].to(value.device), weight_new)  # (1 - w) g + w l
        else:
            value.copy_(local_state[name])
