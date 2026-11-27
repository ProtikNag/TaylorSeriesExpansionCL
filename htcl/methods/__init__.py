"""Method registry: name -> 'module:Class', imported lazily so a missing file only breaks its own name."""

from __future__ import annotations

import importlib

REGISTRY = {
    "finetune": "htcl.methods.finetune:Finetune",
    "er": "htcl.methods.er:ER",
    "der": "htcl.methods.der:DER",
    "ser": "htcl.methods.ser:SER",
    "ewc": "htcl.methods.ewc:EWC",
    "sr": "htcl.methods.sr:SR",
    "icarl": "htcl.methods.icarl:ICaRL",
    "dualnet": "htcl.methods.dualnet:DualNet",
}


def get_method(name: str) -> type:
    """Resolve a registered method class by name."""
    module_name, class_name = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)
