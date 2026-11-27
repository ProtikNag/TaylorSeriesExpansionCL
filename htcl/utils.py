"""Seeding, device selection, RNG capture, atomic writes and stable hashing."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def stable_hash(obj: Any, length: int = 12) -> str:
    """Return a short sha1 hex digest of a JSON-serialisable object (key order independent)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:length]


def derive_seed(*parts: Any) -> int:
    """Derive a 31-bit integer seed from arbitrary JSON-serialisable parts."""
    return int(stable_hash(list(parts), 8), 16) % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every accelerator); deterministic cuDNN and, on CUDA,
    deterministic algorithms where they exist (others, e.g. torch_geometric scatter, only warn)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda and mps generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read when cuBLAS first initialises
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name: str | None = None) -> torch.device:
    """Pick the requested device, else HTCL_DEVICE, else mps > cuda > cpu."""
    name = name or os.environ.get("HTCL_DEVICE")
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rng_state() -> dict:
    """Capture python, numpy, torch CPU and accelerator RNG states."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG states captured by `rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def atomic_save(obj: Any, path: str | Path) -> None:
    """torch.save to a temporary file, then rename, so a crash never leaves a torn file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: str | Path) -> None:
    """Write JSON atomically (temporary file plus rename)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)
