"""prepare_imagenet100 is idempotent (audit F10) and handles a flat val/: `pytest -q tests/test_imagenet_scripts.py`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_imagenet100 import IMAGENET100_WNIDS, prepare  # noqa: E402


def _fake_imagenet(root: Path) -> Path:
    """One image per class in train/<wnid>/ and a flat val/ plus its LOC_val_solution.csv."""
    rows = ["ImageId,PredictionString"]
    (root / "val").mkdir(parents=True)
    for i, wnid in enumerate(IMAGENET100_WNIDS + ["n01440764"]):  # one extra class that must be ignored
        (root / "train" / wnid).mkdir(parents=True)
        (root / "train" / wnid / "a.JPEG").touch()
        (root / "val" / f"ILSVRC2012_val_{i:08d}.JPEG").touch()
        rows.append(f"ILSVRC2012_val_{i:08d},{wnid} 1 2 3 4")
    (root / "sol.csv").write_text("\n".join(rows) + "\n")
    return root / "sol.csv"


def test_prepare_is_idempotent(tmp_path: Path) -> None:
    """Running twice leaves 100 non-empty linked classes per split (the old code emptied them)."""
    labels = _fake_imagenet(tmp_path / "inet")
    out = tmp_path / "out"
    for _ in range(2):
        prepare(tmp_path / "inet", out, labels)
    for split in ("train", "val"):
        dirs = sorted(p.name for p in (out / split).iterdir())
        assert dirs == sorted(IMAGENET100_WNIDS)
        assert all(any((out / split / w).iterdir()) for w in dirs)
